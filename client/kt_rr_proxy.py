#!/usr/bin/env python3
"""Small, auditable round-robin proxy for the KT ATOM+ experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class ForwardLogger:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = asyncio.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, message: str, record: dict | None = None) -> None:
        line = f"[{now_iso()}] {message}"
        if self.path:
            async with self._lock:
                with self.path.open("a", encoding="utf-8") as target:
                    target.write(json.dumps(
                        record or {"message": message}, ensure_ascii=False) + "\n")
        try:
            print(line, flush=True)
        except (BrokenPipeError, OSError):
            pass


class RoundRobinRouter:
    def __init__(self, endpoints: Iterable[str]):
        self.endpoints = [endpoint.rstrip("/") for endpoint in endpoints]
        if len(self.endpoints) < 2:
            raise ValueError("at least two endpoints are required")
        self._sequence = 0
        self._lock = asyncio.Lock()
        self.forwarded = [0 for _ in self.endpoints]

    async def select(self, requested_id: str | None = None) -> tuple[int, int, str]:
        async with self._lock:
            sequence = self._sequence
            self._sequence += 1
            try:
                request_id = int(requested_id) if requested_id is not None else sequence
            except ValueError:
                request_id = sequence
            endpoint_index = request_id % len(self.endpoints)
            self.forwarded[endpoint_index] += 1
            return request_id, endpoint_index, self.endpoints[endpoint_index]


def filtered_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def create_app(endpoints: list[str], log_file: Path | None = None,
               upstream_timeout: float = 900.0) -> Starlette:
    router = RoundRobinRouter(endpoints)
    forward_logger = ForwardLogger(log_file)
    timeout = httpx.Timeout(upstream_timeout, connect=30.0,
                            read=upstream_timeout, write=30.0)
    client = httpx.AsyncClient(timeout=timeout,
                               limits=httpx.Limits(max_connections=1000,
                                                   max_keepalive_connections=200))

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "algorithm": "round_robin"})

    async def status(_: Request) -> Response:
        return JSONResponse({
            "algorithm": "round_robin",
            "endpoints": router.endpoints,
            "forwarded": router.forwarded,
            "total_forwarded": sum(router.forwarded),
        })

    async def set_algorithm(request: Request) -> Response:
        payload = await request.json()
        algorithm = payload.get("algorithm", "round_robin")
        if algorithm not in (1, "round_robin"):
            return JSONResponse({"error": "this proxy supports round_robin only"},
                                status_code=400)
        return JSONResponse({"algorithm": "round_robin", "status": "configured"})

    async def finalize(_: Request) -> Response:
        result = {
            "algorithm": "round_robin",
            "forwarded": router.forwarded,
            "total_forwarded": sum(router.forwarded),
        }
        await forward_logger.write(
            "FINALIZE total={} per_endpoint={}".format(
                result["total_forwarded"], result["forwarded"]), result)
        return JSONResponse(result)

    async def forward(request: Request) -> Response:
        request_id, endpoint_index, endpoint = await router.select(
            request.headers.get("x-bystander-request-id"))
        target_url = f"{endpoint}/v1/chat/completions"
        record = {
            "timestamp": now_iso(),
            "event": "forward",
            "request_id": request_id,
            "rr_slot": endpoint_index,
            "endpoint_index": endpoint_index,
            "endpoint": endpoint,
            "target_url": target_url,
        }
        await forward_logger.write(
            f"FORWARD request_id={request_id:06d} rr_slot={endpoint_index} "
            f"endpoint_index={endpoint_index} endpoint={endpoint}", record)
        body = await request.body()
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        try:
            upstream_request = client.build_request(
                "POST", target_url, content=body, headers=headers)
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            await forward_logger.write(
                f"ERROR request_id={request_id:06d} endpoint_index={endpoint_index} "
                f"error={exc}")
            return JSONResponse({"error": str(exc), "request_id": request_id},
                                status_code=502)

        response_headers = filtered_headers(upstream.headers)
        response_headers.update({
            "X-Bystander-Request-ID": str(request_id),
            "X-Bystander-Endpoint-Index": str(endpoint_index),
            "X-Bystander-Endpoint": endpoint,
        })
        content_type = upstream.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return StreamingResponse(
                upstream.aiter_raw(), status_code=upstream.status_code,
                headers=response_headers, background=BackgroundTask(upstream.aclose))
        content = await upstream.aread()
        await upstream.aclose()
        return Response(content, status_code=upstream.status_code,
                        headers=response_headers)

    @asynccontextmanager
    async def lifespan(_: Starlette):
        await forward_logger.write(
            "RR_PROXY_STARTED algorithm=round_robin endpoints="
            + json.dumps(router.endpoints, ensure_ascii=False),
            {"event": "startup", "algorithm": "round_robin",
             "endpoints": router.endpoints})
        yield
        await client.aclose()
        await forward_logger.write("RR_PROXY_STOPPED")

    app = Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/routing_status", status, methods=["GET"]),
        Route("/set_algorithm", set_algorithm, methods=["POST"]),
        Route("/finalize", finalize, methods=["POST"]),
        Route("/v1/chat/completions", forward, methods=["POST"]),
    ], lifespan=lifespan)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screenshot-friendly KT ATOM+ RR proxy")
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--upstream-timeout", type=float, default=900.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(args.endpoints, args.log_file, args.upstream_timeout)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                access_log=False)


if __name__ == "__main__":
    main()
