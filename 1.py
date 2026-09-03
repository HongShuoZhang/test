#!/usr/bin/env python3
"""Send sustained concurrent streaming chat requests through NewAPI."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import datetime as dt
import itertools
import json
import os
import pathlib
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Iterable, Optional


# ===== 直接修改这里即可 =====
DEFAULT_URL = "https://10.255.87.120:3000/v1/chat/completions"
DEFAULT_API_KEY = ""  # 填写 NewAPI 客户端 Token；可带或不带 "Bearer " 前缀。
TLS_CONTEXT = ssl.create_default_context()
TLS_CONTEXT.check_hostname = False
TLS_CONTEXT.verify_mode = ssl.CERT_NONE  # 自签名证书：等同 curl -k，仅用于内网测试。


@dataclasses.dataclass(frozen=True)
class RequestResult:
    request_id: str
    status: Optional[int]
    ttfb_ms: Optional[float]
    total_ms: float
    error: Optional[str]
    response_excerpt: str
    retry_after: str = ""
    envoy_ratelimited: str = ""


def normalize_api_key(api_key: str) -> str:
    value = api_key.strip()
    if not value:
        raise ValueError("NEWAPI_API_KEY 不能为空")
    if value.lower().startswith("bearer "):
        return "Bearer " + value[7:].strip()
    return "Bearer " + value


def build_payload(model: str, prompt: str, max_tokens: int) -> bytes:
    return json.dumps(
        {
            "model": model,
            "stream": True,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def percentile(values: Iterable[float], percent: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(float(ordered[0]), 3)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(float(value), 3)


def latency_summary(values: Iterable[Optional[float]]) -> dict:
    present = [value for value in values if value is not None]
    return {
        "p50": percentile(present, 50),
        "p95": percentile(present, 95),
        "p99": percentile(present, 99),
        "max": round(max(present), 3) if present else None,
    }


def summarize_results(results: Iterable[RequestResult]) -> dict:
    result_list = list(results)
    counts = collections.Counter(
        str(result.status) if result.status is not None else "error"
        for result in result_list
    )
    total = len(result_list)
    count_429 = counts.get("429", 0)
    return {
        "total": total,
        "status_counts": dict(sorted(counts.items())),
        "rate_429_percent": round(count_429 * 100.0 / total, 3) if total else 0.0,
        "ttfb_ms": latency_summary(result.ttfb_ms for result in result_list),
        "total_ms": latency_summary(result.total_ms for result in result_list),
    }


class ResultStore:
    def __init__(self, output_path: pathlib.Path):
        self._lock = threading.Lock()
        self._results: list[RequestResult] = []
        self._output = output_path.open("w", encoding="utf-8")

    def record(self, result: RequestResult) -> None:
        with self._lock:
            self._results.append(result)
            self._output.write(
                json.dumps(dataclasses.asdict(result), ensure_ascii=False) + "\n"
            )

    def snapshot(self) -> list[RequestResult]:
        with self._lock:
            return list(self._results)

    def close(self) -> None:
        with self._lock:
            self._output.close()


def read_response(response, capture_limit: int = 2048) -> tuple[Optional[float], str]:
    captured = bytearray()
    first_byte = response.read(1)
    if not first_byte:
        return None, ""
    first_byte_at = time.monotonic()
    if capture_limit:
        captured.extend(first_byte[:capture_limit])
    while True:
        chunk = response.read(8192)
        if not chunk:
            break
        if len(captured) < capture_limit:
            captured.extend(chunk[: capture_limit - len(captured)])
    return first_byte_at, captured.decode("utf-8", errors="replace")


def send_request(
    url: str,
    authorization: str,
    payload: bytes,
    request_id: str,
    timeout: float,
) -> RequestResult:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "newapi-stream-load-test/1.0",
            "X-Request-ID": request_id,
        },
        method="POST",
    )
    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=TLS_CONTEXT,
        ) as response:
            first_byte_at, _ = read_response(response, capture_limit=0)
            finished_at = time.monotonic()
            return RequestResult(
                request_id=request_id,
                status=response.status,
                ttfb_ms=(first_byte_at - started_at) * 1000 if first_byte_at else None,
                total_ms=(finished_at - started_at) * 1000,
                error=None,
                response_excerpt="",
                retry_after=response.headers.get("Retry-After", ""),
                envoy_ratelimited=response.headers.get("X-Envoy-Ratelimited", ""),
            )
    except urllib.error.HTTPError as exc:
        try:
            received_at = time.monotonic()
            body = exc.read(2048).decode("utf-8", errors="replace")
            finished_at = time.monotonic()
            return RequestResult(
                request_id=request_id,
                status=exc.code,
                ttfb_ms=(received_at - started_at) * 1000,
                total_ms=(finished_at - started_at) * 1000,
                error=None,
                response_excerpt=body,
                retry_after=exc.headers.get("Retry-After", "") if exc.headers else "",
                envoy_ratelimited=(
                    exc.headers.get("X-Envoy-Ratelimited", "") if exc.headers else ""
                ),
            )
        finally:
            exc.close()
    except Exception as exc:  # Network/TLS/timeout failures must be included in the report.
        finished_at = time.monotonic()
        return RequestResult(
            request_id=request_id,
            status=None,
            ttfb_ms=None,
            total_ms=(finished_at - started_at) * 1000,
            error=f"{type(exc).__name__}: {exc}",
            response_excerpt="",
        )


def status_counts(results: Iterable[RequestResult]) -> collections.Counter:
    return collections.Counter(
        str(result.status) if result.status is not None else "error" for result in results
    )


def run_load_test(args: argparse.Namespace, authorization: str) -> tuple[dict, pathlib.Path]:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = pathlib.Path(args.output or f"newapi-load-{timestamp}.jsonl")
    payload = build_payload(args.model, args.prompt, args.max_tokens)
    store = ResultStore(output_path)
    request_numbers = itertools.count(1)
    start_event = threading.Event()
    stop_event = threading.Event()
    run_id = f"newapi-load-{timestamp}"
    started_at = time.monotonic()
    deadline = started_at + args.duration

    def worker(worker_number: int) -> None:
        start_event.wait()
        while not stop_event.is_set() and time.monotonic() < deadline:
            sequence = next(request_numbers)
            request_id = f"{run_id}-{worker_number:03d}-{sequence:07d}"
            store.record(
                send_request(
                    args.url,
                    authorization,
                    payload,
                    request_id,
                    args.timeout,
                )
            )

    print(
        f"开始压测: concurrency={args.concurrency} duration={args.duration}s "
        f"model={args.model} stream=true"
    )
    print(f"请求地址: {args.url}")
    if args.url.startswith("https://"):
        print("警告: 已跳过 TLS 证书校验（等同 curl -k，仅用于内网测试）")
    print(f"明细文件: {output_path}")

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency,
            thread_name_prefix="newapi-load",
        ) as executor:
            futures = [executor.submit(worker, index) for index in range(args.concurrency)]
            start_event.set()
            previous_total = 0
            while time.monotonic() < deadline:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                snapshot = store.snapshot()
                counts = status_counts(snapshot)
                completed = len(snapshot)
                elapsed = min(args.duration, time.monotonic() - started_at)
                print(
                    f"[{elapsed:6.1f}s] 本秒完成={completed - previous_total} "
                    f"累计={completed} 200={counts.get('200', 0)} "
                    f"429={counts.get('429', 0)} error={counts.get('error', 0)}",
                    flush=True,
                )
                previous_total = completed
            print("已停止发起新请求，等待在途流式请求结束……", flush=True)
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        print("\n收到中断信号，停止发起新请求并等待在途请求结束……")
        stop_event.set()
        start_event.set()
    finally:
        stop_event.set()
        store.close()

    results = store.snapshot()
    return summarize_results(results), output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 NewAPI 持续发送 OpenAI 兼容的流式并发请求并统计 429。"
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("NEWAPI_URL", DEFAULT_URL),
        help=f"完整接口地址；默认 {DEFAULT_URL}，也可设置 NEWAPI_URL",
    )
    parser.add_argument("--model", default="GLM-5.2", help="请求模型名")
    parser.add_argument("--concurrency", type=int, default=100, help="并发 worker 数")
    parser.add_argument("--duration", type=float, default=60.0, help="持续发起请求的秒数")
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时秒数")
    parser.add_argument("--max-tokens", type=int, default=8, help="单请求最大输出 Token")
    parser.add_argument("--prompt", default="只回复 OK", help="测试提示词")
    parser.add_argument("--output", help="JSONL 明细路径；默认使用带时间戳的文件名")
    args = parser.parse_args()
    if not args.url:
        parser.error("必须通过 --url 或 NEWAPI_URL 指定 NewAPI 接口地址")
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url 必须以 http:// 或 https:// 开头")
    if args.concurrency <= 0 or args.duration <= 0 or args.timeout <= 0:
        parser.error("--concurrency、--duration 和 --timeout 必须大于 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens 必须大于 0")
    return args


def print_summary(summary: dict, output_path: pathlib.Path) -> None:
    print("\n压测结果")
    print(f"  总请求数: {summary['total']}")
    print(f"  状态码: {json.dumps(summary['status_counts'], ensure_ascii=False)}")
    print(f"  429 比例: {summary['rate_429_percent']}%")
    print(f"  首字节延迟(ms): {json.dumps(summary['ttfb_ms'], ensure_ascii=False)}")
    print(f"  总耗时(ms): {json.dumps(summary['total_ms'], ensure_ascii=False)}")
    print(f"  请求明细: {output_path}")


def main() -> int:
    args = parse_args()
    try:
        authorization = normalize_api_key(
            os.environ.get("NEWAPI_API_KEY", DEFAULT_API_KEY)
        )
    except ValueError as exc:
        raise SystemExit(
            f"错误: {exc}；请填写脚本顶部 DEFAULT_API_KEY，"
            "或设置 NEWAPI_API_KEY 环境变量"
        ) from exc
    summary, output_path = run_load_test(args, authorization)
    print_summary(summary, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
