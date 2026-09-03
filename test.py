import importlib.util
import json
import pathlib
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "newapi_stream_load_test.py"


def load_script():
    if not SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("newapi_stream_load_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_test = load_script()


class NewApiStreamLoadTest(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT.exists(), "压测脚本尚未实现")

    @unittest.skipIf(load_test is None, "压测脚本尚未实现")
    def test_default_url_targets_the_newapi_entrypoint(self):
        self.assertEqual(
            getattr(load_test, "DEFAULT_URL", None),
            "https://10.255.87.120:3000/v1/chat/completions",
        )

    @unittest.skipIf(load_test is None, "压测脚本尚未实现")
    def test_api_key_is_normalized_without_duplicate_bearer_prefix(self):
        self.assertEqual(load_test.normalize_api_key("abc123"), "Bearer abc123")
        self.assertEqual(load_test.normalize_api_key("Bearer abc123"), "Bearer abc123")

    @unittest.skipIf(load_test is None, "压测脚本尚未实现")
    def test_payload_uses_streaming_and_short_response(self):
        payload = json.loads(load_test.build_payload("GLM-5.2", "只回复 OK", 8))
        self.assertEqual(payload["model"], "GLM-5.2")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(payload["messages"][0]["content"], "只回复 OK")

    @unittest.skipIf(load_test is None, "压测脚本尚未实现")
    def test_summary_separates_429_and_calculates_latency_percentiles(self):
        results = [
            load_test.RequestResult("r1", 200, 10.0, 20.0, None, ""),
            load_test.RequestResult("r2", 429, 20.0, 40.0, None, "limited"),
            load_test.RequestResult("r3", 429, 30.0, 60.0, None, "limited"),
            load_test.RequestResult("r4", None, None, 80.0, "timeout", ""),
        ]

        summary = load_test.summarize_results(results)

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["status_counts"], {"200": 1, "429": 2, "error": 1})
        self.assertEqual(summary["rate_429_percent"], 50.0)
        self.assertEqual(summary["ttfb_ms"]["p50"], 20.0)
        self.assertEqual(summary["total_ms"]["p95"], 77.0)

    @unittest.skipIf(load_test is None, "压测脚本尚未实现")
    def test_request_records_gateway_429_response(self):
        class RateLimitedHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Envoy-Ratelimited", "true")
                self.end_headers()
                self.wfile.write(b'{"error":"rate limited"}')

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), RateLimitedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = load_test.send_request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                "Bearer test",
                load_test.build_payload("GLM-5.2", "只回复 OK", 8),
                "request-429",
                2.0,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(result.status, 429)
        self.assertEqual(result.envoy_ratelimited, "true")
        self.assertIn("rate limited", result.response_excerpt)

    @unittest.skipIf(load_test is None, "压测脚本尚未实现")
    def test_stream_ttfb_is_recorded_from_the_first_byte(self):
        class DelayedBulkReadResponse:
            def __init__(self):
                self.calls = 0

            def read(self, amount):
                self.calls += 1
                if self.calls > 1:
                    return b""
                if amount != 1:
                    time.sleep(0.05)
                    return b"data: complete\n\n"
                return b"d"

        started_at = time.monotonic()
        first_byte_at, body = load_test.read_response(DelayedBulkReadResponse())

        self.assertLess(first_byte_at - started_at, 0.03)
        self.assertEqual(body, "d")


if __name__ == "__main__":
    unittest.main()
