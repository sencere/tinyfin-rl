#!/usr/bin/env python3
import argparse
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class DashboardHandler(SimpleHTTPRequestHandler):
    metrics_path = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = []
            if self.metrics_path and os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            self.wfile.write(json.dumps({"records": data}).encode("utf-8"))
            return
        return super().do_GET()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="runs/ppo_metrics.jsonl")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", type=str, default="dashboard")
    args = parser.parse_args()

    os.chdir(args.root)
    DashboardHandler.metrics_path = os.path.abspath(os.path.join(os.getcwd(), "..", args.metrics))
    httpd = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"dashboard: http://{args.host}:{args.port}")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
