"""Loopback web API with origin/CSRF checks and static replay reports."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import hmac
import json
import mimetypes

from .reports import render_report
from .storage import ROOT


def create_server(service, host="127.0.0.1", port=8766):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[1] if len(args) > 1 else "").startswith("5"):
                super().log_message(fmt, *args)

        def reply(self, status, body, content_type="application/json; charset=utf-8", extra=None):
            data = body if isinstance(body, bytes) else (body.encode("utf-8") if isinstance(body, str) else json.dumps(body, allow_nan=False).encode())
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                raise ValueError("Unrecognized Host")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in allowed}:
                raise ValueError("Cross-origin requests are not allowed")

        def do_GET(self):
            try:
                self.local_request()
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/api/state":
                    return self.reply(200, service.state())
                if path == "/api/events":
                    after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                    return self.reply(200, service.since(max(0, after)))
                if path.startswith("/api/runs/"):
                    parts = path.split("/")
                    if len(parts) not in {4, 5}:
                        raise FileNotFoundError()
                    run = service.store.read_run(parts[3])
                    if len(parts) == 5 and parts[4] == "report":
                        historical = service.store.run_path(parts[3]) / "report.html"
                        html = historical.read_text(encoding="utf-8") if run["historical"] and historical.exists() else render_report(run)
                        return self.reply(200, html, "text/html; charset=utf-8")
                    if len(parts) == 5 and parts[4] == "export":
                        return self.reply(200, run, extra={"Content-Disposition": f'attachment; filename="{parts[3]}.json"'})
                    if len(parts) != 4:
                        raise FileNotFoundError()
                    return self.reply(200, run)
                if path in {"/api/guide", "/guide"}:
                    return self.reply(302, "", extra={"Location": "/docs/guide.html"})
                elif path in {"/plan", "/LAB_PLAN.html"}:
                    file = ROOT / "LAB_PLAN.html"
                elif path == "/findings":
                    return self.reply(302, "", extra={"Location": "/docs/results.html"})
                elif path.startswith(("/docs/", "/studies/")):
                    area = ROOT / path.split("/")[1]
                    file = (ROOT / path.lstrip("/")).resolve()
                    if not file.is_relative_to(area.resolve()) or file.suffix not in {".html", ".json", ".md", ".svg", ".png", ".gz", ".npz"}:
                        raise FileNotFoundError()
                else:
                    name = "index.html" if path == "/" else path.lstrip("/")
                    if name not in {"index.html", "app.js", "style.css", "favicon.svg"}:
                        raise FileNotFoundError()
                    file = ROOT / "lab" / "static" / name
                mime = mimetypes.guess_type(file.name)[0] or "text/plain"
                return self.reply(200, file.read_bytes(), mime + ("; charset=utf-8" if mime.startswith("text/") or mime == "application/javascript" else ""))
            except FileNotFoundError:
                self.reply(404, dict(error="Not found"))
            except (ValueError, TypeError) as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception as exc:
                self.reply(500, dict(error=f"Server error: {exc}"))

        def do_POST(self):
            try:
                self.local_request()
                if self.path != "/api/command":
                    return self.reply(404, dict(error="Not found"))
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    return self.reply(415, dict(error="Use application/json"))
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1048576:
                    return self.reply(413, dict(error="Invalid request size"))
                body = json.loads(self.rfile.read(size), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
                if not isinstance(body, dict) or not isinstance(body.get("csrf"), str) or not hmac.compare_digest(body["csrf"], service.csrf):
                    return self.reply(403, dict(error="Refresh the page to renew its control token"))
                result = service.command(body.get("command"), body.get("payload", {}))
                self.reply(202, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception as exc:
                self.reply(500, dict(error=f"Command failed: {exc}"))
    return ThreadingHTTPServer((host, port), Handler)
