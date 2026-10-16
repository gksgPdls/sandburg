"""
탐지기 웹 화면 (내 컴퓨터에서만 동작, 이미지는 밖으로 나가지 않음)

    cd ~/dev/repo
    .venv/bin/python -m detector.web          # 브라우저가 http://localhost:8765 로 열림
    .venv/bin/python -m detector.web --port 9000 --no-browser

추가 설치 없음 (파이썬 기본 웹서버 사용). 자동 탐색은 tesseract가 있어야 동작.
"""
import argparse
import base64
import hashlib
import io
import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from ..core import load_config
from .analysis import analyze

HERE = Path(__file__).resolve().parent
IMAGES = {}  # 업로드한 이미지 (메모리에만 보관)
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 조용히
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/config":
            cfg = load_config()
            return self._send(200, {k: {"name": m["name"], "enabled": m["enabled"]} for k, m in cfg["methods"].items()}
                              | {"_verdict": cfg["verdict"]})
        self._send(404, {"error": "없는 주소"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/upload":
                raw = base64.b64decode(req["image"].split(",", 1)[-1])
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                key = hashlib.sha1(raw).hexdigest()[:12]
                with LOCK:
                    IMAGES.clear()
                    IMAGES[key] = img
                return self._send(200, {"id": key, "width": img.width, "height": img.height})
            img = IMAGES.get(req.get("id"))
            if img is None:
                return self._send(400, {"error": "이미지를 다시 올려주세요 (서버가 재시작됨)"})
            if self.path == "/api/analyze":
                x, y, w, h = [int(round(v)) for v in req["bbox"]]
                if w < 6 or h < 6:
                    return self._send(400, {"error": "영역이 너무 작아요. 숫자 칸을 넉넉히 감싸서 다시 그려주세요."})
                return self._send(200, analyze(img, (x, y, w, h)))
            if self.path == "/api/locate":
                from ..locate import locate
                try:
                    r = locate(img)
                except Exception as e:  # noqa: BLE001
                    return self._send(500, {"error": f"자동 탐색에 OCR(tesseract)이 필요해요: {e}"})
                return self._send(200, {"suspicious": r["suspicious"], "summary": r["summary"],
                                        "doc_max_rarity": r["doc_max_rarity"],
                                        "candidates": [{"text": c["text"], "bbox": list(c["bbox"]), "rarity": c["rarity"],
                                                        "votes": c["votes"], "flagged_methods": c["flagged_methods"]}
                                                       for c in r["candidates"]]})
            self._send(404, {"error": "없는 주소"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"분석 중 오류: {e}"})


def main():
    ap = argparse.ArgumentParser(prog="python -m detector.web")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"탐지기 화면: {url}  (끄려면 Ctrl+C)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("종료")


if __name__ == "__main__":
    main()
