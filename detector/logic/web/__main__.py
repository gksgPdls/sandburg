"""
로직 탐지기 실험 화면 (이 맥에서만 도는 작은 서버). 사진은 밖으로 나가지 않는다.

    python -m logic_detector.web            # http://localhost:8766 이 열림
    python -m logic_detector.web --port 9000 --no-open

  GET  /                  화면 (index.html)
  GET  /api/samples       demo/samples 의 사진 목록
  GET  /samples/<파일>     샘플 사진 (변조본)
  GET  /originals/<파일>   샘플의 원본 사진 (demo/originals, 나란히 비교용)
  GET  /api/answers       demo/정답.csv (조작 방식·바뀐 곳 좌표·현재 탐지기 결과)
  POST /api/inspect       본문 = 이미지 바이트 → logic_detector.logic.inspect 결과 JSON
"""
import argparse
import csv
import json
import mimetypes
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from logic_detector.logic import inspect  # noqa: E402

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo"                      # logic_detector/demo (11-06 폴더 정리 때 맨 위 demo/ 에서 옮김)
SAMPLES = DEMO / "samples"
ORIGINALS = DEMO / "originals"
ANSWERS = DEMO / "정답.csv"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def _clean(o):
    """JSON으로 못 보내는 값(날짜·튜플 등)을 문자열·목록으로."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/samples":
            files = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in IMAGE_EXT) if SAMPLES.exists() else []
            return self._send(200, files)
        if path == "/api/answers":
            if not ANSWERS.exists():
                return self._send(200, {})
            rows = list(csv.DictReader(open(ANSWERS, encoding="utf-8-sig")))
            return self._send(200, {r["파일"]: r for r in rows})
        for prefix, folder in (("/samples/", SAMPLES), ("/originals/", ORIGINALS)):
            if path.startswith(prefix):
                p = (folder / path[len(prefix):]).resolve()
                if p.parent != folder.resolve() or not p.exists():
                    return self._send(404, {"error": "없는 파일"})
                return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self._send(404, {"error": "없는 주소"})

    def do_POST(self):
        if self.path != "/api/inspect":
            return self._send(404, {"error": "없는 주소"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            img = ImageOps.exif_transpose(Image.open(BytesIO(self.rfile.read(n)))).convert("RGB")
            r = inspect(img)
            r["size"] = img.size
            return self._send(200, _clean(r))
        except Exception as e:   # 화면에 원인을 보여 준다
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    url = f"http://localhost:{a.port}"
    srv = HTTPServer(("127.0.0.1", a.port), Handler)   # OCR(Vision)이 스레드에 안전한지 몰라 한 번에 한 장씩
    print(f"로직 탐지기 실험 화면: {url}  (끝내려면 Ctrl+C)", flush=True)
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
