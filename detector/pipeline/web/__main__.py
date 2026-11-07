"""합친 탐지기 실험 화면 (detector/logic/web 과 같은 화면, 이 맥에서만 도는 작은 서버).

    .venv/bin/python -m detector.pipeline.web --demo <샘플 폴더>              # http://localhost:8771, 저장된 결과만 보여 줌 (외부 호출 없음)
    .venv/bin/python -m detector.pipeline.web --demo <샘플 폴더> --실시간      # 저장된 결과가 없는 사진은 VLM 으로 읽어 검사 (합성 서류만)

샘플 폴더 (git 에 넣지 않음)
  samples/     검사할 사진          originals/  변조 전 원본 (있으면 나란히)
  정답.csv     파일 · 원래 번호 · 서류 · 조작 방식 · 바뀐 내용 · 바뀐 곳(x,y,w,h) · 원본 파일 · 케이스
  info.json   화면 제목 · 목록 이름
  결과.json    {사진 sha1: 화면 결과} — 미리 계산한 결과 (detector.pipeline.web.view 로 만든 것). 같은 사진이면 다시 보내지 않음

  GET  /  /api/samples  /api/answers  /api/info  /samples/<파일>  /originals/<파일>
  POST /api/inspect     본문 = 이미지 바이트 → 화면 결과 (verdict · engines · findings · reading · notes)
"""
import argparse
import csv
import hashlib
import json
import mimetypes
import sys
import tempfile
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from detector.pipeline.web.view import view  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
DEMO = None
LIVE = False


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def saved():
    p = DEMO / "결과.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/samples":
            d = DEMO / "samples"
            return self._send(200, sorted(p.name for p in d.glob("*") if p.suffix.lower() in IMAGE_EXT) if d.exists() else [])
        if path == "/api/info":
            p = DEMO / "info.json"
            return self._send(200, json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
        if path == "/api/answers":
            p = DEMO / "정답.csv"
            return self._send(200, {r["파일"]: r for r in csv.DictReader(open(p, encoding="utf-8-sig"))} if p.exists() else {})
        for prefix, sub in (("/samples/", "samples"), ("/originals/", "originals")):
            if path.startswith(prefix):
                folder = (DEMO / sub).resolve()
                p = (folder / path[len(prefix):]).resolve()
                if p.parent != folder or not p.exists():
                    return self._send(404, {"error": "없는 파일"})
                return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self._send(404, {"error": "없는 주소"})

    def do_POST(self):
        if self.path != "/api/inspect":
            return self._send(404, {"error": "없는 주소"})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        key = hashlib.sha1(body).hexdigest()
        got = saved().get(key)
        if got:
            return self._send(200, {**got, "saved": True})
        if not LIVE:
            return self._send(400, {"error": "저장된 결과가 없는 사진입니다. 새 사진을 검사하려면 서버를 --실시간 으로 켜세요 "
                                             "(VLM 으로 외부 전송 — 합성 서류만)."})
        try:
            from PIL import Image
            from detector.pipeline import pipeline
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / "upload.png"
                f.write_bytes(body)
                Image.open(f).size                                    # 이미지인지 확인
                r = pipeline.inspect(f, "자동")
            return self._send(200, _clean(view(r["logic"], r["pixel"], r["판정"], r["판독"].get("판독") or {}, r["문서"],
                                               r["logic"].get("note", ""))))
        except SystemExit as e:                                       # 실서류 막기
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")


def main():
    global DEMO, LIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="샘플 폴더 (samples/ 정답.csv info.json 결과.json)")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--실시간", action="store_true", help="저장된 결과가 없는 사진은 VLM 으로 읽어 검사 (외부 전송, 합성 서류만)")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    DEMO, LIVE = Path(a.demo).expanduser().resolve(), a.실시간
    url = f"http://localhost:{a.port}"
    srv = HTTPServer(("127.0.0.1", a.port), Handler)
    print(f"합친 탐지기 실험 화면: {url}  (샘플 {DEMO}, {'실시간 검사 켬' if LIVE else '저장된 결과만'})  끝내려면 Ctrl+C", flush=True)
    if not a.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
