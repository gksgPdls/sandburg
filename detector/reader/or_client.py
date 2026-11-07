"""OpenRouter 호출 공통 도구 (VLM 판독기가 쓴다).

- 키는 환경변수 OPENROUTER_API_KEY 또는 ~/.openrouter_key 파일에서만 읽는다. 화면·로그·결과 파일에 절대 남기지 않는다.
  (키 저장: save_key.py — 키를 화면에 안 보이게 입력받아 ~/.openrouter_key 에 저장)
- **합성 서류만 보낸다.** 실서류·1차 과제 데이터는 보내지 않는다. 사진을 보내는 모든 길이 refuse_real() 을 거친다.
- 운영 조건과 같게: 온도 0, 시드 고정, 생각(reasoning) 끔이 기본. 데이터를 학습에 쓰는 제공 업체는 피한다(data_collection: deny).
- 응답마다 실제로 처리한 모델·제공 업체·시간·토큰·비용을 함께 돌려준다 (재현성 기록용).
"""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_FILE = Path.home() / ".openrouter_key"
Image.MAX_IMAGE_PIXELS = None
# 모델별 제공 업체 고정. 2026-11-04 딥시크 요청이 OpenInference·Relace 로 가면 사진을 못 받고 빈 판독값만 돌려주는 일이 많음
# (평가셋 v2 실행에서 OpenInference 108건 중 89건, 다시 시험한 Relace 영수증 5건 중 4건). 사진을 잘 읽은 업체만 쓴다.
# 같은 모델이라도 업체에 따라 결과가 달라지므로 업체는 결과마다 기록한다.
PROVIDER_ONLY = {"deepseek/deepseek-v4.1-flash-1017": ["Cohere", "ModelRun", "Wafer"]}
# 실서류(와 실서류로 만든 1차 과제 데이터)를 알아보는 표시. 폴더 위치가 사람마다 달라도 막히게 이름으로 본다
REAL_PARTS = {"real_docs", "dataset", "demo_real"}
REAL_PATHS = ("detector/logic/demo/",)          # 혜인 님 실험 화면 샘플: 1차 과제(실서류로 만든) 변조가 섞여 있음
REAL_NAME = re.compile(r"(\d{6,})_\d{3}_(?:med|surg|dig)_\d{8}_DTB\d{3}_F\d+", re.I)   # 사고번호_순번_med_날짜_DTB코드_F번호


def refuse_real(path):
    """실서류로 보이면 멈춘다: 경로에 real_docs · dataset 폴더 · 'real_docs' 가 있거나, 파일 이름이 실서류 형식이면.
    가짜 사고번호 999… 로 시작하는 이름은 합성 서류 (합성 서류 패키징 스크립트)."""
    p = Path(path).resolve()
    parts = [s.lower() for s in p.parts]
    m = REAL_NAME.search(p.name)
    if (REAL_PARTS & set(parts) or any("real_docs" in s for s in parts) or any(x in p.as_posix() + "/" for x in REAL_PATHS)
            or (m and not m.group(1).startswith("999"))):
        raise SystemExit(f"실서류로 보여 멈춥니다: {p.name} — 외부 AI 에는 합성 서류만 보냅니다 "
                         "(합성 서류인데 막혔다면 dataset · real_docs 밖 폴더로 옮기고 이름을 바꾸세요)")


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding="utf-8").strip()
    return k


def image_data_url(path, long_side=1500, quality=90):
    """사진을 긴 변 long_side 로 줄여 JPEG data URL 로. (원래 크기, 보낸 크기, 바이트 수)도 돌려준다.
    상자 좌표를 원래 크기로 되돌릴 때 배율 = 원래 가로 / 보낸 가로."""
    refuse_real(path)
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = min(1.0, long_side / max(W, H)) if long_side else 1.0
    if s < 1:
        im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    data = buf.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}", (W, H), im.size, len(data)


def parse_json(text):
    """모델 답에서 JSON 객체 하나를 꺼낸다 (```json 울타리, 앞뒤 설명 허용). 실패하면 None."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            v = json.loads(cand)
            return v if isinstance(v, dict) else None
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _post(body, timeout):
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"), method="POST", headers={
        "Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "X-Title": "doc-forgery-detector"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(model, messages, *, temperature=0, max_tokens=1200, seed=7, json_mode=True, reasoning="off", timeout=240, retries=2):
    """한 번 묻고 결과를 dict 로 돌려준다.
    {ok, text, json, model, provider, seconds, prompt_tokens, completion_tokens, cost, error, notes}
    - 400 오류가 생각 설정·JSON 형식 옵션 때문이면 그 옵션을 빼고 1회 다시 묻고 notes 에 적는다.
    - 429(요청 많음)·5xx 는 잠깐 쉬었다가 retries 번까지 다시 묻는다."""
    if not api_key():
        return {"ok": False, "error": "키 없음: ~/.openrouter_key 또는 OPENROUTER_API_KEY 를 설정하세요"}
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            "usage": {"include": True}, "provider": {"data_collection": "deny"}}
    only = os.environ.get("OR_PROVIDER_ONLY")                  # 실행마다 업체를 고정 (예: 정밀도 fp4 업체만) — e3_pipeline --업체
    if only:
        body["provider"]["only"] = [s.strip() for s in only.split(",") if s.strip()]
    elif PROVIDER_ONLY.get(model):
        body["provider"]["only"] = PROVIDER_ONLY[model]
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning in ("off", "on"):
        body["reasoning"] = {"enabled": reasoning == "on"}
    notes, t0, attempt = [], time.time(), 0
    while True:
        attempt += 1
        try:
            resp = _post(body, timeout)
            break
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode("utf-8", "ignore")
            if e.code == 400 and "reasoning" in body and "reason" in msg.lower():
                body.pop("reasoning")
                notes.append("생각 설정을 지원하지 않아 빼고 다시 물음")
                continue
            if e.code == 400 and "response_format" in body and ("response_format" in msg or "json" in msg.lower()):
                body.pop("response_format")
                notes.append("JSON 형식 옵션을 지원하지 않아 빼고 다시 물음")
                continue
            if e.code in (429, 500, 502, 503, 504) and attempt <= retries:
                time.sleep(5 * attempt)
                continue
            hint = {401: "키가 틀렸거나 만료됨", 402: "크레딧 부족", 403: "권한 없음(데이터 정책으로 맞는 제공 업체가 없을 수 있음)",
                    404: "모델 이름 확인", 429: "요청이 너무 많음"}.get(e.code, "")
            return {"ok": False, "error": f"HTTP {e.code} {hint}: {msg[:200]}", "seconds": round(time.time() - t0, 1), "notes": notes}
        except Exception as e:                              # 시간 초과·연결 끊김
            if attempt <= retries:
                time.sleep(5 * attempt)
                continue
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1), "notes": notes}
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    text = msg.get("content") or ""
    u = resp.get("usage") or {}
    return {"ok": True, "text": text, "json": parse_json(text), "model": resp.get("model", model),
            "provider": resp.get("provider", ""), "seconds": round(time.time() - t0, 1),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cost": u.get("cost"), "error": None, "notes": notes}
