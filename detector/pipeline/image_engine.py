"""소영 님 이미지 탐지기(detector v1) 연결: 사진에서 금액 · 날짜 칸을 스스로 찾아(로컬 OCR) 칸마다 흔적 검사.

detector.v1.locate() 그대로 씀: 가장 이상한 칸의 희귀도가 기준(config 의 doc_max_rarity)보다 낮으면 '의심'.
칸 찾기 OCR 은 소영 님 기본값 tesseract (rapid 는 칸을 훨씬 많이 찾지만, 11-06 6장 시험에서 정상 4장을 모두 의심 — 칸이 많으면
그중 하나는 우연히 기준을 넘음). 사진 한 장에 3~7초라 결과를 outputs/combine/이미지_저장/<OCR>/<평가셋>/<문제_id>.json 에 저장해 다시 쓴다.
소영 님 코드는 고치지 않는다. 사진은 밖으로 나가지 않는다.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SAVE = REPO / "outputs" / "combine" / "이미지_저장"
TOP = 3                                                       # 근거 상자로 쓰는 칸 수 (상자가 많으면 위치 적중이 부풀어서)


def judge(job):
    """(평가셋 이름, 문제_id, 사진 경로, OCR) → {판정(의심/정상/판정못함), 상자, 이유, 후보_칸}. 저장해 둔 결과가 있으면 그것을 씀."""
    name, qid, img, ocr = job
    path = SAVE / ocr / name / f"{qid}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from detector.v1.locate import locate
    try:
        r = locate(str(img), ocr=ocr)
        thr = r["doc_max_rarity"]
        odd = [c for c in r["candidates"] if c["rarity"] is not None and c["rarity"] < thr][:TOP]
        out = {"판정": "의심" if r["suspicious"] else "정상", "상자": [list(c["bbox"]) for c in odd],
               "이유": (f"흔적 의심 칸 {len(odd)}개 (" + ", ".join(f"'{c['text']}' 희귀도 {c['rarity']:.4f}" for c in odd) + ")")
               if odd else r["summary"], "후보_칸": r["n_candidates"]}
    except Exception as e:                                    # noqa: BLE001  (사진 한 장 실패는 '판정 못 함'으로 남김)
        out = {"판정": "판정못함", "상자": [], "이유": f"{type(e).__name__}: {e}"[:200], "후보_칸": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out
