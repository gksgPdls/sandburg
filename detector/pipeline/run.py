"""합친 탐지기 실행기: 사진 폴더 · 평가셋 → 서류마다 VLM 읽기 + 로직 · 픽셀 병렬 → 판정 · 답안.

    .venv/bin/python combine/run.py testsets/평가셋_v3                 # 평가셋: 답안 4개(등급형OR·순수OR·로직만·픽셀만) + 채점
    .venv/bin/python combine/run.py <사진 폴더> --처음 5
    .venv/bin/python combine/run.py --다시판정 outputs/combine/<결과 폴더>   # 호출 없이 문서별.jsonl 로 판정만 다시 (기준값 바꾼 뒤 등)
옵션: --모델 qwen27b · --크기 2500 · -j 6 (VLM 동시 요청 수) · --위치없이 · --캐시없이 · --픽셀없이

결과: outputs/combine/<폴더>_<시각>/
  문서별.jsonl   서류마다 두 엔진의 원래 결과 · 판독값 · 비용 (이것만 있으면 결합 방식을 바꿔 다시 판정 가능)
  판정.csv       파일 · 문서 · 판정(등급형 OR) · 로직 · 픽셀 · 픽셀 점수 · 이유 · 위치 · 비용
  답안_<방식>.csv 팀 답안 형식 (평가셋이면 <평가셋>/답안/ 에도 답안_합친_<방식>_<버전>.csv 로 복사)
  요약.md
VLM 원본 응답은 outputs/combine/vlm_cache/ 에 남아 같은 사진을 다시 돌려도 비용이 들지 않는다.
**합성 서류만.** 실서류(dataset · real_docs · demo_real · 실서류 이름 형식)는 아무것도 보내기 전에 멈춘다.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from combine import decide as D  # noqa: E402
from combine import engines as E  # noqa: E402
from combine import pipeline as P  # noqa: E402

OUT = REPO / "outputs" / "combine"
DTB = {"DOC210": "영수증", "DOC205": "영수증", "DOC203": "진단서"}
IMG_EXT = {".png", ".jpg", ".jpeg"}
ANS_COLS = ["문제_id", "판정", "점수", "위치", "근거_종류", "이유"]
SCORE = {"의심": "1.00", "확인필요": "0.50", "정상": "0.00"}
MODE_NAME = {"graded": "등급형OR", "or": "순수OR", "logic": "로직만", "pixel": "픽셀만"}


def rows(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def items_of(base):
    """평가셋이면 문제목록.csv, 아니면 폴더의 사진들 (문서 종류: 파일 이름 DTB 코드 → 없으면 '자동')."""
    if (base / "문제목록.csv").exists():
        return [{"문제_id": q["문제_id"], "파일": q["파일"], "문서_종류": q["문서_종류"]} for q in rows(base / "문제목록.csv")]
    out = []
    for f in sorted(p for p in base.rglob("*") if p.suffix.lower() in IMG_EXT and "_mask" not in p.stem and "마스크" not in p.stem):
        m = re.search(r"DTB\d{3}", f.name)
        out.append({"문제_id": f.stem, "파일": str(f.relative_to(base)), "문서_종류": DTB.get(m.group(), "자동") if m else "자동"})
    return out


def rejudge(rec, thr):
    """저장된 기록 + (새) 픽셀 기준값 → 방식별 판정. 호출 없음."""
    px = dict(rec["pixel"])
    if thr and px.get("score") is not None:
        s = px["score"]
        px["level"] = "의심" if s > thr["hi"] else "확인필요" if s > thr["lo"] else "정상"
        px["strong"] = s > thr["hi"]
        px["boxes"] = [",".join(str(int(v)) for v in c["bbox"]) for c in px.get("findings", []) if c["score"] > thr["lo"]][:3]
    out = {}
    for m in D.MODES:
        v, why = D.decide(rec["logic"], px, m)
        lg = rec["logic"]["boxes"] if rec["logic"]["level"] in D.FLAG and m != "pixel" else []
        pb = px["boxes"] if px["level"] in D.FLAG and m != "logic" else []
        out[m] = {"판정": v, "이유": why, "위치": list(dict.fromkeys(lg + pb)), "pixel": px}
    return out


def reason_text(rec, px):
    parts = []
    for f in rec["logic"].get("findings", [])[:3]:
        parts.append(f"[로직 {f.get('rule')} {f.get('grade')}] {f.get('title')}")
    if px["level"] in D.FLAG:
        top = (px.get("findings") or [{}])[0]
        parts.append(f"[픽셀 {px['level']}] 다시 그린 칸 점수 {px.get('score')} ({top.get('why', '')})")
    if rec["logic"].get("note"):
        parts.append(f"(메모: {rec['logic']['note']})")
    return " / ".join(parts)


def kind_of(rec, px, m):
    lf, pf = rec["logic"]["level"] in D.FLAG and m != "pixel", px["level"] in D.FLAG and m != "logic"
    return "규칙위반+글자모양" if lf and pf else "규칙위반" if lf else "글자모양" if pf else ""


def write_outputs(out, recs, base, version):
    thr = E.load_thresholds()
    judged = {r["문제_id"]: rejudge(r, thr) for r in recs}
    with open(out / "판정.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["문제_id", "파일", "문서", "판정", "로직", "픽셀", "픽셀_점수", "이유", "위치", "초", "비용"])
        for r in recs:
            j = judged[r["문제_id"]]["graded"]
            w.writerow([r["문제_id"], r["파일"], r["문서"], j["판정"], r["logic"]["level"], j["pixel"]["level"], j["pixel"].get("score"),
                        reason_text(r, j["pixel"]), ";".join(j["위치"]), r["초"], f"{r['비용']:.4f}"])
    is_test = (base / "문제목록.csv").exists()
    answers = {}
    for m in D.MODES:
        p = out / f"답안_{MODE_NAME[m]}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ANS_COLS)
            w.writeheader()
            for r in recs:
                j = judged[r["문제_id"]][m]
                w.writerow({"문제_id": r["문제_id"], "판정": j["판정"], "점수": SCORE.get(j["판정"], ""), "위치": ";".join(j["위치"]),
                            "근거_종류": kind_of(r, j["pixel"], m), "이유": reason_text(r, j["pixel"])})
        answers[m] = p
        if is_test:
            (base / "답안").mkdir(exist_ok=True)
            shutil.copy(p, base / "답안" / f"답안_합친_{MODE_NAME[m]}_{version}.csv")
    lines = [f"# 합친 탐지기 결과 — {base.name}", "", f"- 서류 {len(recs)}장 · VLM 비용 ${sum(r['비용'] for r in recs):.3f} · "
             f"픽셀 기준값 {'있음 ' + json.dumps(thr, ensure_ascii=False) if thr else '없음 (픽셀은 보류)'}", "",
             "| 방식 | 의심 | 확인필요 | 정상 |", "|---|---|---|---|"]
    for m in D.MODES:
        c = Counter(judged[r["문제_id"]][m]["판정"] for r in recs)
        lines.append(f"| {MODE_NAME[m]} | {c['의심']} | {c['확인필요']} | {c['정상']} |")
    if is_test and (REPO / "eval" / "score_testset.py").exists():
        lines += ["", "## 채점 (확인필요 = 정상으로 셈)", ""]
        for m in D.MODES:
            res = subprocess.run([sys.executable, str(REPO / "eval" / "score_testset.py"), str(answers[m]), "--평가셋", str(base)],
                                 capture_output=True, text=True)
            lines += [f"### {MODE_NAME[m]}", "```", (res.stdout or res.stderr).strip()[-3000:], "```"]
    (out / "요약.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12 + len(D.MODES)]))
    print(f"\n결과: {out}")


def main():
    ap = argparse.ArgumentParser(description="합친 탐지기 (VLM 읽기 + 로직·픽셀 병렬)")
    ap.add_argument("폴더", nargs="?")
    ap.add_argument("--모델", default="qwen27b")
    ap.add_argument("--크기", type=int, default=2500)
    ap.add_argument("--처음", type=int, default=0)
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--버전", default="v1")
    ap.add_argument("--위치없이", action="store_true", help="로직이 걸린 칸의 위치를 VLM 에게 다시 묻지 않음 (비용 절약)")
    ap.add_argument("--캐시없이", action="store_true")
    ap.add_argument("--픽셀없이", action="store_true")
    ap.add_argument("--다시판정", help="이전 결과 폴더: 호출 없이 판정 · 답안만 다시 만든다")
    ap.add_argument("--실서류허용", action="store_true", help=f"실서류 막기를 끔 (한 번에 {20}장까지, 허용 범위)")
    a = ap.parse_args()

    if a.다시판정:
        out = Path(a.다시판정)
        cfg = json.loads((out / "설정.json").read_text(encoding="utf-8"))
        recs = [json.loads(l) for l in open(out / "문서별.jsonl", encoding="utf-8")]
        write_outputs(out, recs, Path(cfg["폴더"]), cfg.get("버전", "v1"))
        return
    if not a.폴더:
        ap.error("폴더를 주세요")
    base = (Path(a.폴더) if Path(a.폴더).exists() else REPO / a.폴더).resolve()
    items = items_of(base)[: a.처음 or None]
    if a.실서류허용:
        P.V.allow_real(len(items))
        print(f"주의: 실서류 막기를 끄고 {len(items)}장을 외부 AI(OpenRouter)로 보냅니다 (허용 범위 {P.V.REAL_LIMIT}장 이내)")
    P.V.oc.refuse_real(base)
    for q in items:                                                   # 한 장이라도 실서류로 보이면 아무것도 보내기 전에 멈춤
        P.V.oc.refuse_real(base / q["파일"])
    if not P.V.oc.api_key():
        raise SystemExit("OpenRouter 키가 없습니다: .venv/bin/python ai_judge/save_key.py")
    thr = E.load_thresholds()
    if not thr and not a.픽셀없이:
        print("주의: 픽셀 기준값이 없어 픽셀은 '보류'로 남깁니다 (점수는 저장 → 기준값을 만든 뒤 --다시판정).")
    out = OUT / f"{base.name}_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "설정.json").write_text(json.dumps({"폴더": str(base), "모델": a.모델, "크기": a.크기, "처음": a.처음, "버전": a.버전,
                                             "위치": not a.위치없이, "픽셀": not a.픽셀없이, "픽셀_기준값": thr,
                                             "실서류허용": a.실서류허용},
                                            ensure_ascii=False, indent=1), encoding="utf-8")

    def one(q):
        r = P.inspect(base / q["파일"], q["문서_종류"], a.모델, a.크기, use_cache=not a.캐시없이, locate=not a.위치없이,
                      pixel_thr=thr, use_pixel=not a.픽셀없이)
        return {"문제_id": q["문제_id"], **r}

    recs = []
    with open(out / "문서별.jsonl", "w", encoding="utf-8") as f, ThreadPoolExecutor(a.j) as ex:
        futs = {ex.submit(one, q): q for q in items}
        for i, fu in enumerate(as_completed(futs), 1):
            q = futs[fu]
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(items)}] {q['문제_id']} 오류 {type(e).__name__}: {e}")
                continue
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            f.flush()
            recs.append(r)
            print(f"  [{i}/{len(items)}] {q['문제_id']} {r['문서']} → {r['판정']} (로직 {r['logic']['level']}, "
                  f"픽셀 {r['pixel']['level']} {r['pixel'].get('score')}) {r['초']}s ${r['비용']:.4f}", flush=True)
    order = {q["문제_id"]: i for i, q in enumerate(items)}
    recs.sort(key=lambda r: order[r["문제_id"]])
    write_outputs(out, recs, base, a.버전)


if __name__ == "__main__":
    main()
