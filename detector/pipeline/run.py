"""2단계 합치기 실행기 (병렬, 11-06 팀 결정): 평가셋 하나를 세 엔진으로 보고 최종 판정 하나를 낸다.

읽기는 AI(OpenRouter Qwen) 판독값 하나로 통일한다. 평가셋에 저장된 판독값을 다시 쓰므로 키 · 비용이 들지 않는다.
  동현  ai_judge        판독값 → 동현 검사기            평가셋에 저장된 같은 실행의 동현 답안 (판정 · 위치 · 근거)
  혜인  logic_detector  판독값 → 혜인 규칙              combine/hyein_adapter.py (혜인 님 코드는 그대로)
  소영  detector        사진을 직접 → 금액 · 날짜 칸 흔적  combine/image_engine.py (로컬 OCR, 사진은 밖으로 안 나감)
병렬 결합
  규칙 엔진(동현 · 혜인) 중 하나라도 '의심'          → 의심      (서식상 반드시 맞아야 하는 계산 · 날짜가 깨진 확실한 근거)
  아니면 어느 엔진이든 '확인필요' 또는 이미지 '의심'  → 확인필요  (예외가 있을 수 있거나 흔적만 있는 약한 근거)
  셋 다 정상                                        → 정상

    .venv/bin/python combine/run.py testsets/평가셋_v3
    .venv/bin/python combine/run.py testsets/평가셋_v3 --이미지없이          소영 엔진 빼고 (빠름)
    .venv/bin/python combine/run.py testsets/평가셋_v3 --이미지OCR rapid    칸 찾기를 RapidOCR 로
결과
  평가셋/답안/답안_합침_<엔진>_v1.csv  혜인규칙 · 이미지_<OCR> · 규칙둘(동현+혜인) · 병렬_<OCR> → eval 채점 (점수판에 한 줄씩)
  outputs/combine/<평가셋>_<시각>/     엔진별.csv (서류마다 세 엔진 판정을 나란히) · 요약.md
"""
import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "ai_judge"), str(REPO)]
import hyein_adapter as ha  # noqa: E402
import image_engine as ie  # noqa: E402
from readings_check import load  # noqa: E402

READINGS = "판독값/판독값_동현_AI판단_v1.csv"
DONGHYUN = "답안/답안_동현_AI판단_v1.csv"
COLS = ["문제_id", "판정", "점수", "위치", "근거_종류", "이유"]


def rows(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write(path, res):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(res)


def boxes_str(bs):
    return ";".join(",".join(str(int(v)) for v in b) for b in bs)


def combine(engines, names):
    """엔진 판정들 → (최종 판정, 근거가 된 엔진 이름들). 규칙 엔진의 '의심'만 의심, 나머지 신호는 확인필요."""
    hit = [n for n in names if n in ("동현", "혜인") and engines[n]["판정"] == "의심"]
    if hit:
        return "의심", hit
    weak = [n for n in names if engines[n]["판정"] == "확인필요" or (n == "소영" and engines[n]["판정"] == "의심")]
    return ("확인필요", weak) if weak else ("정상", [])


def answer(qid, engines, names):
    verdict, by = combine(engines, names)
    pos = [engines[n]["위치"] for n in by if engines[n]["위치"]]
    return {"문제_id": qid, "판정": verdict, "위치": ";".join(pos),
            "근거_종류": "규칙위반" if verdict == "의심" else ("흔적" if by == ["소영"] else ""),
            "이유": " / ".join(f"[{n}] {engines[n]['이유']}" for n in by)[:600]}


def main():
    ap = argparse.ArgumentParser(description="세 엔진 병렬 합치기")
    ap.add_argument("평가셋")
    ap.add_argument("--판독값", default=READINGS, help="평가셋 폴더 기준")
    ap.add_argument("--동현답안", default=DONGHYUN, help="평가셋 폴더 기준 (판독값과 같은 실행의 답안)")
    ap.add_argument("--이미지OCR", choices=["tesseract", "rapid"], default="tesseract")
    ap.add_argument("--이미지없이", action="store_true")
    ap.add_argument("--처음", type=int, default=0)
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--버전", default="v1")
    a = ap.parse_args()
    base = Path(a.평가셋).resolve()
    probs = rows(base / "문제목록.csv")[: a.처음 or None]
    vals = load(base / a.판독값)
    dh = {r["문제_id"]: r for r in rows(base / a.동현답안)}
    print(f"{base.name}: 서류 {len(probs)} · 판독값 {len(vals)} · 동현 답안 {len(dh)} ({a.동현답안})")

    eng = {}
    for p in probs:
        q, doc = p["문제_id"], p["문서_종류"]
        d = dh.get(q) or {}
        e = {"동현": {"판정": d.get("판정") or "확인필요", "위치": d.get("위치", ""), "이유": (d.get("이유") or "동현 답안 없음")[:300]}}
        if q in vals:
            h = ha.judge(vals[q], doc)
            hb = [x["boxes"] for x in h["findings"] if x.get("boxes")]
            e["혜인"] = {"판정": h["판정"], "위치": ";".join(boxes_str(b) for b in hb), "이유": ha.reason(h["findings"]) or "걸린 규칙 없음",
                       "모르는_행": h["모르는_행"]}
        else:
            e["혜인"] = {"판정": "확인필요", "위치": "", "이유": "판독값 없음", "모르는_행": []}
        eng[q] = e
    names = ["동현", "혜인"]
    if not a.이미지없이:
        jobs = [(base.name, p["문제_id"], base / p["파일"], a.이미지OCR) for p in probs]
        print(f"소영 이미지 엔진 ({a.이미지OCR}) {len(jobs)}장 … (저장된 결과는 다시 씀)", flush=True)
        with ProcessPoolExecutor(a.j) as ex:
            for job, s in zip(jobs, ex.map(ie.judge, jobs, chunksize=2)):
                eng[job[1]]["소영"] = {"판정": s["판정"], "위치": boxes_str(s["상자"]), "이유": s["이유"], "후보_칸": s.get("후보_칸", 0)}
        names.append("소영")

    tag = a.이미지OCR
    outs = {f"혜인규칙_{a.버전}": [{"문제_id": q, "판정": e["혜인"]["판정"], "위치": e["혜인"]["위치"],
                                  "근거_종류": "규칙위반" if e["혜인"]["판정"] == "의심" else "", "이유": e["혜인"]["이유"]}
                                 for q, e in eng.items()],
            f"규칙둘_{a.버전}": [answer(q, e, ["동현", "혜인"]) for q, e in eng.items()]}
    if "소영" in names:
        outs[f"이미지_{tag}_{a.버전}"] = [{"문제_id": q, "판정": "정상" if e["소영"]["판정"] == "판정못함" else e["소영"]["판정"],
                                       "위치": e["소영"]["위치"], "근거_종류": "흔적" if e["소영"]["판정"] == "의심" else "",
                                       "이유": e["소영"]["이유"]} for q, e in eng.items()]
        outs[f"병렬_{tag}_{a.버전}"] = [answer(q, e, names) for q, e in eng.items()]
    run = REPO / "outputs" / "combine" / f"{base.name}_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=True)
    dest = run if a.처음 else base / "답안"                    # 몇 장만 시험할 때는 평가셋 폴더 · 점수판을 건드리지 않음
    dest.mkdir(exist_ok=True)
    files = [base / a.동현답안]
    for name, res in outs.items():
        files.append(dest / f"답안_합침_{name}.csv")
        write(files[-1], res)

    # 채점: 팀 채점기 그대로 (점수판에 한 줄씩, 위치 적중) + 정상 중 확인필요 비율은 여기서 셈
    key = {r["문제_id"]: r["정답"] for r in rows(base / "정답_채점용" / "정답.csv")}
    loc = {}
    if not a.처음:
        sc = subprocess.run([sys.executable, str(REPO / "eval" / "score_testset.py"), *map(str, files), "--평가셋", str(base)],
                            capture_output=True, text=True)
        loc = {m.group(1): m.group(2) for m in re.finditer(r"\[(.+?)\.csv\].*?위치 적중 (\d+/\d+ \([\d.]+%\)|-)", sc.stdout)}
    table = ["| 답안 | 변조: 의심 | 변조: 확인필요까지 | 정상: 의심(오탐) | 정상: 확인필요 | 위치 적중 |", "|---|---|---|---|---|---|"]
    for f in files:
        got = {r["문제_id"]: r["판정"] for r in rows(f) if r["문제_id"] in eng}
        fo = [q for q in got if key[q] == "변조"]
        no = [q for q in got if key[q] == "정상"]
        pct = lambda n, d: f"{n}/{d} ({100 * n / d:.1f}%)" if d else "-"  # noqa: E731
        table.append(f"| {f.stem.removeprefix('답안_')} | {pct(sum(got[q] == '의심' for q in fo), len(fo))} | "
                     f"{pct(sum(got[q] in ('의심', '확인필요') for q in fo), len(fo))} | {pct(sum(got[q] == '의심' for q in no), len(no))} | "
                     f"{pct(sum(got[q] == '확인필요' for q in no), len(no))} | {loc.get(f.stem, '-')} |")
    with open(run / "엔진별.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["문제_id", "문서", "정답", *names, "병렬" if "소영" in names else "규칙둘", "근거"])
        final = {r["문제_id"]: r for r in rows(files[-1])}
        doc_of = {p["문제_id"]: p["문서_종류"] for p in probs}
        for q, e in eng.items():
            w.writerow([q, doc_of[q], key[q], *[e[n]["판정"] for n in names], final[q]["판정"], final[q]["이유"][:300]])
    notes = []
    if "소영" in names:
        zero = sum(e["소영"].get("후보_칸", 0) == 0 for e in eng.values())
        fail = sum(e["소영"]["판정"] == "판정못함" for e in eng.values())
        notes.append(f"- 소영 이미지 엔진({tag}): 금액 · 날짜 칸을 하나도 못 찾은 서류 {zero}/{len(eng)} (못 찾으면 '정상'), 판정 못 함 {fail}")
    unk = sorted({u for e in eng.values() for u in e["혜인"]["모르는_행"]})
    notes.append(f"- 혜인 규칙: 행 이름을 못 맞춘 행 {unk or '없음'} · 판독값 없는 서류 {sum(q not in vals for q in eng)}")
    summary = [f"# 세 엔진 합치기 — {base.name} ({datetime.now():%Y-%m-%d %H:%M})", "",
               f"판독값 `{a.판독값}` · 동현 답안 `{a.동현답안}` · 이미지 {'없이' if a.이미지없이 else tag}", "", *table, "", *notes, "",
               "결합 규칙: 규칙 엔진(동현 · 혜인) 중 하나라도 의심 → 의심 / 아니면 확인필요 · 이미지 의심 → 확인필요 / 셋 다 정상 → 정상"]
    (run / "요약.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary[4:]))
    where = f"{run.relative_to(REPO)}/ (시험 모드: 채점 안 함)" if a.처음 else f"{base.name}/답안/ (채점 결과는 {base.name}/채점결과/)"
    print(f"\n→ {run.relative_to(REPO)}/ (요약.md · 엔진별.csv), 답안 {len(outs)}개는 {where}")


if __name__ == "__main__":
    main()
