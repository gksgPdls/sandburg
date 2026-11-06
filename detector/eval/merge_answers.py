"""여러 엔진의 답안을 하나로 합친다 (작은 서비스). 평가셋 폴더에 '합치기.py'로 복사되어 들어간다 (make_testset.py).

    python 합치기.py 답안/답안_동현_딥시크_v1.csv 답안/답안_혜인_규칙_v1.csv 답안/답안_소영_이미지_v1.csv --이름 합친_v1
    python 합치기.py 답안/*_v1.csv --이름 합친_v1_다수결 --방식 다수결
    → 답안/답안_<이름>.csv 를 만든다. 채점은 python 채점.py 답안/답안_<이름>.csv

합치는 방식
- 하나라도 (기본): 한 엔진이라도 '의심'이면 의심. 아무도 의심하지 않았는데 하나라도 '확인필요'면 확인필요. 아니면 정상
- 다수결: 답한 엔진의 과반이 '의심'이면 의심 ('확인필요'는 반 표). 아니면 정상
위치는 의심·확인필요로 답한 엔진들의 상자를 모두 모으고, 근거_종류와 이유는 엔진 이름을 붙여 이어 붙인다.
"""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = ["문제_id", "판정", "점수", "위치", "근거_종류", "이유"]
SUS = {"의심", "변조", "위조", "위변조", "suspect", "suspicious", "1", "true", "yes"}
CHECK = {"확인필요", "확인 필요", "check", "review"}


def read(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def kind(v):
    s = (v or "").strip().lower()
    return "의심" if s in SUS else "확인필요" if s in CHECK else ("정상" if s else "")


def main():
    ap = argparse.ArgumentParser(description="여러 엔진의 답안을 하나로 합친다")
    ap.add_argument("답안", nargs="+", help="합칠 답안 CSV 들")
    ap.add_argument("--이름", required=True, help="결과 이름 → 답안/답안_<이름>.csv")
    ap.add_argument("--방식", choices=["하나라도", "다수결"], default="하나라도")
    ap.add_argument("--평가셋", default=str(HERE))
    a = ap.parse_args()
    base = Path(a.평가셋)
    ids = [r["문제_id"] for r in read(base / "문제목록.csv")]
    srcs = [(Path(p).stem.removeprefix("답안_"), {r["문제_id"]: r for r in read(p)}) for p in a.답안]
    out = []
    for qid in ids:
        got = [(name, d[qid]) for name, d in srcs if qid in d and kind(d[qid].get("판정"))]
        if not got:
            out.append({"문제_id": qid})
            continue
        ks = [kind(r["판정"]) for _, r in got]
        if a.방식 == "하나라도":
            v = "의심" if "의심" in ks else "확인필요" if "확인필요" in ks else "정상"
        else:
            votes = ks.count("의심") + 0.5 * ks.count("확인필요")
            v = "의심" if votes > len(ks) / 2 else "정상"
        flagged = [(name, r) for name, r in got if kind(r["판정"]) in ("의심", "확인필요")]
        scores = [float(r["점수"]) for _, r in got if (r.get("점수") or "").replace(".", "", 1).isdigit()]
        out.append({"문제_id": qid, "판정": v, "점수": f"{max(scores):.2f}" if scores else "",
                    "위치": ";".join(r["위치"] for _, r in flagged if (r.get("위치") or "").strip()),
                    "근거_종류": "+".join(dict.fromkeys(r["근거_종류"] for _, r in flagged if r.get("근거_종류"))),
                    "이유": " / ".join(f"[{name}] {r.get('이유', '').strip()}" for name, r in flagged)[:500]})
    dst = base / "답안" / f"답안_{a.이름}.csv"
    dst.parent.mkdir(exist_ok=True)
    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    n = sum(1 for r in out if r.get("판정"))
    print(f"{len(srcs)}개 답안을 '{a.방식}' 방식으로 합침 · 답한 문제 {n}/{len(ids)} → {dst}")
    print(f"채점: python 채점.py {dst.relative_to(base)}")


if __name__ == "__main__":
    main()
