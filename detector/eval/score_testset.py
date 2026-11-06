"""팀 공용 평가셋 채점기. 평가셋 폴더에 '채점.py'로 복사되어 들어간다 (make_testset.py).

    python 채점.py 답안/답안_동현_딥시크_v1.csv           한 개
    python 채점.py 답안/*.csv                           여러 개를 한꺼번에 (점수판이 한 번에 갱신됨)
    python 채점.py 답안.csv --확인필요 의심              '확인필요'를 의심으로 셈 (기본은 정상으로 셈)
    python 채점.py 답안.csv --최소겹침 0.7               위치: 정답 영역의 70% 이상을 상자가 덮어야 적중 (기본: 1픽셀이라도 겹치면)

답안 열: 문제_id, 판정(의심 / 정상 / 확인필요), 점수(선택), 위치(선택, x,y,w,h 여러 개는 ; 로), 근거_종류(선택), 이유(선택)
결과 (채점결과/)
- <답안이름>.md          요약 · 방식별 · 내용 모순별 · 문서별 · 놓친 변조 · 헛짚은 정상
- <답안이름>_문제별.csv   문제마다 정답 · 답 · 결과(잡음/놓침/헛짚음/맞음/답없음) · 위치 적중 → 엔진끼리 어떤 문제를 놓쳤는지 비교
- 점수판.csv · 점수판.md  답안마다 한 줄 (같은 이름을 다시 채점하면 그 줄을 새로 씀) → 셋의 점수를 한 표로
위치 채점은 numpy·Pillow 가 있을 때만 한다 (없으면 판정만 채점).
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUS = {"의심", "변조", "위조", "위변조", "위변조의심", "suspect", "suspicious", "forged", "fake", "1", "true", "yes", "y"}
OK = {"정상", "normal", "genuine", "real", "0", "false", "no", "n"}
CHECK = {"확인필요", "check", "review", "unsure"}
BOARD = ["답안", "채점_시각", "푼_문제", "잡은_비율", "오탐률", "정밀도", "정확도", "위치_적중", "확인필요_처리"]


def read(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def verdict(v):
    s = (v or "").strip().lower().replace(" ", "")
    return "의심" if s in SUS else "정상" if s in OK else "확인필요" if s in CHECK else None


def parse_boxes(s):
    out = []
    for part in (s or "").replace("|", ";").split(";"):
        nums = [x for x in part.replace(" ", "").split(",") if x]
        if len(nums) == 4:
            try:
                out.append(tuple(float(x) for x in nums))
            except ValueError:
                pass
    return out


def location(mask_path, boxes, W, H, min_cover, max_area):
    """(적중 여부, 정답 영역 중 상자가 덮은 비율). 서류 넓이의 max_area 를 넘는 상자는 세지 않는다."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    m = np.asarray(Image.open(mask_path).convert("L")) > 0
    cov = np.zeros_like(m)
    for x, y, w, h in boxes:
        if w * h > max_area * W * H:
            continue
        x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
        if x1 > x0 and y1 > y0:
            cov[y0:y1, x0:x1] = True
    total, inter = int(m.sum()), int((cov & m).sum())
    cover = inter / total if total else 0.0
    return inter > 0 and cover >= min_cover, cover


def pct(a, b):
    return f"{a}/{b} ({a / b:.1%})" if b else "-"


def score(answer_path, base, a, key, size):
    ans, bad = {}, []
    for r in read(answer_path):
        qid = (r.get("문제_id") or r.get("id") or "").strip()
        if not qid:
            continue
        if qid not in key:
            bad.append(f"{qid}: 없는 문제 번호")
            continue
        if not (r.get("판정") or "").strip():
            continue                                  # 빈 칸 = 안 푼 문제
        v = verdict(r.get("판정"))
        if v is None:
            bad.append(f"{qid}: 판정 '{r.get('판정')}' 을 알 수 없음 (의심 / 정상 / 확인필요)")
            continue
        ans[qid] = {**r, "v": v}
    tp, fn, fp, tn, unanswered, per_q = [], [], [], [], [], []
    loc_hit, loc_try, loc_none, n_check = 0, 0, 0, 0
    for qid, k in sorted(key.items()):
        r = ans.get(qid)
        row = {"문제_id": qid, "정답": k["정답"], "답": r["v"] if r else "", "결과": "", "위치_적중": "", "덮은_비율": "",
               "방식": k["방식"], "내용_모순": k["내용_모순"], "문서_종류": k["문서_종류"]}
        per_q.append(row)
        if r is None:
            unanswered.append(qid)
            row["결과"] = "답없음"
            continue
        n_check += r["v"] == "확인필요"
        sus = r["v"] == "의심" or (r["v"] == "확인필요" and a.확인필요 == "의심")
        if k["정답"] == "변조":
            (tp if sus else fn).append(qid)
            row["결과"] = "잡음" if sus else "놓침"
            if sus:
                bx = parse_boxes(r.get("위치"))
                if not bx:
                    loc_none += 1
                    continue
                W, H = size[qid]
                res = location(base / "정답_채점용" / k["마스크"], bx, W, H, a.최소겹침, a.최대상자)
                if res is not None:
                    loc_try += 1
                    loc_hit += res[0]
                    row["위치_적중"], row["덮은_비율"] = ("예" if res[0] else "아니요"), f"{res[1]:.2f}"
        else:
            (fp if sus else tn).append(qid)
            row["결과"] = "헛짚음" if sus else "맞음"
    n_f, n_n = len(tp) + len(fn), len(fp) + len(tn)

    def recall_by(field):
        g = defaultdict(lambda: [0, 0])
        for qid in tp + fn:
            g[key[qid][field] or "-"][1] += 1
            g[key[qid][field] or "-"][0] += qid in tp
        return g

    lines = [f"# 채점 결과 — {Path(answer_path).name}", "",
             f"*평가셋 {base.name} · 문제 {len(key)} · '확인필요'는 {a.확인필요}으로 셈 ({n_check}개) · "
             f"위치: {'1픽셀 이상 겹치면' if a.최소겹침 <= 0 else f'정답 영역의 {a.최소겹침:.0%} 이상 덮으면'} 적중, "
             f"서류 넓이 {a.최대상자:.0%} 넘는 상자는 제외*", "",
             "## 요약", "", "| 지표 | 값 | 뜻 |", "|---|---|---|",
             f"| **잡은 비율 (재현율)** | {pct(len(tp), n_f)} | 변조 중 '의심'으로 본 비율 |",
             f"| **오탐률** | {pct(len(fp), n_n)} | 정상 중 '의심'으로 본 비율 (낮을수록 좋음) |",
             f"| 정밀도 | {pct(len(tp), len(tp) + len(fp))} | '의심'이라고 한 것 중 진짜 변조 |",
             f"| 정확도 | {pct(len(tp) + len(tn), n_f + n_n)} | 전체 중 맞힌 비율 |",
             f"| 위치 적중 | {pct(loc_hit, loc_try)} | 잡은 변조 중 위치까지 맞힌 비율 (위치를 적은 것만, 안 적은 것 {loc_none}개) |",
             f"| 답 없음 | {len(unanswered)} | 채점에서 뺌. 다른 답안과 비교할 때는 푼 문제가 같은지 확인 |", "",
             "> 전체 점수보다 **방식별 · 내용 모순별** 점수를 보세요. 변조 61장 중 51장이 AI 방식이고, 같은 변조가 글꼴만 바꿔 세 벌씩 있습니다.", ""]
    for title, field, note in (("변조 방식별 잡은 비율", "방식", {}),
                               ("내용 모순별 잡은 비율", "내용_모순",
                                {"예": "서류만 봐도 말이 안 되는 곳이 남음 → 내용을 보는 엔진(AI 판단·규칙)이 잡아야 함",
                                 "약함": "있을 수는 있지만 어색함", "아니요": "서류 안 모순 없음 → 이미지 흔적으로만 잡을 수 있음"})):
        lines += [f"## {title}", "", f"| {field.replace('_', ' ')} | 잡은 비율 | 참고 |", "|---|---|---|"]
        for g, (h, n) in sorted(recall_by(field).items()):
            lines.append(f"| {g} | {pct(h, n)} | {note.get(g, '')} |")
        lines.append("")
    lines += ["## 문서 종류별", "", "| 문서 | 잡은 비율 | 오탐률 |", "|---|---|---|"]
    for d in sorted({k["문서_종류"] for k in key.values()}):
        f_ = [q for q in tp + fn if key[q]["문서_종류"] == d]
        n_ = [q for q in fp + tn if key[q]["문서_종류"] == d]
        lines.append(f"| {d} | {pct(sum(q in tp for q in f_), len(f_))} | {pct(sum(q in fp for q in n_), len(n_))} |")
    lines += ["", f"## 놓친 변조 ({len(fn)})", "", "| 문제 | 방식 | 바뀐 내용 | 내용 모순 |", "|---|---|---|---|"]
    lines += [f"| {q} | {key[q]['방식']} | {key[q]['바뀐_내용']} | {key[q]['내용_모순']} |" for q in fn]
    lines += ["", f"## 헛짚은 정상 ({len(fp)})", "", "| 문제 | 문서 종류 | 답안의 이유 |", "|---|---|---|"]
    lines += [f"| {q} | {key[q]['문서_종류']} | {(ans[q].get('이유') or '').strip()[:80]} |" for q in fp]
    if bad:
        lines += ["", f"## 답안 문제 ({len(bad)})", ""] + [f"- {b}" for b in bad]
    out = base / "채점결과"
    out.mkdir(exist_ok=True)
    stem = Path(answer_path).stem
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / f"{stem}_문제별.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q[0]))
        w.writeheader()
        w.writerows(per_q)
    print(f"[{Path(answer_path).name}] 잡은 비율 {pct(len(tp), n_f)} · 오탐률 {pct(len(fp), n_n)} · "
          f"정밀도 {pct(len(tp), len(tp) + len(fp))} · 위치 적중 {pct(loc_hit, loc_try)} · 답 없음 {len(unanswered)}"
          + (f" · 답안 문제 {len(bad)}" if bad else ""))
    return {"답안": stem, "채점_시각": datetime.now().strftime("%Y-%m-%d %H:%M"), "푼_문제": f"{n_f + n_n}/{len(key)}",
            "잡은_비율": pct(len(tp), n_f), "오탐률": pct(len(fp), n_n), "정밀도": pct(len(tp), len(tp) + len(fp)),
            "정확도": pct(len(tp) + len(tn), n_f + n_n), "위치_적중": pct(loc_hit, loc_try), "확인필요_처리": a.확인필요}


def update_board(out, rows):
    """점수판: 답안마다 한 줄. 같은 답안 이름을 다시 채점하면 그 줄을 새로 쓴다."""
    p = out / "점수판.csv"
    board = {r["답안"]: r for r in read(p)} if p.exists() else {}
    for r in rows:
        board[r["답안"]] = r
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOARD, extrasaction="ignore")
        w.writeheader()
        w.writerows(board.values())
    md = ["# 점수판", "", "| " + " | ".join(c.replace("_", " ") for c in BOARD) + " |", "|" + "---|" * len(BOARD)]
    md += ["| " + " | ".join(r.get(c, "") for c in BOARD) + " |" for r in board.values()]
    (out / "점수판.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="팀 공용 평가셋 채점기")
    ap.add_argument("답안", nargs="+", help="채운 답안 CSV (답안_양식.csv 형식). 여러 개 가능")
    ap.add_argument("--평가셋", default=str(HERE), help="평가셋 폴더 (기본: 이 파일이 있는 폴더)")
    ap.add_argument("--확인필요", choices=["정상", "의심"], default="정상", help="'확인필요' 답을 어느 쪽으로 셀지")
    ap.add_argument("--최소겹침", type=float, default=0.0, help="위치 적중 기준: 정답 영역을 상자가 덮은 비율 (0 = 1픽셀이라도)")
    ap.add_argument("--최대상자", type=float, default=0.10, help="서류 넓이의 이 비율을 넘는 상자는 위치로 인정하지 않음")
    a = ap.parse_args()
    base = Path(a.평가셋)
    key_path = base / "정답_채점용" / "정답.csv"
    if not key_path.exists():
        sys.exit(f"정답 파일이 없습니다: {key_path}  (--평가셋 로 평가셋 폴더를 알려 주세요)")
    key = {r["문제_id"]: r for r in read(key_path)}
    size = {r["문제_id"]: (int(r["가로"]), int(r["세로"])) for r in read(base / "문제목록.csv")}
    rows = [score(p, base, a, key, size) for p in a.답안]
    update_board(base / "채점결과", rows)
    print(f"→ {base / '채점결과'} (답안별 결과, 문제별 표, 점수판)")


if __name__ == "__main__":
    main()
