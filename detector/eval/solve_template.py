"""풀이 틀: 아래 판단() 함수 하나만 채우면 문제를 차례로 돌려 답안 CSV를 만든다.
평가셋 폴더에 '풀이_틀.py'로 복사되어 들어간다 (make_testset.py). 자기 이름으로 복사해서 고쳐 쓴다 (예: 풀이_동현_딥시크.py).

    python 풀이_동현_딥시크.py --이름 동현_딥시크_v1 --처음 10      먼저 10문제만 시험
    python 풀이_동현_딥시크.py --이름 동현_딥시크_v1               전체 → 답안/답안_동현_딥시크_v1.csv
    python 풀이_동현_딥시크.py --이름 동현_딥시크_v1 --긴변 1500    사진을 긴 변 1500px로 줄여 판단()에 넘김 (AI API 비용·속도)
                                                                 판단()이 돌려준 상자는 자동으로 원래 크기 좌표로 되돌려 적는다
    python 풀이_동현_딥시크.py --이름 동현_딥시크_v1 --이어서       이미 답한 문제는 건너뜀 (중간에 끊겼을 때)

규칙: 한 장씩 따로 판단한다(다른 문제 사진과 비교하지 않는다). 정답_채점용/ 은 쓰지 않는다.
필요: Python 3, Pillow
"""
import argparse
import csv
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
COLS = ["문제_id", "판정", "점수", "위치", "근거_종류", "이유"]
Image.MAX_IMAGE_PIXELS = None


def 판단(image, info):
    """여기만 채우세요.

    image: PIL 이미지 (--긴변 을 주면 줄인 사진)
    info:  {'문제_id': 'T001', '문서_종류': '영수증' 또는 '진단서', '가로': 원래 가로, '세로': 원래 세로, '파일': 사진 경로}
    돌려줄 것 (판정만 필수):
        {'판정': '의심' | '정상' | '확인필요',
         '점수': 0~1,                                   의심스러울수록 크게
         '위치': [(x, y, w, h), ...],                    넘겨받은 image 기준 픽셀. 여러 칸이면 칸마다 상자
         '근거_종류': '규칙위반' | '같은픽셀' | '글자모양' | 'AI판단' | '기타',
         '이유': '한 줄'}
    예외가 나면 그 문제는 판정을 비워 두고(안 푼 문제) 이유에 오류를 적는다.
    """
    return {"판정": "정상", "점수": 0.0, "위치": [], "근거_종류": "", "이유": "예시: 아직 판단 코드 없음"}


def read(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="풀이 틀: 판단() 함수로 평가셋을 풀어 답안 CSV를 만든다")
    ap.add_argument("--이름", required=True, help="답안 이름 → 답안/답안_<이름>.csv (예: 동현_딥시크_v1)")
    ap.add_argument("--처음", type=int, default=0, help="앞에서 이만큼만 (0이면 전부)")
    ap.add_argument("--긴변", type=int, default=0, help="사진을 이 길이(긴 변, px)로 줄여 판단()에 넘김 (0이면 원래 크기)")
    ap.add_argument("--이어서", action="store_true", help="이미 답한 문제는 건너뜀")
    ap.add_argument("--평가셋", default=str(HERE))
    a = ap.parse_args()
    base = Path(a.평가셋)
    probs = read(base / "문제목록.csv")[: a.처음 or None]
    out = base / "답안" / f"답안_{a.이름}.csv"
    out.parent.mkdir(exist_ok=True)
    done = {r["문제_id"] for r in read(out) if (r.get("판정") or "").strip()} if a.이어서 and out.exists() else set()
    keep = [r for r in read(out) if r["문제_id"] in done] if done else []
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(keep)
        for i, q in enumerate(probs, 1):
            if q["문제_id"] in done:
                continue
            t0 = time.time()
            im = Image.open(base / q["파일"]).convert("RGB")
            W, H = im.size
            s = 1.0
            if a.긴변 and max(W, H) > a.긴변:
                s = a.긴변 / max(W, H)
                im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
            info = {"문제_id": q["문제_id"], "문서_종류": q["문서_종류"], "가로": W, "세로": H, "파일": str(base / q["파일"])}
            try:
                r = 판단(im, info) or {}
            except Exception as e:                       # 한 문제가 실패해도 나머지는 계속
                r = {"판정": "", "이유": f"오류: {type(e).__name__} {e}"[:200]}
            boxes = ";".join(f"{round(x / s)},{round(y / s)},{round(w_ / s)},{round(h / s)}"
                             for x, y, w_, h in (r.get("위치") or []))
            w.writerow({"문제_id": q["문제_id"], "판정": r.get("판정", ""), "점수": r.get("점수", ""), "위치": boxes,
                        "근거_종류": r.get("근거_종류", ""), "이유": (r.get("이유") or "").replace("\n", " ")})
            f.flush()
            print(f"{i}/{len(probs)} {q['문제_id']} {r.get('판정') or '(안 풂)'} · {time.time() - t0:.1f}초", flush=True)
    print(f"→ {out}\n채점: python 채점.py {out.relative_to(base)}")


if __name__ == "__main__":
    main()
