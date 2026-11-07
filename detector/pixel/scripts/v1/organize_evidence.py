"""
근거 이미지를 판정 결과별 폴더로 정리하고 요약표·안내문을 만든다.

  evidence/정탐/   위조본을 위조로 맞힌 샘플
  evidence/미탐/   위조본을 놓친 샘플
  evidence/오탐/   같은 위치 원본을 위조로 잘못 판정한 샘플 (정탐 또는 미탐 폴더에도 같은 이미지가 있음)

사용: python organize_evidence.py   (catalog_v1/labels.csv의 판정결과 열 기준)
"""
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EV = ROOT / "outputs/reports/evidence"
LABELS = ROOT / "outputs/reports/catalog_v1/labels.csv"


def find_any(sid):
    hits = [p for p in EV.rglob(f"{sid}.jpg") if "_예전" not in str(p)]
    return hits[0] if hits else None


def main():
    rows = list(csv.DictReader(open(LABELS, encoding="utf-8-sig")))
    for d in ("정탐", "미탐", "오탐"):
        (EV / d).mkdir(parents=True, exist_ok=True)
    summary, counts = [], {"정탐": 0, "미탐": 0, "오탐": 0}
    for r in rows:
        sid, rt, ro = r["id"], r["판정결과_위조본"], r["판정결과_원본"]
        src = find_any(sid)
        if src is None:
            continue
        main_dst = EV / rt / f"{sid}.jpg"
        if src != main_dst:
            shutil.copy2(src, main_dst)
        counts[rt] += 1
        fp_dst = EV / "오탐" / f"{sid}.jpg"
        if ro == "오탐":
            shutil.copy2(main_dst, fp_dst); counts["오탐"] += 1
        elif fp_dst.exists():
            fp_dst.unlink()
        other = EV / ("미탐" if rt == "정탐" else "정탐") / f"{sid}.jpg"
        if other.exists():
            other.unlink()
        desc = r["description"].split(" (")[0].replace(" → ", "에서 ") + "로 변경"
        summary.append([sid, desc, rt, "오탐" if ro == "오탐" else "정상", f"{rt}/{sid}.jpg"])
    # 예전 구조(번호 붙은 폴더, all) 정리
    for d in list(EV.iterdir()):
        if d.is_dir() and (d.name[:2] in ("1_", "2_", "3_", "4_") or d.name == "all"):
            shutil.rmtree(d)
    with open(EV / "요약표.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["샘플", "변경 내용", "위조본 판정", "원본 판정", "근거 이미지"])
        w.writerows(summary)
    (EV / "읽어보기.txt").write_text(f"""판정 근거 이미지 (샘플 {len(summary)}개)

폴더
  정탐  {counts['정탐']}장  위조본을 위조로 맞힘
  미탐  {counts['미탐']}장  위조본을 정상으로 판정해 놓침
  오탐  {counts['오탐']}장  위조하지 않은 원본(같은 위치)을 위조로 잘못 판정
        오탐 샘플은 위조본 결과에 따라 정탐 또는 미탐 폴더에도 같은 이미지가 있음
  요약표.csv  샘플별 변경 내용, 위조본 판정, 원본 판정, 이미지 위치 (엑셀에서 정렬, 필터 가능)

이미지 한 장 보는 법
  위쪽   왼쪽 원본, 오른쪽 위조본. 빨간 박스가 검사한 칸. 제목 옆 색 글씨가 판정 결과
         (위조본: 정탐 또는 미탐 / 원본: 정상 또는 오탐)
         아래 글씨는 탐지법별 측정값, 걸림/통과, 정상 문서 중 몇 % 수준인지
  아래쪽 탐지기가 실제로 비교한 패치들
         패치 줄       의심 영역과 비교 패치 1~5
         노이즈 지도 줄 밝을수록 노이즈가 많음. 위조된 글자는 스캔 노이즈가 없어 까맣게 나옴
""", encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
