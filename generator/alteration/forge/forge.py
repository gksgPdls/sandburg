"""위조 이미지 만들기 한 번에 (슈퍼 스크립트). 단계별 스크립트를 순서대로 부른다.

  1. 목록   --src 가 폴더면: 사진마다 바꿀 칸을 자동으로 고름 (3_logic/logic_list.py, 논리 맞추기 규칙)
            --src 가 CSV면: 그 목록 그대로 (칸 좌표와 새 값을 사람이 적은 목록)
  2. 생성   2_render/forge_v2.py run (1단계 지우기, 4단계 글꼴, 5단계 저장 + B1~B6, C1~C2)
  3. 비교   2_render/forge_v2.py compare (원본 | 결과 비교 이미지, 점수)

예)
  python generator/alteration/forge/forge.py --src ~/Desktop/새사진 --out 새사진
  python generator/alteration/forge/forge.py --src outputs/고도화/원본목록.csv --out 최종_내방식 --limit 10
  python generator/alteration/forge/forge.py --src ~/Desktop/새사진 --out 새사진 --logic 엄격 --off b6,c2
설정 기본값: generator/alteration/forge/forge_config.json
결과: outputs/고도화/<out>/AI/ (위조 이미지, 마스크, 생성기록.csv), 비교이미지/, 점수.csv"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFF_FLAGS = {"sd": "--no-sd", "c6": "--no-c6", "tilt": "--no-tilt", "persp": "--no-persp", "b3": "--no-b3", "b4": "--no-b4",
             "b6": "--no-b6", "c1": "--no-c1", "c2": "--no-c2", "c5": "--no-c5"}


def run(cmd, env=None):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env)
    if r.returncode:
        sys.exit(r.returncode)


def main():
    cfg = json.loads((HERE / "forge_config.json").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="원본 사진 폴더, 또는 목록 CSV")
    ap.add_argument("--out", required=True, help="결과 폴더 이름 (outputs/고도화/<out>)")
    ap.add_argument("--logic", choices=["끔", "시도", "엄격"], default=cfg["논리"],
                    help="폴더일 때. 끔: 칸 하나만 바꿈 / 시도: 관계(합계·날짜 순서)를 맞추고 못 찾으면 칸 하나 (개발 중) / 엄격: 관계를 찾은 사진만 (개발 중)")
    ap.add_argument("--ocr", choices=["rapid", "tesseract"], default=cfg["ocr"])
    ap.add_argument("--steps", default=cfg["steps"])
    ap.add_argument("--off", default=",".join(cfg["끄기"]), help="끌 기능 (쉼표로): " + ", ".join(OFF_FLAGS))
    ap.add_argument("--limit", type=int, default=cfg["limit"], help="앞에서 몇 장만 (0이면 전부)")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--resume", action="store_true", help="이미 만든 이미지는 건너뜀 (끊긴 뒤 이어서)")
    a = ap.parse_args()
    py = sys.executable
    src = Path(a.src).expanduser()
    if not src.is_absolute():
        src = ROOT / src
    # 1. 목록
    if src.is_dir():
        name = f"{a.out}"
        cmd = [py, HERE / "3_logic" / "logic_list.py", "--src", src, "--name", name, "--ocr", a.ocr, "--want", a.limit]
        if a.logic == "엄격":
            cmd.append("--strict")
        elif a.logic == "끔":
            cmd.append("--no-logic")
        run(cmd)
        lst = ROOT / "outputs" / "고도화" / f"원본목록_{name}.csv"
    elif src.suffix.lower() == ".csv":
        lst = src
    else:
        sys.exit(f"--src 는 폴더나 CSV여야 함: {src}")
    # 2. 생성
    env = dict(os.environ, FORGE_LIST=str(lst), FORGE_LIMIT=str(a.limit))
    offs = [o.strip() for o in a.off.split(",") if o.strip()]
    bad = [o for o in offs if o not in OFF_FLAGS]
    if bad:
        sys.exit(f"모르는 끄기 항목: {bad}. 가능: {list(OFF_FLAGS)}")
    run([py, HERE / "2_render" / "forge_v2.py", "run", "--branch", "ai", "--steps", a.steps, "--out", a.out, "--name", "AI",
         *[OFF_FLAGS[o] for o in offs], *(["--resume"] if a.resume else [])], env=env)
    # 3. 비교
    if not a.no_compare:
        run([py, HERE / "2_render" / "forge_v2.py", "compare", "--out", a.out, "--prev", "_없음"], env=env)
    print(f"완료: outputs/고도화/{a.out}/")


if __name__ == "__main__":
    main()
