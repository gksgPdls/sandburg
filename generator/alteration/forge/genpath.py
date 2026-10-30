"""scripts/generate 경로 도우미 (2026-10-30 폴더 정리).

변조 코드가 역할별 하위 폴더(1_vlm, 2_render, 3_logic, 4_compare)로 나뉘어 있어서,
어느 파일에서 실행하든 서로를 이름만으로 import 할 수 있게 경로를 넣는다.

    import genpath            # scripts/generate 가 경로에 있을 때
    genpath.ROOT              # 저장소 루트 (repo)
"""
import sys
from pathlib import Path

GEN = Path(__file__).resolve().parent          # scripts/generate
ROOT = GEN.parents[1]                          # 저장소 루트
SUBDIRS = sorted(p for p in GEN.iterdir() if p.is_dir() and p.name[:1].isdigit())

for _d in [*reversed(SUBDIRS), GEN]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
for _d in (ROOT, ROOT / "dataset" / "scripts"):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))
