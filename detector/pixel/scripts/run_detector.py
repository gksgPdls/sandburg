"""
(호환용) 예전 사용법 그대로 동작 — 실제 탐지는 detector 패키지가 한다.

    python run_detector.py --image 의심문서.jpg --bbox 893 224 122 26
    = python -m detector --image 의심문서.jpg --bbox 893 224 122 26   (프로젝트 루트에서)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from detector.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
