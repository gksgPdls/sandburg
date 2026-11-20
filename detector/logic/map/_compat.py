"""로직 엔진 패키지 찾기: 팀 저장소(logic_detector.logic)가 있으면 그걸, 없으면 이 저장소의 detector.logic 을 같은 이름으로 쓴다."""
import importlib
import sys
import types
from pathlib import Path

try:
    import logic_detector.logic  # noqa: F401
except ImportError:
    repo = Path(__file__).resolve().parents[3]          # .../detector/logic/map → 저장소 맨 위
    sys.path.insert(0, str(repo))
    L = importlib.import_module("detector.logic")
    pkg = types.ModuleType("logic_detector")
    pkg.logic = L
    sys.modules["logic_detector"] = pkg
    sys.modules["logic_detector.logic"] = L
    for sub in ("rules", "focus", "refdata", "ocr", "reading", "read_receipt", "read_diagnosis"):
        try:
            sys.modules[f"logic_detector.logic.{sub}"] = importlib.import_module(f"detector.logic.{sub}")
        except ImportError:
            pass
