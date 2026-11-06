"""영역을 정답 칸과 다르게 그렸을 때(웹에서 손으로 그리는 상황) 성능 변화. 검증용 절반 중 200쌍."""
import json, sys, numpy as np
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from detector import detect, load_config
D = ROOT / "dataset"
VARIANTS = {"정답 칸 그대로": (0, 0, 0), "사방 4px 넓게": (4, 4, 0), "사방 10px 넓게": (10, 10, 0),
            "가로세로 30% 넓게": ("30%", "30%", 0), "오른쪽으로 5px 밀림": (0, 0, 5)}
CFG = load_config()
def boxes(b):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    out = {}
    for k, (px, py, sh) in VARIANTS.items():
        px = int(w * .15) if px == "30%" else px; py = int(h * .15) if py == "30%" else py
        out[k] = (x - px + sh, y - py, w + 2 * px, h + 2 * py)
    return out
def work(base):
    b = json.load(open(D / f"labels/meta/{base}_ai_inpaint.json"))["forged_regions"][0]["bbox"]
    t = Image.open(D / f"tampered/ai_inpainting/{base}_ai_inpaint.png").convert("RGB")
    o = Image.open(next(p for p in (D / "original/bankbook").glob(base + ".*") if p.suffix.lower() in (".jpg", ".png"))).convert("RGB")
    r = {}
    for k, bb in boxes(b).items():
        try: r[k] = (detect(t, bbox=bb, config=CFG)["verdict"]["suspicious"], detect(o, bbox=bb, config=CFG)["verdict"]["suspicious"])
        except Exception: r[k] = None
    return r
if __name__ == "__main__":
    recs = [json.loads(l) for l in open(ROOT / "outputs/reports/detector_measurements.jsonl")]
    bases = [r["base"] for r in recs if "skip" not in r]
    idx = np.random.default_rng(0).permutation(len(bases)); test = [bases[i] for i in idx[len(bases) // 2:]][:200]
    with Pool(4) as p: res = p.map(work, test, chunksize=4)
    print(f"\n[영역을 다르게 그렸을 때, 검증용 {len(test)}쌍]  정탐 / 오탐 / 정확도")
    for k in VARIANTS:
        v = [r[k] for r in res if r.get(k)]
        tp = np.mean([a for a, _ in v]); fp = np.mean([b for _, b in v])
        print(f"  {k:<14} {tp*100:5.1f}%  {fp*100:5.1f}%  {(tp+1-fp)*50:5.1f}%")
