"""웹 화면용 분석: 탐지 결과 + 사람이 읽는 설명 문장 + 근거 이미지(확대, 비교 패치, 노이즈 지도)."""
import base64
import io

import numpy as np
from PIL import Image, ImageDraw
from skimage.restoration import denoise_wavelet

from ..v1 import features as F
from ..v1.core import bbox_to_mask, judge, load_config, measure
from ..methods import Context


def png64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _noise_map(p):
    x = p.astype(np.float32) / 255.0
    return np.abs(x - denoise_wavelet(x, channel_axis=None, rescale_sigma=True)) * 255.0


def _tile(arr, scale, vmax=None):
    a = arr.astype(np.float32)
    if vmax is not None:
        a = np.clip(a / vmax * 255.0, 0, 255)
    im = Image.fromarray(a.astype(np.uint8))
    return im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.NEAREST)


def _zoom(rgb, box, margin=60):
    x0, y0, x1, y1 = box
    c = (max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, rgb.width), min(y1 + margin, rgb.height))
    scale = max(1.0, min(4.0, 640 / (c[2] - c[0])))
    cr = rgb.crop(c).resize((int((c[2] - c[0]) * scale), int((c[3] - c[1]) * scale)), Image.LANCZOS)
    ImageDraw.Draw(cr).rectangle([(x0 - c[0]) * scale, (y0 - c[1]) * scale, (x1 - c[0]) * scale, (y1 - c[1]) * scale],
                                 outline=(242, 201, 76), width=4)
    return cr


def _rule_text(rule):
    lo, hi = rule.get("low"), rule.get("high")
    if lo is not None and hi is not None:
        return f"{lo:g} 미만 또는 {hi:g} 초과"
    if lo is not None:
        return f"{lo:g} 미만"
    if hi is not None:
        return f"{hi:g} 초과"
    return "-"


def _sentence(key, m):
    s, r = m.get("score"), m.get("rarity")
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        return "측정할 수 없음 (비교할 주변 영역이 없거나 비어 있음)"
    tail = f" 정상 문서 중 {r * 100:.1f}%만 이 정도로 다름." if r is not None else ""
    if key == "noise":
        body = (f"의심 영역 노이즈가 주변 비슷한 칸들보다 {1 / s:.1f}배 적음." if s < 1
                else f"의심 영역 노이즈가 주변 비슷한 칸들보다 {s:.1f}배 많음.")
    elif key == "fft":
        body = f"세밀한 질감(고주파) 비율이 주변보다 {abs(s) * 100:.1f}%p {'낮음' if s < 0 else '높음'}."
    elif key == "pixel_std":
        body = f"밝기 변화 폭이 주변의 {s:.2f}배."
    else:
        body = f"측정값 {s:.3f}."
    return body + tail


def analyze(rgb, bbox, config=None, padding=3):
    """rgb: PIL 이미지, bbox: (x, y, w, h)."""
    config = config or load_config()
    mask, box = bbox_to_mask((rgb.height, rgb.width), bbox, padding)
    res = judge(measure(rgb, mask=mask, config=config), config)
    ctx = Context(rgb, mask, box, params=config["methods"])
    inside, outs = ctx.comparison("noise")

    methods = []
    for key, m in res["methods"].items():
        mc = config["methods"][key]
        item = {"key": key, "name": m["name"], "enabled": m["enabled"], "flagged": m.get("flagged"),
                "score": m.get("score"), "inside": m.get("inside"), "outside": m.get("outside"),
                "rarity": m.get("rarity"), "rule": _rule_text(mc.get("rule", {})), "meaning": mc.get("meaning", "")}
        if m["enabled"]:
            item["sentence"] = _sentence(key, m)
        else:
            item["sentence"] = "사용 안 함: " + mc.get("why_disabled", "")
        methods.append(item)

    bw, bh = box[2] - box[0], box[3] - box[1]
    scale = max(1.0, min(6.0, 80 / max(bh, 1), 170 / max(bw, 1)))
    vmax = max(_noise_map(p).max() for p in [inside] + outs) * 0.6 + 1e-6
    tiles = []
    for i, p in enumerate([inside] + outs):
        tiles.append({"label": "의심 영역" if i == 0 else f"비교 {i}",
                      "noise": round(F.wavelet_noise_std(p), 2),
                      "patch": png64(_tile(p, scale)), "map": png64(_tile(_noise_map(p), scale, vmax)),
                      "used_by": "노이즈, FFT, 픽셀통계" if i == 1 else ("노이즈" if i > 1 else "")})
    v = res["verdict"]
    return {"bbox": list(bbox), "box": list(box), "verdict": v, "methods": methods, "zoom": png64(_zoom(rgb, box)),
            "tiles": tiles, "noise_formula": {"inside": res["methods"]["noise"].get("inside"),
                                              "outside": res["methods"]["noise"].get("outside"),
                                              "score": res["methods"]["noise"].get("score")}}
