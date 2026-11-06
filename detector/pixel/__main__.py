"""
CLI: 이미지 1장 + 의심영역 → 탐지 결과

    python -m detector --image 의심문서.jpg --bbox 893 224 122 26
    python -m detector --image 의심문서.jpg --mask 마스크.png --json
    python -m detector --image 문서.jpg --auto        # 좌표 없이 OCR로 금액·날짜 칸 자동 탐색 (tesseract 필요)
"""
import argparse
import json

from .v1.core import detect


def main():
    ap = argparse.ArgumentParser(prog="python -m detector", description="의심 영역 위변조 탐지")
    ap.add_argument("--image", required=True)
    ap.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--mask", help="의심영역 마스크 png (bbox 대신)")
    ap.add_argument("--config", help="기준값 파일 (기본: detector/v1/config.json)")
    ap.add_argument("--json", action="store_true", help="JSON으로 출력")
    ap.add_argument("--auto", action="store_true", help="좌표 없이 OCR로 금액·날짜 칸을 찾아 전부 검사")
    a = ap.parse_args()
    if a.auto:
        from .v1.locate import locate
        r = locate(a.image)
        if a.json:
            for c in r["candidates"]:
                c.pop("detail", None)
            print(json.dumps(r, ensure_ascii=False, indent=2)); return
        print(r["summary"])
        for c in r["candidates"][:5]:
            print(f"  {c['text']:<14} 위치={c['bbox']}  희귀도={c['rarity']:.4f}  걸린 탐지법={', '.join(c['flagged_methods']) or '-'}")
        return
    if not a.bbox and not a.mask:
        ap.error("--bbox 또는 --mask 중 하나는 필요 (좌표가 없으면 --auto)")
    from PIL import Image
    from .v1.core import load_config
    res = detect(a.image, bbox=a.bbox, mask=Image.open(a.mask) if a.mask else None,
                 config=load_config(a.config) if a.config else None)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    print(res["verdict"]["summary"])
    for key, m in res["methods"].items():
        if not m["enabled"]:
            print(f"  -  {m['name']:<22} 사용 안 함 ({m.get('why_disabled', '')})")
            continue
        mark = "⚠" if m["flagged"] else "✓"
        print(f"  {mark}  {m['name']:<22} 측정값={m.get('score')}  (안쪽={m.get('inside')}, 비교={m.get('outside')})")


if __name__ == "__main__":
    main()
