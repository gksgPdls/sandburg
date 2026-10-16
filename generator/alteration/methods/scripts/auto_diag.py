#!/usr/bin/env python3
"""진단서 원본 풀에서 KCD 코드를 자동으로 바꿔 samples/kcd_swap/ 에 등록한다 (콘텐츠 로직, 케이스 4).

1. 진단서 전체를 로컬 OCR(tesseract)로 읽어 `D18.0`, `(D18.0)` 같은 코드 단어를 찾는다
2. 괄호를 뺀 코드 부분만 지우고, 다른 장(첫 글자)의 코드로 다시 쓴다.
   바꿀 코드는 kcd_swap_list.csv (골절·암·중대질병 — 보험금이 커지는 쪽)
3. 괄호를 포함한 단어 범위를 다시 읽어, 원래 단어에서 코드만 바뀐 글이 나올 때만 채택한다
   (괄호를 같이 지웠거나 글자가 뭉개졌으면 되돌린다)

문서 내용은 출력하지 않는다.

사용: auto_diag.py 5 [--seed 0]
"""

import argparse
import csv
import io
import os
import random
import re
import subprocess
import tempfile

import cv2
import numpy as np

from labels import ROOT, load_all
from pick_session import (POOL, TMP, CODE_CHARS, DOC_TYPE, changed_mask, kcd_choice, manifest,
                          read_box, rewrite)
from register_sample import register

CODE = re.compile(r"^(\(?)([A-Z]\d{2}\.\d{1,2})(\)?)$")


def find_codes(gray):
    """페이지에서 KCD 코드 단어들: [(단어, 코드, 단어 외곽, 코드만의 외곽)]."""
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, big)
        out = subprocess.run(["tesseract", path, "-", "-l", "eng", "--psm", "11",
                              "-c", "tessedit_char_whitelist=" + CODE_CHARS + "()", "tsv"],
                             capture_output=True).stdout.decode("utf-8", "replace")
    finally:
        os.remove(path)
    found = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        t = (r.get("text") or "").strip()
        m = CODE.match(t)
        if not m or float(r.get("conf") or 0) < 70:
            continue
        x, y, w, h = (int(r[k]) // 2 for k in ("left", "top", "width", "height"))
        cw = w / len(t)
        lead, trail = len(m.group(1)), len(m.group(3))
        # 괄호 폭만큼 안쪽으로. 괄호를 지우지 않도록 1px 더 안쪽
        cx = int(round(x + lead * cw)) + (1 if lead else 0)
        cw_total = int(round(w - (lead + trail) * cw)) - (1 if lead else 0) - (1 if trail else 0)
        found.append((t, m.group(2), (x - 2, y - 2, w + 4, h + 4), (cx, y - 1, cw_total, h + 2)))
    return found


def orient(img):
    best = None
    ks = (0,) if img.shape[0] >= img.shape[1] else (-1, 1)
    for k in ks:
        r = np.ascontiguousarray(np.rot90(img, k))
        codes = find_codes(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))
        if best is None or len(codes) > len(best[1]):
            best = (r, codes, k)
    return best


def try_swap(img, codes, rng):
    rng.shuffle(codes)
    for word, old, word_box, code_box in codes:
        entry = {"text_i": rng.randrange(1000)}
        new = kcd_choice(entry, old)
        before = img.copy()
        if not rewrite(img, code_box, new["code"], seed=rng.randint(0, 9999)):
            img[:] = before
            continue
        expect = word.replace(old, new["code"])
        x, y, w, h = word_box
        wide = (x - int(w * 0.4), y, w + int(w * 0.4), h)  # 새 코드가 왼쪽으로 조금 길어질 수 있다
        got = read_box(img, wide, CODE_CHARS + "()")
        if got != expect:
            img[:] = before
            continue
        cx, cy, cw, ch = code_box
        return {
            "description": f"진단서 질병 코드 {old} → {new['code']} (x={cx}, y={cy}). 병명은 그대로. "
                           f"코드만 배경으로 지우고 다시 씀. 새 코드는 '{new['meaning']}' ({new['why']})",
            "detection": "콘텐츠 로직: 병명과 KCD 코드가 맞지 않음. 공개 마스터(KCD 코드표) 대조로 결정적 판정 가능 "
                         "(방법론 케이스 4). 글리프: 다시 쓴 코드의 폰트가 다름",
            "notes": f"code={cx},{cy},{cw},{ch}",
        }
    return None


def main():
    p = argparse.ArgumentParser(description="진단서 KCD 코드 자동 교체")
    p.add_argument("count", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--author", default="김동현")
    a = p.parse_args()

    ids = manifest()
    used = {r.get("source", "") for r in load_all()}
    files = sorted(f for f in (POOL / "DOC203").iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rng = random.Random(a.seed)
    rng.shuffle(files)
    TMP.mkdir(parents=True, exist_ok=True)
    made, tried = [], 0
    for f in files:
        if len(made) == a.count:
            break
        pid = ids.get(f.relative_to(POOL).as_posix(), "")
        if pid in used:
            continue
        tried += 1
        img = cv2.imread(str(f))
        up, codes, k = orient(img)
        if not codes:
            continue
        before = up.copy()
        info = try_swap(up, codes, rng)
        if not info:
            continue
        mask = changed_mask(before, up)
        t_path, m_path = TMP / f"t_{os.getpid()}.jpg", TMP / f"m_{os.getpid()}.png"
        cv2.imwrite(str(t_path), np.ascontiguousarray(np.rot90(up, -k)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(m_path), np.ascontiguousarray(np.rot90(mask, -k)))
        if k:
            info["notes"] += f" 좌표는 세로로 돌린 이미지 기준 (np.rot90 k={k})"
        sid = register(method="kcd_swap", axis="content_logic", original=f, tampered=t_path, mask=m_path,
                       description=info["description"], detection=info["detection"],
                       doc_type=DOC_TYPE["DOC203"], difficulty="쉬움", external_ai="N",
                       author=a.author, source=pid, notes=f"auto_diag.py seed={a.seed}. {info['notes']}")
        t_path.unlink()
        m_path.unlink()
        used.add(pid)
        made.append(sid)
    print(f"kcd_swap: {len(made)}개 생성 (시도한 원본 {tried}장) → samples/kcd_swap/  {' '.join(made)}")


if __name__ == "__main__":
    main()
