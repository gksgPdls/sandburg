#!/usr/bin/env python3
"""정상 대조군: 조작하지 않은 원본을 samples/normal/ 에 등록한다.

조작본만 있으면 "전부 위조"라고 답하는 엔진이 만점을 받으므로 오탐률을 재려면 필요하다.
문서 종류 비율을 조작본과 비슷하게 맞춰, 문서 종류만으로 정상·조작을 맞히지 못하게 한다.
조작본의 원본으로 쓴 문서는 고르지 않는다.

사용: make_normals.py DOC210=20 DOC205=4 DOC203=6 [--seed 0]
"""

import argparse
import random

from labels import load_all
from pick_session import DOC_TYPE, POOL, manifest
from register_sample import register


def main():
    p = argparse.ArgumentParser(description="정상 대조군 등록")
    p.add_argument("counts", nargs="+", help="DTB코드=장수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--author", default="김동현")
    a = p.parse_args()

    ids = manifest()
    used = {r.get("source", "") for r in load_all()}
    made = []
    for spec in a.counts:
        dtb, n = spec.split("=")
        files = sorted(f for f in (POOL / dtb).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        random.Random(f"{a.seed}-{dtb}").shuffle(files)
        got = 0
        for f in files:
            if got == int(n):
                break
            pid = ids.get(f.relative_to(POOL).as_posix(), "")
            if not pid or pid in used:
                continue
            sid = register(method="normal", axis="none", original=f,
                           description="조작 없음. real_docs 원본을 그대로 복사",
                           detection="정상 문서. 어떤 축도 반응하면 안 됨 (오탐률 측정용)",
                           doc_type=DOC_TYPE.get(dtb, dtb), external_ai="N", author=a.author,
                           source=pid, notes=f"make_normals.py seed={a.seed}")
            used.add(pid)
            made.append(sid)
            got += 1
        print(f"{dtb}: {got}장")
    print(f"정상 대조군 {len(made)}개 → samples/normal/  {made[0]} ~ {made[-1]}")


if __name__ == "__main__":
    main()
