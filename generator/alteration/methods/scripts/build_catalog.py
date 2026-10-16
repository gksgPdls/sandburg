#!/usr/bin/env python3
"""labels.csv 를 읽어 브라우저용 목록(catalog.html)을 만든다.

    build_catalog.py                          samples/ 아래 전부 → dataset/catalog.html
    build_catalog.py 내폴더/labels.csv ...     지정한 CSV 만 → 첫 CSV 옆의 catalog.html
    build_catalog.py a.csv -o 보기.html        출력 위치 지정

CSV 는 필수 열(id, target_axis, original, description)만 있으면 된다. 파일 칸은 CSV 위치 기준 경로.

스프레드시트 없이 샘플을 훑어보는 용도다. 이미지는 HTML에 넣지 않고 로컬 파일을
상대 경로로 가리키기만 한다. 인터넷으로 아무것도 불러오거나 보내지 않는다.
catalog.html 은 git 에 올리지 않는다.
"""

import argparse
import html
import os
from pathlib import Path

from labels import AXIS_NAME, ROOT, load_all, path_of, preview_of

OUT_DIR = ROOT  # 이미지 상대 경로의 기준. main 에서 출력 위치로 바꾼다

# 조작 방법(폴더) 한글 이름. 목록에 없는 폴더는 폴더 이름 그대로 나온다
METHOD_NAME = {
    "copymove": "금액 칸 복제", "digit_combo": "숫자 조합", "date_copy": "날짜 복제",
    "amount_sum_break": "합계 불일치", "amount_digit": "자릿수 조작", "date_conflict": "날짜 모순",
    "kcd_swap": "KCD 코드 교체", "date_shift": "날짜 치환", "add_diag": "여백에 진단 추가",
    "add_text": "여백 추가 (수동)", "amount_sum_fixed": "금액+합계 보정", "normal": "정상 원본",
    "edms": "EDMS 변형",
}
AXIS_ORDER = ["copy_move", "content_logic", "glyph", "none"]

SHOWN = {"id", "method", "target_axis", "doc_type", "source", "source_file", "source_folder", "original", "tampered", "mask",
         "preview", "description", "detection", "difficulty", "external_ai", "author",
         "created", "notes"}

STYLE = """
*{box-sizing:border-box}
body{font-family:-apple-system,'Apple SD Gothic Neo',sans-serif;margin:0;background:#f4f5f7;color:#1d1d1f;font-size:15px;line-height:1.6}
main{max-width:1280px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:#6e6e73;font-size:13px}
.summary{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
.summary button{border:1px solid #d2d2d7;background:#fff;border-radius:10px;padding:8px 14px;cursor:pointer;font-size:14px;text-align:left}
.summary button b{display:block;font-size:20px}
.summary button.on{border-color:#1d1d1f;box-shadow:inset 0 0 0 1px #1d1d1f}
.summary button.dim{opacity:.35}
.row-label{font-size:12px;font-weight:600;color:#6e6e73;margin:14px 0 -8px}
.methods button{padding:6px 12px;font-size:13px}
.methods button b{font-size:15px;display:inline;margin-left:6px}
.count{font-size:13px;color:#6e6e73;margin:4px 0 14px}
.card{background:#fff;border:1px solid #e5e5ea;border-radius:12px;padding:18px;margin-bottom:14px;display:grid;grid-template-columns:minmax(300px,2fr) 3fr;gap:20px}
.id{font-size:20px;font-weight:700;margin-right:8px}
.tags{margin:2px 0 12px}
.tag{display:inline-block;font-size:12px;padding:2px 9px;border-radius:10px;background:#f0f0f3;margin:0 4px 4px 0}
.tag.axis{background:#1d1d1f;color:#fff}
.tag.ai{background:#ffe5e5;color:#b00020}
.label{font-size:12px;font-weight:600;color:#6e6e73;margin-top:10px}
.text{white-space:pre-wrap;word-break:break-word}
.extra{font-size:13px;color:#6e6e73;margin-top:10px}
.imgs{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;align-content:start}
.imgs figure{margin:0}
.imgs img{width:100%;height:300px;object-fit:contain;background:#f4f5f7;border-radius:8px;display:block}
.imgs .none{height:300px;border-radius:8px;background:#f4f5f7;display:flex;align-items:center;justify-content:center;color:#aeaeb2}
figcaption{font-size:12px;color:#6e6e73;text-align:center;margin-top:4px}
figcaption a{color:inherit}
@media (max-width:860px){.card{grid-template-columns:1fr}.imgs img,.imgs .none{height:200px}}
"""

SCRIPT = """
let cur={axis:'all',method:'all'};
function pick(kind,val){
  cur[kind]=val;
  if(kind==='axis') cur.method='all';
  let shown=0;
  document.querySelectorAll('.card').forEach(c=>{
    const ok=(cur.axis==='all'||c.dataset.axis===cur.axis)&&(cur.method==='all'||c.dataset.method===cur.method);
    c.style.display=ok?'':'none'; shown+=ok;
  });
  document.querySelectorAll('.axes button').forEach(b=>b.classList.toggle('on',b.dataset.v===cur.axis));
  document.querySelectorAll('.methods button').forEach(b=>{
    b.classList.toggle('on',b.dataset.v===cur.method);
    const has=b.dataset.v==='all'||cur.axis==='all'||(b.dataset.axes||'').split(' ').includes(cur.axis);
    b.classList.toggle('dim',!has);
  });
  document.getElementById('count').textContent=shown+'개 표시 중';
}
"""


def rel(p):
    return html.escape(os.path.relpath(p, OUT_DIR).replace(os.sep, "/"))


def figure(path, caption, alt_link=None):
    if path is None or not path.exists():
        return f'<figure><div class="none">없음</div><figcaption>{caption}</figcaption></figure>'
    extra = f' · <a href="{rel(alt_link)}" target="_blank">표시 없이 보기</a>' if alt_link else ""
    return (f'<figure><a href="{rel(path)}" target="_blank"><img src="{rel(path)}" loading="lazy"></a>'
            f'<figcaption>{caption}{extra}</figcaption></figure>')


def card(r):
    e = html.escape
    axis = r.get("target_axis", "")
    tags = [f'<span class="tag axis">{e(AXIS_NAME.get(axis, axis))}</span>']
    method = r.get("method", "")
    tags += [f'<span class="tag">{e(METHOD_NAME.get(method, method))}</span>'] if method else []
    tags += [f'<span class="tag">{e(r[c])}</span>' for c in ("doc_type", "difficulty") if r.get(c)]
    if r.get("external_ai", "").upper() == "Y":
        tags.append('<span class="tag ai">외부 AI 투입</span>')

    info = [f'<div><span class="id">{e(r.get("id", ""))}</span></div><div class="tags">{"".join(tags)}</div>']
    for col, label in (("description", "조작"), ("detection", "탐지"), ("notes", "메모")):
        if r.get(col):
            info.append(f'<div class="label">{label}</div><div class="text">{e(r[col])}</div>')
    extras = [f"{e(k)}: {e(v)}" for k, v in r.items()
              if not k.startswith("_") and k not in SHOWN and v]
    who = " · ".join(e(r[c]) for c in ("author", "created") if r.get(c))
    if extras or who:
        info.append(f'<div class="extra">{" · ".join(extras + ([who] if who else []))}</div>')

    tampered, preview = path_of(r, "tampered"), preview_of(r)
    if preview is not None and preview.exists():
        shown = figure(preview, "조작본 (빨간 표시 = 조작 위치)", alt_link=tampered)
    else:
        shown = figure(tampered, "조작본")
    imgs = figure(path_of(r, "original"), "원본") + shown + figure(path_of(r, "mask"), "정답 마스크")
    return (f'<div class="card" data-axis="{e(axis)}" data-method="{e(method)}"><div>{"".join(info)}</div>'
            f'<div class="imgs">{imgs}</div></div>')


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="labels.csv → 브라우저용 목록")
    ap.add_argument("csv", nargs="*", help="labels.csv 경로 (없으면 samples/ 아래 전부)")
    ap.add_argument("-o", "--output", help="출력 HTML 경로")
    a = ap.parse_args()
    csvs = [Path(c).expanduser().resolve() for c in a.csv]
    out = Path(a.output).expanduser().resolve() if a.output else \
        (csvs[0].parent / "catalog.html" if csvs else ROOT / "catalog.html")
    OUT_DIR = out.parent
    rows = sorted(load_all(csvs or None), key=lambda r: r.get("id", ""))
    e = html.escape
    present = {r.get("target_axis", "") for r in rows}
    axes = [a for a in AXIS_ORDER if a in present] + sorted(present - set(AXIS_ORDER))
    axis_btn = [f'<button class="on" data-v="all" onclick="pick(\'axis\',\'all\')">전체<b>{len(rows)}</b></button>']
    for a in axes:
        n = sum(r.get("target_axis", "") == a for r in rows)
        axis_btn.append(f'<button data-v="{e(a)}" onclick="pick(\'axis\',\'{e(a)}\')">'
                        f'{e(AXIS_NAME.get(a, a))}<b>{n}</b></button>')
    # 조작 방법: 축 순서대로, 축 안에서는 개수 많은 순
    methods = {}
    for r in rows:
        m = r.get("method", "")
        methods.setdefault(m, {"n": 0, "axes": set()})
        methods[m]["n"] += 1
        methods[m]["axes"].add(r.get("target_axis", ""))
    def order(m):
        first = min((AXIS_ORDER.index(a) if a in AXIS_ORDER else 99) for a in methods[m]["axes"])
        return (m == "edms", first, -methods[m]["n"])
    method_btn = ['<button class="on" data-v="all" onclick="pick(\'method\',\'all\')">전체</button>']
    for m in sorted(methods, key=order):
        method_btn.append(f'<button data-v="{e(m)}" data-axes="{e(" ".join(sorted(methods[m]["axes"])))}" '
                          f'onclick="pick(\'method\',\'{e(m)}\')">{e(METHOD_NAME.get(m, m))}'
                          f'<b>{methods[m]["n"]}</b></button>')
    external = sum(r.get("external_ai", "").upper() == "Y" for r in rows)

    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>샘플 목록</title><style>{STYLE}</style></head><body><main>'
            f'<h1>위변조 테스트 샘플 목록</h1>'
            f'<div class="sub">외부 AI 투입 {external}건 / 상한 20건 · 사진을 누르면 원본 크기로 열린다 · '
            f'로컬 전용 (인터넷 연결 없음)</div>'
            f'<div class="row-label">탐지 축</div><div class="summary axes">{"".join(axis_btn)}</div>'
            f'<div class="row-label">조작 방법</div><div class="summary methods">{"".join(method_btn)}</div>'
            f'<div class="count" id="count">{len(rows)}개 표시 중</div>'
            f'{"".join(card(r) for r in rows)}<script>{SCRIPT}</script></main></body></html>')
    out.write_text(page, encoding="utf-8")
    print(f"{out}: 샘플 {len(rows)}개. 브라우저로 연다: open \"{out}\"")


if __name__ == "__main__":
    main()
