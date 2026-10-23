"""육안 채점 페이지 만들기 (8단계 간단 평가).

같은 원본에 대해 각 방식 결과를 '원본 | 결과' 한 쌍으로, 이름을 가린 채(A, B, C...) 순서를 섞어 보여준다.
배경, 글자, 질감을 0(티 안 남) / 1(자세히 보면 보임) / 2(바로 보임)로 채점하고
'결과 보기'를 누르면 방식 이름이 드러나며 방식별 평균이 나온다. 'CSV 저장'으로 채점표를 받는다.

    python scripts/generate/forge_v2_review.py        -> outputs/고도화/육안채점.html (브라우저로 열기)
"""
import base64
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_v2 as F  # noqa: E402
import forge_v2_compare as C  # noqa: E402

STAGES = ["00_기존", "01_배경", "04_폰트맞추기", "06_AI블렌딩"]


def tile(im, crop, scale):
    c = im.crop(crop)
    c = c.resize((c.width * scale, c.height * scale), Image.LANCZOS)
    buf = io.BytesIO()
    c.convert("RGB").save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    cols = []
    for st in STAGES:
        if (F.OUT_ROOT / st).exists():
            cols += C._folders(F.OUT_ROOT, st)
    rng = random.Random(0)
    items = []
    for row in F.load_list():
        sid = row["id"]
        orig = Image.open(F.ROOT / row["original"]).convert("RGB")
        cands = []
        for n, p in cols:
            f, m = C._file(p, sid), p / f"{sid}_mask.png"
            if f is None or not m.exists():
                continue
            pair = C._pair(orig, Image.open(f).convert("RGB"), C._mask_box(m), "왼쪽 원본 | 오른쪽 결과")
            buf = io.BytesIO(); pair.save(buf, "JPEG", quality=88)
            cands.append({"method": n, "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
        rng.shuffle(cands)
        items.append({"id": sid, "cands": cands, "change": ""})
    html = PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False))
    out = F.OUT_ROOT / "육안채점.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out} ({len(items)}장, 방식 {len(cols)}개)")


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>육안 채점</title>
<style>
:root{--bg:#f6f6f4;--fg:#222;--card:#fff;--line:#ddd;--muted:#666}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"Apple SD Gothic Neo",sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:2}
main{max-width:980px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:16px}
.row{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;margin:8px 0;padding-top:8px;border-top:1px solid var(--line)}
img{max-width:100%;display:block;border:1px solid var(--line)}
.tag{font-weight:600;width:190px;flex:none}
select{font-size:14px}
.muted{color:var(--muted)}
table{border-collapse:collapse;margin-top:8px}td,th{border:1px solid var(--line);padding:4px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
button{font-size:14px;padding:6px 12px}
</style></head><body>
<header><b>육안 채점</b><span class="muted">0 티 안 남 / 1 자세히 보면 보임 / 2 바로 보임. 방식 이름은 결과 보기 전까지 가려짐</span>
<button onclick="reveal()">결과 보기</button><button onclick="save()">CSV 저장</button><span id="prog" class="muted"></span></header>
<main><div id="summary"></div><div id="list"></div></main>
<script>
const DATA=__DATA__;const KEYS=["배경","글자","질감"];const S={};
const list=document.getElementById("list");
DATA.forEach((it,i)=>{const c=document.createElement("div");c.className="card";
 let h=`<div><b>${i+1}. ${it.id}</b> <span class="muted">${it.change}</span></div>`;
 it.cands.forEach((cd,j)=>{const L=String.fromCharCode(65+j);
  h+=`<div class="row"><div class="tag"><span>${L}</span><div class="muted m" data-m="${cd.method}"></div></div><img src="${cd.src}"><div>`+
  KEYS.map(k=>`${k} <select data-id="${it.id}" data-method="${cd.method}" data-k="${k}"><option value="">-</option><option>0</option><option>1</option><option>2</option></select>`).join(" ")+`</div></div>`;});
 c.innerHTML=h;list.appendChild(c);});
document.addEventListener("change",e=>{const t=e.target;if(!t.dataset.k)return;
 (S[t.dataset.id+"|"+t.dataset.method]??={})[t.dataset.k]=t.value;prog();});
function prog(){const n=document.querySelectorAll("select").length;const f=[...document.querySelectorAll("select")].filter(s=>s.value!=="").length;document.getElementById("prog").textContent=`${f}/${n} 채점`;}
function reveal(){document.querySelectorAll(".m").forEach(d=>d.textContent=d.dataset.m);
 const agg={};Object.entries(S).forEach(([k,v])=>{const m=k.split("|")[1];agg[m]??={};KEYS.forEach(x=>{if(v[x]!==undefined&&v[x]!==""){(agg[m][x]??=[]).push(+v[x]);}});});
 let h="<div class='card'><b>방식별 평균 점수</b> <span class='muted'>(낮을수록 티 안 남)</span><table><tr><th>방식</th>"+KEYS.map(k=>`<th>${k}</th>`).join("")+"<th>채점 수</th></tr>";
 Object.entries(agg).forEach(([m,v])=>{h+=`<tr><td>${m}</td>`+KEYS.map(k=>{const a=v[k]||[];return `<td>${a.length?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(2):"-"}</td>`}).join("")+`<td>${Math.max(...KEYS.map(k=>(v[k]||[]).length))}</td></tr>`;});
 document.getElementById("summary").innerHTML=h+"</table></div>";window.scrollTo(0,0);}
function save(){let csv="﻿id,방식,"+KEYS.join(",")+"\n";Object.entries(S).forEach(([k,v])=>{const [id,m]=k.split("|");csv+=`${id},${m},`+KEYS.map(x=>v[x]??"").join(",")+"\n";});
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="육안채점_결과.csv";a.click();}
prog();
</script></body></html>"""

if __name__ == "__main__":
    main()
