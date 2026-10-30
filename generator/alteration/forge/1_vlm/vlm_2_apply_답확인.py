"""VLM 논리 맞추기 2단계: VLM 답(JSON)을 계산으로 확인하고, 통과한 것만 위조 목록 CSV로.

python scripts/generate/1_vlm/vlm_2_apply_답확인.py --name <이름>
읽는 것: outputs/고도화/_vlm/<이름>/<사진이름>_ocr.json, <사진이름>_vlm.json
결과:   outputs/고도화/원본목록_<이름>.csv  -> forge.py --src 이 CSV 로 위조 생성
        outputs/고도화/_vlm/<이름>/확인결과.csv  사진마다 통과/탈락과 이유

확인하는 것 (하나라도 틀리면 그 사진은 버림)
- 바꿀 번호가 OCR 목록에 있고, '원래' 값이 OCR이 읽은 값과 같음
- 새 금액: 자릿수·쉼표 형식 같음 / 새 날짜: 진짜 날짜, 형식 같음, 오늘 이후 아님
- VLM이 적은 관계식: 바꾸기 전에 맞던 식은 바꾼 뒤에도 맞아야 함 (전에도 안 맞던 식은 OCR 오류로 보고 무시, 기록만)
- 날짜: 바꾼 날짜와 문서의 다른 날짜의 앞뒤 순서 유지, 영수증번호 연월과 같던 달은 그대로
- 칸 상자: 그 칸에서 원래 값 글자만 정확히 잡힐 때만 (원래 숫자가 덜 지워지는 것 방지)"""
import argparse
import ast
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/generate
import genpath  # noqa: E402,F401  (하위 폴더 1_vlm·2_render·3_logic·4_compare 경로)
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))
OUT = ROOT / "outputs" / "고도화"
DATE = re.compile(r"^(20\d{2})([.\-/])(\d{2})\2(\d{2})$")
AMT = re.compile(r"^\d{1,3}(,\d{3})*$")


MAXE = 12   # VLM은 관계를 직접 확인하므로 자동 논리(8칸)보다 넉넉히 (10/30 납입확인서 9칸)


def _load_answer(p):
    t = p.read_text(encoding="utf-8").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])


def _norm(s):
    return str(s).replace(" ", "").replace(".", ",") if AMT.match(str(s).replace(" ", "").replace(".", ",")) else str(s).replace(" ", "")


def _date(s):
    m = DATE.match(str(s).strip())
    if not m:
        return None, None
    try:
        return dt.date(int(m.group(1)), int(m.group(3)), int(m.group(4))), m.group(2)
    except ValueError:
        return None, None


def _eval(expr, vals):
    """'12 = 5 + 6' 의 한쪽 식 계산. 숫자는 번호. floor100/floor10 허용."""
    tree = ast.parse(expr.replace("floor100", "F100").replace("floor10", "F10"), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            if n.value not in vals:
                raise KeyError(n.value)
            return vals[n.value]
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult):
            l, r = ev(n.left), ev(n.right)
            return l + r if isinstance(n.op, ast.Add) else l - r if isinstance(n.op, ast.Sub) else l * r
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("F100", "F10") and len(n.args) == 1:
            v = ev(n.args[0])
            u = 100 if n.func.id == "F100" else 10
            return v // u * u
        raise ValueError("허용 안 되는 식")
    return ev(tree)


def _rel_ok(rel, vals):
    if "=" not in rel:
        raise ValueError("등호 없음")
    l, r = rel.split("=", 1)
    return _eval(l.strip(), vals) == _eval(r.strip(), vals)


def check(ocr, ans, gray, ink_fn, today):
    """(통과 여부, 이유, 편집 목록[(원래, 새, 상자)], 메모)"""
    import logic_list as LL
    cells = {c["id"]: c for c in ocr["cells"]}
    ch = ans.get("변경") or []
    if not ch:
        return False, "VLM이 바꾸지 않음: " + str(ans.get("메모", ""))[:80], [], ""
    new_vals, edits, notes = {}, [], []
    for e in ch:
        try:
            k = int(e["번호"])
        except (KeyError, ValueError, TypeError):
            return False, f"번호 없음: {e}", [], ""
        c = cells.get(k)
        if c is None:
            return False, f"목록에 없는 번호 {k}", [], ""
        old, new = str(e.get("원래", "")).strip(), str(e.get("새", "")).strip()
        same_amt = AMT.match(_norm(old)) and c["amount"] is not None and int(_norm(old).replace(",", "")) == c["amount"]
        if not same_amt and AMT.match(_norm(old)):   # '⑤110,000'을 '0110,000'으로 읽은 경우: 앞 0 빼고 숫자 비교 (10/30)
            dig = "".join(ch for ch in c["text"] if ch.isdigit()).lstrip("0")
            same_amt = dig == _norm(old).replace(",", "").lstrip("0")
        if not same_amt and _norm(old) != _norm(c["text"]) and _norm(old) != _norm(c["raw"]):   # 금액은 숫자로 비교 ('⑤13,160'을 '013,160'으로 읽은 경우)
            return False, f"[{k}] 원래 값 다름: VLM {old} / OCR {c['text']}", [], ""
        if AMT.match(_norm(old)):
            if not AMT.match(new) or len(new.replace(",", "")) != len(_norm(old).replace(",", "")) or new == old:
                return False, f"[{k}] 새 금액 형식: {old} -> {new}", [], ""
            new_vals[k] = int(new.replace(",", ""))
        else:
            d0, s0 = _date(old)
            d1, s1 = _date(new)
            if d0 is None:
                return False, f"[{k}] 금액도 날짜도 아님: {old}", [], ""
            if d1 is None or s1 != s0 or d1 > today or d1 == d0:
                return False, f"[{k}] 새 날짜 형식·범위: {old} -> {new}", [], ""
            # 다른 날짜와 앞뒤 순서
            for o in ocr["cells"]:
                if o["id"] == k or o["id"] in {int(x["번호"]) for x in ch}:
                    continue
                for m in re.finditer(r"(20\d{2})[.\-/](\d{2})[.\-/](\d{2})", o["raw"]):
                    try:
                        od = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    except ValueError:
                        continue
                    if (od < d0) != (od < d1) or (od > d0) != (od > d1):
                        return False, f"[{k}] 날짜 순서 깨짐: {o['raw']} 기준 {old} -> {new}", [], ""
            for o in ocr["cells"]:
                m = re.match(r"^(20\d{2})(0[1-9]|1[0-2])-\d{2,}", o["raw"].replace(" ", ""))
                if m and (int(m.group(1)), int(m.group(2))) == (d0.year, d0.month) and (d1.year, d1.month) != (d0.year, d0.month):
                    return False, f"[{k}] 영수증번호 {o['raw']} 연월과 달라짐", [], ""
        edits.append((k, _norm(old) if AMT.match(_norm(old)) else old, new))
    before = {c["id"]: c["amount"] for c in ocr["cells"] if c["amount"] is not None}
    after = {**before, **new_vals}
    for rel in ans.get("관계") or []:
        try:
            ok0 = _rel_ok(rel, before)
        except Exception as ex:
            notes.append(f"식 못 읽음 '{rel}' ({type(ex).__name__})")
            continue
        if not ok0:
            notes.append(f"원래도 안 맞음 '{rel}'")
            continue
        if not _rel_ok(rel, after):
            return False, f"바꾼 뒤 식이 깨짐: {rel}", [], ""
    if len(edits) > MAXE:
        return False, f"바꿀 칸이 {len(edits)}개 (최대 {MAXE})", [], ""
    # 칸 상자는 식까지 통과한 뒤에 (시간이 걸려서)
    boxed = []
    for k, old, new in edits:
        b = LL._ok_box(gray, cells[k], old, ink_fn)
        if not b:
            return False, f"[{k}] 칸에서 값 글자를 정확히 못 잡음", [], ""
        boxed.append((old, new, b))
    return True, "통과", boxed, " / ".join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vlm")
    a = ap.parse_args()
    import forge_v2 as F
    import logic_list as LL
    LL.OCR = "rapid"
    d = OUT / "_vlm" / a.name
    today = dt.date.today()
    rows, report = [], []
    for oj in sorted(d.glob("*_ocr.json")):
        stem = oj.name[:-len("_ocr.json")]
        aj = d / f"{stem}_vlm.json"
        if not aj.exists():
            report.append({"사진": stem, "결과": "답 없음", "이유": "", "메모": ""})
            continue
        ocr = json.loads(oj.read_text(encoding="utf-8"))
        try:
            ans = _load_answer(aj)
        except Exception as ex:
            report.append({"사진": stem, "결과": "탈락", "이유": f"JSON 못 읽음 ({type(ex).__name__})", "메모": ""})
            continue
        gray = cv2.imread(ocr["original"], cv2.IMREAD_GRAYSCALE)
        ok, why, edits, memo = check(ocr, ans, gray, F.stroke_only, today)
        report.append({"사진": stem, "결과": "통과" if ok else "탈락", "이유": why, "메모": memo,
                       "문서": ans.get("문서", ""), "확신": ans.get("확신", ""), "칸수": len(edits)})
        if not ok:
            continue
        orig = Path(ocr["original"])
        try:
            orig_s = str(orig.relative_to(ROOT))
        except ValueError:
            orig_s = str(orig)
        r = {"id": f"vlm_{stem}", "original": orig_s, "kind": "날짜" if _date(edits[0][0])[0] else "금액",
             "규칙": f"VLM {ans.get('문서', '')} 확신:{ans.get('확신', '')}"}
        for k, (ov, nv, b) in enumerate(edits, 1):
            s = "" if k == 1 else str(k)
            r.update({f"original_value{s}": ov, f"forged_value{s}": nv, f"x{s}": b[0], f"y{s}": b[1], f"w{s}": b[2], f"h{s}": b[3]})
        rows.append(r)
    lst = OUT / f"원본목록_{a.name}.csv"
    with open(lst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LL.FIELDS, restval="")
        w.writeheader(); w.writerows(rows)
    with open(d / "확인결과.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["사진", "결과", "이유", "메모", "문서", "확신", "칸수"], restval="")
        w.writeheader(); w.writerows(report)
    n_ok = sum(1 for x in report if x["결과"] == "통과")
    print(f"통과 {n_ok} / 답 있음 {sum(1 for x in report if x['결과'] != '답 없음')} / 전체 {len(report)}")
    print(f"목록: {lst}\n다음: .venv/bin/python scripts/generate/forge.py --src {lst} --out {a.name}")


if __name__ == "__main__":
    main()
