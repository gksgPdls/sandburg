# generator/forgery/ — 가상 진단서·진료비 계산서 생성기

가상 환자 1명의 진단서(별지 제5호의2)와 진료비 계산서·영수증(별지 제6호)을 대량 생성합니다.
모든 단가는 심평원 오픈 API 공식 자료(config/prices.yaml)에서 가져옵니다.
생성 로직 설명: `가상의료문서_생성로직_보고서.docx`

## 실행
```bash
python3 -m venv .venv && .venv/bin/pip install pyyaml pymupdf openpyxl
.venv/bin/python main.py --n 10 --seed 1        # → out_rebuild/
```

## 단가표 새로 받기 (선택)
공공데이터포털 인증키를 `config/hira_api_key.txt` 에 한 줄로 저장한 뒤 (git 에 올라가지 않음):
```bash
.venv/bin/python tools/fetch_prices.py --fresh
```
