# generator/forgery/ — 가상 진단서·진료비 계산서 생성기

가상 환자 1명의 진단서(의료법 시행규칙 별지 제5호의2)와 진료비 계산서·영수증(요양급여규칙 별지 제6호, 외래·입원)을 대량 생성합니다.
생성 로직 설명: `가상의료문서_생성로직_보고서_최종.docx`

원칙: 공식 자료로 확인한 값만 생성하고, 근거가 애매하거나 공식 자료가 없는 항목은 만들지 않습니다.
- 금액: 심평원 공식 급여 단가 (2026 오픈 API `config/prices.yaml`, 2024·2025 그 해 책자·약제목록 `config/prices_<연도>.yaml`)와 보건복지부 고시 제증명수수료 상한.
  병원·종합병원 행위료에는 종별가산율(5%·10%)을, 같은 날 심층열치료와 함께 한 표층열치료에는 50%(MM015)를 적용합니다.
  원내 투약이 있으면 의약품관리료를, 입원은 기관 간호등급별 입원료 코드와 식대(기본식 + 가산)를 산정합니다.
- 진단: 7개 진료과 50개 진단, 모두 심평원 상병마스터 완전코드이고 병명·영문명은 상병마스터 표기 그대로 (실행 시 `assets/references/hira_kcd_master_2025_9.csv` 와 대조).
- 진료: 외래(초진·재진)와 병원·종합병원 입원, 2024-01-01 ~ 2026-09-30.
  외래는 평일 낮·야간, 토요일, 일요일·공휴일 진료를 만들고 책자 규정대로 진찰료 야간·공휴·토요 가산과 의원 처치 가산을 적용합니다.
- 기관: 한 세무서가 구 전체를 관할하는 38개 시·구. 사업자등록번호 앞 3자리 = 관할 세무서 코드번호 (국세청).
- 만들지 않는 것: 비급여, 도수치료 관리급여, 상해(S·T) 진단·상해진단서, 응급진료, 발병일.

## 파일 구성
| 파일 | 역할 |
|---|---|
| `main.py` | 실행 시작점. 시작 점검 → 사건 N건 생성 → 진단서·계산서 PDF/PNG, data.json 저장 |
| `generator/generate.py` | 사건 1건의 모든 값(기관·진단·날짜·시각·처방·금액·기재값)을 만드는 핵심 로직 |
| `generator/pricing.py` | 진료일 연도의 공식 단가표에서 단가 조회, 종별가산, 시작 점검 |
| `generator/calc.py` | 계산서 금액(본인/공단 분할, 외래 100원 미만 절사, ①~⑪) 계산·검산 |
| `generator/kcd.py` | 진단이 상병마스터 완전코드·공식 병명인지 점검 |
| `generator/ids.py` | 주민번호 검증, 사업자등록번호·전화·면허번호 생성 |
| `generator/render.py`, `render_rebuild.py` | 공식 서식을 다시 그리고 값을 같은 글씨체로 기입 |
| `config/codebook.yaml` | 생성 규칙의 모든 확률·목록 (진료과·진단·처방·시각·기관·지역) |
| `config/price_sources.yaml` | 항목별 공식 단가 출처 (수가코드·주성분코드·고시) |
| `config/prices.yaml`, `prices_2024.yaml`, `prices_2025.yaml` | 연도별 공식 단가표 (도구가 만듦, 직접 고치지 않음) |
| `config/variants.yaml`, `fonts.yaml`, `patient.yaml` | 표기 방식, 글씨체, 고정 환자정보 |
| `tools/fetch_prices.py` | 2026 단가표 만들기 (심평원 API) |
| `tools/build_book_prices.py`, `build_list_prices.py` | 2024·2025 단가표 만들기 (책자 PDF, 약제목록 엑셀) |
| `tools/verify_cases.py` | 생성 결과 검증 |
| `assets/forms/` | 법제처 공식 서식 PDF |
| `assets/references/` | 근거 자료 (API 원본 응답, 상병마스터, 단가 근거 엑셀) |

## 실행
```bash
python3 -m venv .venv && .venv/bin/pip install pyyaml pymupdf openpyxl
.venv/bin/python main.py --n 10 --seed 1        # → out/
```
실행 시작 때 진단 코드(상병마스터)와 단가표(진료기간의 모든 연도)를 점검하고, 하나라도 어긋나면 생성하지 않고 멈춥니다.

## 생성 결과 검증
```bash
.venv/bin/python tools/verify_cases.py out
```
data.json 을 공식 단가표·규칙으로 다시 계산해 대조합니다 (금액 산식, 진료일 연도 단가, 종별가산, 표층열 50%, 의약품관리료, 본인부담률, 입원료·식대, 날짜·공휴일, 상병코드, 사업자번호).

## 단가표 새로 받기 (선택)
공공데이터포털 인증키를 `config/hira_api_key.txt` 에 한 줄로 저장한 뒤 (git 에 올라가지 않음):
```bash
.venv/bin/python tools/fetch_prices.py --fresh
```

## 과거 연도 단가표 다시 만들기 (선택)
API 는 현재 단가만 주므로 과거 연도는 심평원 파일로 만듭니다 (파일 출처는 각 도구 상단 설명).
```bash
.venv/bin/python tools/build_book_prices.py 2025 <2025 요양급여비용 책자.pdf>      # 수가 (약제 항목은 기존 값 유지)
.venv/bin/python tools/build_list_prices.py 2025 <약제목록.xlsx>                    # 약제
```
