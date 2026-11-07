# generator — 위조 · 변조 문서 자동 생성기

탐지기를 재려면 정답이 붙은 가짜 서류가 많이 필요합니다. 두 가지 방식으로 만듭니다.

| | **위조** [`forgery/`](forgery/) | **변조** [`alteration/`](alteration/) |
|---|---|---|
| 뜻 | 존재하지 않는 서류를 **가상 정보로 새로** 만듦 | **실제 서류에서 한 부분만** 바꿈 |
| 입력 | 없음 (공식 서식 + 공식 기준 자료) | 실제 서류 사진 |
| 출력 | 진단서 · 진료비 계산서·영수증 PDF/PNG + 모든 값(data.json) | 변조본 + 바뀐 곳 정답 마스크 + 라벨 |
| 핵심 | 금액 · 진단 · 날짜 · 번호를 심평원 단가 · 상병마스터 · 국세청 코드로 맞춰 **정상 서류처럼 보이게** | 획을 따라 지우고 원본과 비슷한 글씨체로 다시 써서 **흔적이 안 보이게**, 합계 · 날짜 논리까지 맞춤 |

```
위조   공식 서식 PDF + 심평원 단가 · 상병마스터 ──→ 가상 환자의 진단서 · 영수증 (한 건씩 대량)
변조   실제 서류 사진 ──→ 칸 고르기 ──→ 지우기 ──→ 다시 쓰기 ──→ 블렌딩 ──→ 변조본 + 마스크 + 라벨
```

## 위조: `forgery/`

```bash
.venv/bin/pip install pyyaml pymupdf openpyxl
.venv/bin/python generator/forgery/main.py --n 10 --seed 1 --out outputs/위조      # 10건 생성
.venv/bin/python generator/forgery/tools/verify_cases.py outputs/위조               # 생성 결과 검산
```

자세한 규칙은 [`forgery/README.md`](forgery/README.md).

## 변조: `alteration/`

두 가지 다시 쓰기 방식과 조작 13종이 있습니다. 자세한 내용은 [`alteration/README.md`](alteration/README.md).
