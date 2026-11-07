# 데이터셋 정리 규칙

각자 편한 방식으로 만들되, 나중에 합칠 수 있도록 **이것만 맞춘다.**

## 1. 폴더에 `labels.csv`를 둔다

`generator/alteration/methods/samples/` 아래에 폴더를 만들고(이름 자유, 조작 방법별 권장) 그 안에 이미지와 `labels.csv`를 둔다.
하위 폴더를 써도 되고, 파일 이름도 자유다.

```
generator/alteration/methods/samples/
  copymove/
    labels.csv
    README.md          (권장) 조작 방식, 사용한 스크립트
    CM_001_original.jpg ...
  팀원_자동생성/
    labels.csv
    imgs/doc_a.jpg, imgs/doc_a_fake.jpg, imgs/doc_a_gt.png ...
```

## 2. CSV 열

템플릿: [`labels_template.csv`](labels_template.csv). 저장은 **UTF-8**로 한다(엑셀이면 "CSV UTF-8").

| 열 | | 내용 |
|---|---|---|
| `id` | **필수** | 샘플 ID. 전체에서 겹치지만 않으면 된다. 권장: 축 접두어 + 번호 (`CM_001`) |
| `target_axis` | **필수** | 주로 시험하는 탐지 축. `copy_move` `glyph` `content_logic` `metadata` `c2pa` `pixel` `none`(정상) |
| `original` | **필수** | 원본 파일 경로 (CSV 위치 기준) |
| `description` | **필수** | 무엇을 어떻게 바꿨는지 |
| `tampered` | | 조작본 경로 |
| `mask` | | 정답 마스크 경로 |
| `detection` | | 어떻게 잡히는지 / 어떤 축을 피하는지 |
| `method` `doc_type` `difficulty` | | 조작 방법, 문서 종류, 난이도 |
| `source` | | 원본 출처. 짧은 ID로 쓴다(예: `DOC210_0042`). 긴 원본 경로는 로컬 `_work/pool/manifest.csv`에 둔다 |
| `external_ai` | | 원본을 외부 AI에 넣었으면 `Y` |
| `author` `created` `notes` | | 만든 사람, 날짜, 메모 |

**다른 열은 얼마든지 추가해도 된다**(예: `pipeline_ver`, `seed`, `command`). 합칠 때 그대로 따라간다.

## 3. 마스크

평가에 쓰는 정답이라 이것만은 지킨다.

- 조작 영역 흰색(255), 나머지 검은색(0)
- **PNG로 저장** (JPG는 경계에 회색이 생긴다)
- 원본과 해상도가 같아야 한다
- 메타데이터처럼 영역이 없는 조작은 비워 둔다

## 4. 지켜야 할 것

- **이미지는 git에 올라가지 않는다**(`.gitignore`가 막는다). 이미지는 둔다
- **외부 AI에 원본을 넣는 건 둘이 합쳐 10~20건까지.** 넣었으면 `external_ai`에 `Y`
- 흰 박스로 덮는 조작은 쓰지 않는다(비식별화 흰 박스와 구분이 안 된다)
- 전자문서(디지털 원본 PDF)는 copy-move 원본으로 쓰지 않는다

## 5. 도구 (`dataset/`에서 `./.venv/bin/python scripts/...`)

| 명령 | 하는 일 |
|---|---|
| `verify_sample.py` | CSV와 파일이 맞는지, 마스크 규격, ID 중복, 외부 AI 건수 검사 |
| `build_catalog.py` → `open catalog.html` | 브라우저로 샘플 목록 보기 (로컬 전용, 인터넷 연결 없음) |
| `merge_labels.py` | 모든 CSV를 `labels_all.csv` 하나로 합치기  |
| `make_preview.py` | 조작 위치에 빨간 사각형을 그린 확인용 이미지 만들기 |
| `register_sample.py` | (선택) 파일 복사, ID 자동 부여, CSV 추가를 한 번에 |
| `import_labels.py` | 열 이름이 다른 CSV(팀원 결과 등)를 우리 형식으로 가져오기. 엑셀 한글 CSV(cp949)도 읽는다 |

자동화 코드에서 `register_sample`을 함수로 쓸 수도 있다(선택).

```python
import sys; sys.path.insert(0, "generator/alteration/methods/scripts")
from register_sample import register
register(method="copymove", axis="copy_move", original=src, tampered=out,
         mask=mask_path, description="...", detection="...", author="김동현")
```

## 6. 팀원 결과 합치기

형식이 같으면 폴더째 `samples/` 아래에 넣고 `merge_labels.py`만 돌리면 된다.

열 이름이 다르면 `import_labels.py`로 짝지어 가져온다. ID가 겹치면 `--id-prefix`로 앞에 문자를 붙인다.

```bash
cd generator/alteration/methods/scripts
../.venv/bin/python import_labels.py ~/팀원결과/결과.csv --dest teammate_ai \
    --map id=번호 original=원본 tampered=조작본 mask=정답 description=설명 \
    --set target_axis=c2pa author=팀원 --id-prefix T_
../.venv/bin/python verify_sample.py && ../.venv/bin/python merge_labels.py
```

- `--map 우리열=그쪽열`: 그쪽 열 값을 가져온다
- `--set 우리열=값`: 모든 행에 같은 값을 넣는다 (그쪽 CSV에 없는 정보)
- 짝짓지 않은 그쪽 열은 그대로 따라온다
- 이미지는 `samples/{dest}/`로 복사된다
