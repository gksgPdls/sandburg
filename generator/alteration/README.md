# alteration — 변조 문서 자동 생성기

실제 진료비 영수증 · 진단서 사진에서 **한 부분만** 바꾸고, 바뀐 곳의 정답 마스크와 라벨을 함께 남깁니다.

## 다시 쓰기 방식

| 방식 | 지우기 | 다시 쓰기 | 코드 |
|---|---|---|---|
| **칸 덮고 다시 쓰기** (v1) | 칸을 배경색 상자로 덮음 | 글꼴로 새 값을 쓰고, 경계만 로컬 Stable Diffusion 으로 섞음 | [`forge/2_render/ai_inpainting.py`](forge/2_render/ai_inpainting.py) |
| **획 따라 지우고 비슷한 글씨체로 다시 쓰기** (v2) | 칸 전체가 아니라 **글자 획 자리만** 주변 종이 결로 메움. 표 선은 남김 | 원래 값을 후보 글꼴로 그려 보고 **가장 닮은 글꼴 · 굵기 · 자간 · 기울기 · 흐림**을 골라 새 값을 씀. 문서 안 같은 글자를 잘라 다시 쓰는 방식(재사용)도 있음 | [`forge/2_render/forge_v2.py`](forge/2_render/forge_v2.py), [`font_match.py`](forge/2_render/font_match.py), [`glyph_library.py`](forge/2_render/glyph_library.py) |

v2 는 단계별로 켜고 끌 수 있습니다: 획 지우기 → 글꼴 맞추기 → 자리 · 기울기 · 원근 → 종이 질감 · 흐림 → SD 테두리 블렌딩 (`forge_config.json`).

## 바꿀 값 고르기 (서식 논리 맞춤)

아무 값이나 바꾸면 합계가 안 맞아 금방 들킵니다. 그래서 바꿀 칸과 새 값을 서식 논리에 맞춰 고릅니다.

- `forge/1_vlm/` — VLM 에게 칸 목록을 주고 "합계 · 날짜 순서 · KCD 코드 체계가 맞는" 새 값을 받음 (프롬프트는 `vlm_prompt.txt`)
- `forge/3_logic/` — 규칙으로 고르기, 바꾼 칸이 들어가는 합계 칸도 같이 고치기 (합계까지 맞춘 변조)

## 조작 방법별 샘플 13종: `methods/`

`methods/scripts/edit_ops.py` 의 획 지우기 · 다시 쓰기 함수를 써서 방법별로 만듭니다.

| 축 | 조작 |
|---|---|
| 픽셀 복사 | 복사-붙여넣기, 날짜 복사, 자릿수 조합 |
| 내용 논리 | 합계 깨기, 자릿수 키우기, 날짜 충돌(퇴원 < 입원), KCD 코드 교체 |
| 글자 다시 쓰기 | 날짜 이동, 병명 추가, 글자 추가, 합계까지 맞추기 |
| 기타 | EDMS 변형(재압축 · 회전 · 축소), 정상 대조군 |

방법마다 `methods/samples/<방법>/README.md` 에 수법 · 탐지 축 · 실행 방법이 있고, 라벨 형식은 [`methods/FORMAT.md`](methods/FORMAT.md) 입니다.

## 실행

```bash
# v2: 사진 폴더 → 칸 자동 선택 → 변조 (설정 forge/forge_config.json)
.venv/bin/python generator/alteration/forge/forge.py --src <사진 폴더> --out 실험1 --logic 시도

# VLM 으로 값 고르기 → 변조 (1 → 2)
.venv/bin/python generator/alteration/forge/1_vlm/vlm_1_prep_칸목록.py --src <사진 폴더> --name 실험1 --limit 10
.venv/bin/python generator/alteration/forge/1_vlm/vlm_2_apply_답확인.py --name 실험1

# 조작 방법별 (예: EDMS 변형)
cd generator/alteration/methods/scripts && ../../../../.venv/bin/python make_edms.py --per-axis 5 --normal 5 --seed 0
```

맥에서는 `run/` 의 번호 순 실행 파일이나 `0_변조_한번에_v3.command` 를 더블클릭해도 됩니다. 결과는 저장소 맨 위 `outputs/` 에 쌓입니다 (git 제외).
글꼴 맞추기에 쓰는 윈도우 글꼴은 [`fonts/README.md`](fonts/README.md) 를 보고 직접 넣습니다.
