# ai_inpainting (허소영)

- 1,005쌍. 금액/날짜 칸 하나를 바꿈 (scripts/generate/ai_inpainting.py, auto_regions.py)
- 방식(v1): 칸을 배경색 박스로 덮고 폰트로 새 값을 쓴 뒤, 경계 테두리만 로컬 Stable Diffusion으로 섞음
- FORMAT.md의 '흰 박스로 덮는 조작은 쓰지 않는다'에 해당하는 구버전. 고도화 버전(글자 획만 지우기)은 별도 폴더로 추가 예정
- 이미지는 이 폴더에 복사하지 않고 기존 위치를 상대 경로로 가리킴 (dataset/original/bankbook, dataset/tampered/ai_inpainting, dataset/labels/masks)
- labels.csv는 scripts/export/to_samples_format.py로 생성
