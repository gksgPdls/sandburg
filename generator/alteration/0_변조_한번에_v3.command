#!/bin/bash
# 변조 한 번에 (v3): 사진 폴더 ->
#   앞 K장(기본 15): (1) 칸 목록 -> (2) VLM가 VLM 답 작성 -> (3) 계산 확인 -> 위조 이미지   [합계까지 맞춤]
#   나머지 사진:     (4) 자동 규칙으로 위조 이미지 (VLM 토큰 안 씀)                     [영수증 서식이면 합계까지, 못 알아보면 칸 하나]
# (2)는 각자 VLM 계정으로 한다. VLM 채팅에 문장을 붙여넣으면 이어서 진행.
cd "$(dirname "$0")/../.."   # 저장소 루트로
PY=.venv/bin/python
$PY -c "import rapidocr_onnxruntime" 2>/dev/null || .venv/bin/pip install rapidocr-onnxruntime

echo "사진 폴더를 이 창에 끌어다 놓고 Enter:"; read -r SRC; SRC="${SRC%\'}"; SRC="${SRC#\'}"; SRC="${SRC% }"
read -r -p "이름 (Enter면 vlm_$(date +%m%d_%H%M)): " NAME; NAME=${NAME:-vlm_$(date +%m%d_%H%M)}
read -r -p "VLM에게 맡길 장수 (Enter면 15, 나머지는 자동 규칙. 전부 VLM는 0): " N; N=${N:-15}

echo; echo "===== 1/4 칸 목록 만들기 (VLM에게 맡길 사진) ====="
$PY generator/alteration/forge/1_vlm/vlm_1_prep_칸목록.py --src "$SRC" --name "$NAME" --limit "$N" || exit 1
DIR="outputs/고도화/_vlm/$NAME"
TOTAL=$(ls "$DIR"/*_ocr.json 2>/dev/null | wc -l | tr -d ' ')
BATCH=15   # 한 대화(한 번 부르기)에 맡길 장수. 많이 맡기면 대화가 길어져 실수가 늘어남
REQ="$DIR 폴더의 VLM 답을 써줘. docs/VLM_답쓰기.md 지침대로. 아직 _vlm.json 이 없는 사진부터 최대 ${BATCH}장만 하고, 사진마다 <사진>_vlm.json 파일을 그 폴더에 저장해."
missing() { for f in "$DIR"/*_ocr.json; do s=$(basename "$f" _ocr.json); [ -f "$DIR/${s}_vlm.json" ] || echo "$s"; done; }
count_done() { ls "$DIR"/*_vlm.json 2>/dev/null | wc -l | tr -d ' '; }

echo; echo "===== 2/4 VLM가 답 쓰기 ($TOTAL장, ${BATCH}장씩) ====="
DONE=$(count_done)
if [ "$DONE" -ge "$TOTAL" ]; then
  echo "답 파일이 이미 다 있음 -> 건너뜀 (같은 이름으로 다시 실행한 경우)"
elif command -v vlm >/dev/null 2>&1; then
  echo "VLM Code로 자동 진행 (내 VLM 계정 사용). 묶음마다 새로 부름"
  while :; do
    LIST=$(missing | head -n "$BATCH" | tr '\n' ' ')
    [ -z "$LIST" ] && break
    BEFORE=$(count_done)
    echo "  이번 묶음: $LIST"
    vlm -p "$DIR 폴더의 VLM 답을 써줘. docs/VLM_답쓰기.md 지침대로. 이 사진들만: $LIST. 사진마다 <사진>_vlm.json 파일을 그 폴더에 저장해." \
      --allowedTools "Read,Write,Glob,Grep" || { echo "VLM Code 실행 실패 -> 아래 수동 방법으로"; break; }
    [ "$(count_done)" -le "$BEFORE" ] && { echo "답이 늘지 않음 (사용량 한도 등) -> 아래 수동 방법으로"; break; }
  done
fi
DONE=$(count_done)
if [ "$DONE" -lt "$TOTAL" ]; then
  printf "%s" "$REQ" | pbcopy
    echo "VLM 앱이 열렸습니다. repo 폴더를 연결한 대화에 붙여넣기(⌘V) 하고 보내세요. (문장은 복사돼 있음)"
  echo "한 대화에 ${BATCH}장씩 합니다. 한 묶음이 끝나면 '새 대화'를 열고 같은 문장을 다시 붙여넣으세요."
  echo "답 파일이 다 생기면 자동으로 다음 단계로 넘어갑니다. 기다리지 않고 넘어가려면 Enter."
  while :; do
    DONE=$(count_done)
    printf "\r  답 파일 %s / %s " "$DONE" "$TOTAL"
    [ "$DONE" -ge "$TOTAL" ] && break
    read -r -t 5 && break
  done
  echo
fi

echo; echo "===== 3/4 계산 확인 + 위조 이미지 만들기 (VLM 답) ====="
$PY generator/alteration/forge/1_vlm/vlm_2_apply_답확인.py --name "$NAME" || exit 1
ROWS=$(($(wc -l < "outputs/고도화/원본목록_$NAME.csv" 2>/dev/null || echo 1) - 1))
if [ "$ROWS" -gt 0 ]; then
  $PY generator/alteration/forge/forge.py --src "outputs/고도화/원본목록_$NAME.csv" --out "VLM_$NAME"
  open "outputs/고도화/VLM_$NAME" 2>/dev/null || true
else
  echo "통과한 사진이 없습니다. $DIR/확인결과.csv 의 이유를 확인하세요."
fi
echo "확인 결과: $DIR/확인결과.csv"

# 4. 나머지 사진: VLM에게 안 맡긴 사진은 자동 규칙으로 (토큰 안 씀)
REST="$DIR/_나머지_사진"
mkdir -p "$REST"
for f in "$SRC"/*; do
  case "$f" in *.jpg|*.JPG|*.jpeg|*.JPEG|*.png|*.PNG) ;; *) continue ;; esac   # (맥 기본 bash 3.2 호환)
  s=$(basename "${f%.*}")
  [ -f "$DIR/${s}_ocr.json" ] || cp -n "$f" "$REST/"
done
NREST=$(ls "$REST" 2>/dev/null | wc -l | tr -d ' ')
if [ "$NREST" -gt 0 ]; then
  echo; echo "===== 4/4 나머지 ${NREST}장: 자동 규칙으로 위조 이미지 (VLM 안 씀) ====="
  $PY generator/alteration/forge/forge.py --src "$REST" --out "자동_$NAME" --logic 시도 --limit 0
  open "outputs/고도화/자동_$NAME" 2>/dev/null || true
fi

echo; echo "결과: outputs/고도화/VLM_$NAME/ (VLM, 합계까지) + outputs/고도화/자동_$NAME/ (나머지, 자동 규칙)"
read -r -p "끝. 엔터로 닫기"
