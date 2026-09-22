#!/usr/bin/env bash
# Full pipeline: preprocessing -> training (5 folds x 4 architectures x 2 regimes x 5 seeds)
# -> degradation grid evaluation -> zero-shot TSFM -> tables -> figures.
# LightGBM and torch always run in separate processes (their OpenMP runtimes clash on macOS).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS=ignore
mkdir -p logs

[ -f data/processed/streams/C10.npz ] || python -m dtq1.preprocess | tee logs/preprocess.log

train() {  # fold arch regime
  python -m dtq1.train --fold "$1" --arch "$2" --regime "$3" > "logs/train_f$1_$2_$3.log" 2>&1 && echo "trained $*"
}
evaluate() {  # fold tag archs...
  local k=$1 tag=$2; shift 2
  python -m dtq1.evaluate --fold "$k" --tag "$tag" --archs "$@" > "logs/eval_f${k}${tag}.log" 2>&1 && echo "evaluated $k $tag"
}
export -f train evaluate

# gradient boosting on CPU (2 parallel), neural models on the GPU (3 parallel)
for k in 0 1 2 3 4; do for r in clean aug; do echo "$k lgbm $r"; done; done \
  | xargs -P 2 -L 1 bash -c 'train "$@"' _ &
for k in 0 1 2 3 4; do for r in clean aug; do for a in gru transformer ssm; do echo "$k $a $r"; done; done; done \
  | xargs -P 3 -L 1 bash -c 'train "$@"' _
wait

for k in 0 1 2 3 4; do echo "$k _lgbm lgbm"; echo "$k _nn gru transformer ssm"; done \
  | xargs -P 3 -L 1 bash -c 'evaluate "$@"' _

[ -f results/raw/tsfm.pkl ] || python -m dtq1.tsfm > logs/tsfm.log 2>&1
python -m dtq1.analysis
python -m dtq1.figures
