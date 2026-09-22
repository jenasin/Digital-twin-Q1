#!/usr/bin/env bash
# Extra CPU workers for GRU/Transformer jobs, taking the job list from the end; seed-level lock
# files in dtq1.train keep them from duplicating work done by the GPU pool in run_all.sh.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS=ignore DTQ1_DEVICE=cpu DTQ1_THREADS=4
for k in 4 3 2 1 0; do for r in aug clean; do for a in gru transformer; do echo "$k $a $r"; done; done; done \
  | xargs -P 3 -L 1 bash -c 'python -m dtq1.train --fold $0 --arch $1 --regime $2 > logs/cpu_train_f$0_$1_$2.log 2>&1 && echo "cpu trained $0 $1 $2"'
