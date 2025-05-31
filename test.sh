#!/usr/bin/env bash
# ------------------------------------------------------------------
# run_jobs.sh  –  顺序执行一批 Python 命令，最大并行 2 个，
#                 每条命令启动时间至少间隔 2 秒
# ------------------------------------------------------------------

set -euo pipefail

###############################################################################
# 1) 把需要执行的命令写进下面的数组（按顺序）
###############################################################################
COMMANDS=(
  "python main.py --bce_ratio=1"
  "python main.py --bce_ratio=0.5"
  "python main.py --bce_ratio=0.2"
  "python main.py --bce_ratio=0.1"
  # 在此继续添加 ...
)
###############################################################################

MAX_JOBS=2          # 并行上限
MIN_GAP=2           # 每条命令启动间隔（秒）

# 等待直到后台运行的任务数 < MAX_JOBS
wait_for_slot() {
  while (( $(jobs -r -p | wc -l) >= MAX_JOBS )); do
    sleep 1
  done
}

last_start=0        # 记录上一条命令启动时间

for cmd in "${COMMANDS[@]}"; do
  wait_for_slot

  # 保证间隔 >= MIN_GAP
  now=$(date +%s)
  elapsed=$(( now - last_start ))
  if (( elapsed < MIN_GAP )); then
    sleep $(( MIN_GAP - elapsed ))
  fi

  echo "[`date '+%H:%M:%S'`] RUN  ->  $cmd"
  eval "$cmd" &          # 后台运行
  last_start=$(date +%s)
done

# 等待全部后台任务结束
wait
echo "All jobs finished."
