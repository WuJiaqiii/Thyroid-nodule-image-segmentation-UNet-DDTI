#!/usr/bin/env bash

# 运行所有子目录下的 config1.yaml~config10.yaml
# 最多并发运行 MAX_JOBS 条命令，且各命令启动时间间隔至少 1 秒

set -euo pipefail

CONFIG_DIR="config"
MAX_JOBS=3

# 等待并发任务数低于 MAX_JOBS
wait_for_slot() {
  while (( $(jobs -r -p | wc -l) >= MAX_JOBS )); do
    sleep 1
  done
}

for model_dir in "$CONFIG_DIR"/*; do
  if [[ -d "$model_dir" ]]; then
    echo "Processing directory: $model_dir"
    for cfg in "$model_dir"/config*.yaml; do
      if [[ -f "$cfg" ]]; then
        # 等待可用的插槽
        wait_for_slot
        echo "Running: python main.py --config_path '$cfg'"
        # 启动任务并在后台运行
        python main.py --config_path "$cfg" &
        # 确保下一任务启动时至少间隔 1 秒
        sleep 1
      fi
    done
  fi
 done

# 等待所有后台任务完成
wait