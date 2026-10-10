#!/usr/bin/env bash
set -Eeuo pipefail

if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
    printf 'This script requires Bash 5.1 or newer (for wait -n -p).\n' >&2
    exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
PPO_STEPS="${PPO_STEPS:-3000000}"
SAC_STEPS="${SAC_STEPS:-1000000}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-1}"
SEEDS=(1 2 3 4 5)
LOG_DIR="$ROOT_DIR/batch_logs/$(date +%Y%m%d-%H%M%S)"
PYTHON_PATH="$ROOT_DIR/cleanrl:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! "$PPO_STEPS" =~ ^[0-9]+$ || ! "$SAC_STEPS" =~ ^[0-9]+$ ]]; then
    printf 'PPO_STEPS and SAC_STEPS must be positive integers.\n' >&2
    exit 2
fi

if [[ "$CAPTURE_VIDEO" != "0" && "$CAPTURE_VIDEO" != "1" ]]; then
    printf 'CAPTURE_VIDEO must be 0 or 1.\n' >&2
    exit 2
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON" >&2
    exit 2
fi

if ! PYTHONPATH="$PYTHON_PATH" "$PYTHON" -c \
    'import gymnasium, matt_ant_env, cleanrl_utils.buffers; env = gymnasium.make("AntBackflip-v0"); env.close()' \
    >/dev/null 2>&1; then
    printf 'Environment preflight failed. Activate the CleanRL environment and run from this repository.\n' >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
printf 'Run logs: %s\n' "$LOG_DIR"
printf 'Seeds: %s\n' "${SEEDS[*]}"
printf 'Steps: PPO=%s SAC=%s\n' "$PPO_STEPS" "$SAC_STEPS"

declare -a running_pids=()
declare -A algorithm_by_pid=()
declare -A seed_by_pid=()
declare -A log_by_pid=()
next_ppo=0
next_sac=0
failed=0

stop_children() {
    trap - INT TERM
    printf '\nInterrupted; stopping active training processes.\n' >&2
    for pid in "${running_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${running_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_children INT TERM

launch_run() {
    local algorithm="$1"
    local seed="$2"
    local steps
    local script
    local log_path="$LOG_DIR/${algorithm}_seed${seed}.log"
    local -a args

    if [[ "$algorithm" == "ppo" ]]; then
        steps="$PPO_STEPS"
        script="cleanrl/ppo_continuous_action.py"
    else
        steps="$SAC_STEPS"
        script="cleanrl/sac_continuous_action.py"
    fi

    args=(--env-id AntBackflip-v0 --seed "$seed" --total-timesteps "$steps" --cuda --track --save-model)
    if [[ "$CAPTURE_VIDEO" == "1" ]]; then
        args+=(--capture-video)
    fi

    {
        printf 'Command: PYTHONPATH=%q %q %q' "$PYTHON_PATH" "$PYTHON" "$script"
        printf ' %q' "${args[@]}"
        printf '\n'
    } >"$log_path"

    (
        cd "$ROOT_DIR"
        PYTHONPATH="$PYTHON_PATH" "$PYTHON" "$script" "${args[@]}"
    ) >>"$log_path" 2>&1 &

    local pid=$!
    running_pids+=("$pid")
    algorithm_by_pid["$pid"]="$algorithm"
    seed_by_pid["$pid"]="$seed"
    log_by_pid["$pid"]="$log_path"
    printf 'Started %s seed %s (PID %s); log: %s\n' "$algorithm" "$seed" "$pid" "$log_path"
}

launch_run ppo "${SEEDS[$next_ppo]}"
next_ppo=$((next_ppo + 1))
launch_run sac "${SEEDS[$next_sac]}"
next_sac=$((next_sac + 1))

while ((${#running_pids[@]} > 0)); do
    finished_pid=""
    if wait -n -p finished_pid "${running_pids[@]}"; then
        status=0
    else
        status=$?
    fi

    if [[ -z "$finished_pid" ]]; then
        printf 'Could not determine which training process finished.\n' >&2
        failed=1
        break
    fi

    algorithm="${algorithm_by_pid[$finished_pid]}"
    seed="${seed_by_pid[$finished_pid]}"
    log_path="${log_by_pid[$finished_pid]}"
    remaining_pids=()
    for pid in "${running_pids[@]}"; do
        if [[ "$pid" != "$finished_pid" ]]; then
            remaining_pids+=("$pid")
        fi
    done
    running_pids=("${remaining_pids[@]}")
    unset 'algorithm_by_pid[$finished_pid]' 'seed_by_pid[$finished_pid]' 'log_by_pid[$finished_pid]'

    if ((status == 0)); then
        printf 'Finished %s seed %s successfully.\n' "$algorithm" "$seed"
    else
        printf 'FAILED: %s seed %s (exit %s); see %s\n' "$algorithm" "$seed" "$status" "$log_path" >&2
        failed=1
    fi

    if ((failed == 0)); then
        if [[ "$algorithm" == "ppo" ]] && ((next_ppo < ${#SEEDS[@]})); then
            launch_run ppo "${SEEDS[$next_ppo]}"
            next_ppo=$((next_ppo + 1))
        elif [[ "$algorithm" == "sac" ]] && ((next_sac < ${#SEEDS[@]})); then
            launch_run sac "${SEEDS[$next_sac]}"
            next_sac=$((next_sac + 1))
        fi
    fi
done

trap - INT TERM
if ((failed != 0)); then
    printf 'Batch stopped after a failed run. Logs are in %s\n' "$LOG_DIR" >&2
    exit 1
fi

printf 'All paired PPO/SAC seed runs finished. Logs are in %s\n' "$LOG_DIR"
