#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <window_name> <topic> <command>"
  echo "Example: $0 env_setup bootstrap_env \"bash scripts/bootstrap_env.sh\""
  exit 1
fi

WINDOW_NAME="$1"
TOPIC="$2"
COMMAND="$3"

ROOT="/home/howard/Splat-Nav"
TMUX_BIN="/data/howard/miniconda3/bin/tmux"
TMUX_LD_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
SESSION="splatnav_jobs"

if [[ ! -x "$TMUX_BIN" ]]; then
  echo "tmux not found at $TMUX_BIN"
  exit 1
fi

tmux_cmd() {
  LD_LIBRARY_PATH="$TMUX_LD_PATH" "$TMUX_BIN" "$@"
}

RUN_DIR="$("$ROOT/scripts/create_run_dir.sh" "$TOPIC")"

{
  echo "# command"
  echo "$COMMAND"
} > "$RUN_DIR/cmd.txt"

JOB_SCRIPT="$RUN_DIR/job.sh"
cat > "$JOB_SCRIPT" << EOF
#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/data/howard/splatnav/tmp
export PIP_CACHE_DIR=/data/howard/splatnav/cache/pip
# Keep Python wheels self-contained; avoid forcing system CUDA libs into runtime resolution.
if [[ "\${KEEP_LD_LIBRARY_PATH:-0}" != "1" ]]; then
  unset LD_LIBRARY_PATH
fi
mkdir -p "\$TMPDIR" "\$PIP_CACHE_DIR"
cd "$ROOT"
$COMMAND
EOF
chmod +x "$JOB_SCRIPT"

if ! tmux_cmd has-session -t "$SESSION" 2>/dev/null; then
  tmux_cmd new-session -d -s "$SESSION" -c "$ROOT"
fi

tmux_cmd new-window -t "$SESSION" -n "$WINDOW_NAME" -c "$ROOT" \
  "bash '$JOB_SCRIPT' |& tee '$RUN_DIR/stdout.log'"

echo "started"
echo "run_dir=$RUN_DIR"
echo "session=$SESSION window=$WINDOW_NAME"
echo "attach: $TMUX_BIN attach -t $SESSION"
