#!/usr/bin/env bash
# One Colab benchmark configuration end to end (ADR-001), unattended:
#   start vLLM with the scenario's model and settings -> wait until healthy ->
#   start Relay standalone -> probe + open-loop run (--auto-rates) -> stop both ->
#   zip results (and copy to Google Drive if it is mounted).
#
#   scripts/colab_run.sh colab-t4-1.5b-fp16
#
# Everything about the configuration comes from bench/scenarios/<name>.yaml, so the
# notebook never edits code. Needs: `uv sync` done, `vllm` on PATH, an NVIDIA GPU.
set -euo pipefail

NAME="${1:?usage: scripts/colab_run.sh <scenario name> [--quality]}"
# --quality: GSM8K accuracy (bench/quality/gsm8k.py) instead of the latency benchmark.
MODE="${2:---bench}"
SCENARIO="bench/scenarios/${NAME}.yaml"
PY=".venv/bin/python"
VLLM_PORT=8001
RELAY_PORT=8000
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOGS="results/dev/colab-logs-${NAME}-${STAMP}"
mkdir -p "$LOGS"

# Read the configuration from the scenario (one source of truth).
read -r REPO REVISION PUBLIC_NAME SERVER_ARGS < <("$PY" - "$SCENARIO" <<'EOF'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))
repo, _, rev = s["setup"]["model_file"].partition("@")
print(repo, rev or "main", s["target"]["model"], s["setup"]["server_args"])
EOF
)

# One random key for this session: Relay accepts only it, the harness sends it.
KEY="rk_colab_$("$PY" -c 'import secrets; print(secrets.token_hex(16))')"
export RELAY_STANDALONE=1 RELAY_STANDALONE_API_KEY="$KEY" RELAY_BENCH_API_KEY="$KEY"
export RELAY_CONFIG=config/bench/colab-vllm.yaml RELAY_PORT="$RELAY_PORT"

VLLM_PID="" RELAY_PID=""
cleanup() {
  # Always free the GPU and the ports, even after a failure, so the next config can run.
  [ -n "$RELAY_PID" ] && kill "$RELAY_PID" 2>/dev/null || true
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    [ "${used:-0}" -lt 1000 ] && break
    sleep 2
  done
}
trap cleanup EXIT

wait_for() {  # url, seconds, pid, what
  for _ in $(seq 1 "$2"); do
    curl -sf "$1" >/dev/null && return 0
    kill -0 "$3" 2>/dev/null || { echo "$4 exited during start-up; log: $LOGS"; tail -30 "$LOGS"/*.log; exit 1; }
    sleep 1
  done
  echo "$4 not healthy after $2 s; log: $LOGS"; tail -30 "$LOGS"/*.log; exit 1
}

echo "== $NAME: vLLM serving $REPO@$REVISION as $PUBLIC_NAME ($SERVER_ARGS)"
# shellcheck disable=SC2086  # SERVER_ARGS is a list of flags
vllm serve "$REPO" --revision "$REVISION" --tokenizer-revision "$REVISION" \
  --served-model-name "$PUBLIC_NAME" --port "$VLLM_PORT" $SERVER_ARGS \
  > "$LOGS/vllm.log" 2>&1 &
VLLM_PID=$!
wait_for "http://127.0.0.1:${VLLM_PORT}/health" 1200 "$VLLM_PID" "vLLM"   # includes download
echo "   vLLM ready"

"$PY" -m relay.serve > "$LOGS/relay.log" 2>&1 &
RELAY_PID=$!
wait_for "http://127.0.0.1:${RELAY_PORT}/readyz" 60 "$RELAY_PID" "Relay"
echo "   Relay ready (standalone)"

status=0
if [ "$MODE" = "--quality" ]; then
  "$PY" -m bench.quality.gsm8k "$SCENARIO" || status=$?
else
  "$PY" -m bench.loadgen "$SCENARIO" --auto-rates || status=$?
fi

cleanup
trap - EXIT
KIND=""; [ "$MODE" = "--quality" ] && KIND="-gsm8k"
ZIP="/content/relay-results-${NAME}${KIND}-${STAMP}.zip"
[ -d /content ] || ZIP="relay-results-${NAME}${KIND}-${STAMP}.zip"
"$PY" - "$ZIP" <<'EOF'
import shutil, sys
shutil.make_archive(sys.argv[1].removesuffix(".zip"), "zip", ".", "results")
EOF
echo "== results zipped: $ZIP"
if [ -d /content/drive/MyDrive ]; then
  mkdir -p /content/drive/MyDrive/relay-results && cp "$ZIP" /content/drive/MyDrive/relay-results/
  echo "   copied to Google Drive: MyDrive/relay-results/"
fi
exit "$status"
