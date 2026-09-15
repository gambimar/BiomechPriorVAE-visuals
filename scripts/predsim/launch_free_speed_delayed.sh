#!/bin/bash
# Deferred launcher for run_free_speed.m.
#
# Waits DELAY seconds (default 4 h) for the account's other MATLAB -batch jobs
# to drain, then runs the free-speed PredSim set. MATLAB refuses a checkout with
# "Licensing error: -39,147" once the per-user concurrent-session cap is hit, so
# the delay alone is not enough: after waking it probes for a seat and retries
# rather than failing outright.
#
# Launch detached so it outlives the ssh session:
#   setsid nohup ~/phd/predsim/launch_free_speed_delayed.sh >/dev/null 2>&1 &

set -u

REPO=/home/rzlin/ri94mihu/phd/predsim/PredSim
RESULTS=/home/rzlin/ri94mihu/phd/predsim/results/free_speed
LOG="$RESULTS/launcher.log"
LOCK="$RESULTS/.launcher.lock"

DELAY=${DELAY:-14400}          # 4 h
RETRY_EVERY=${RETRY_EVERY:-900}  # probe every 15 min once awake
RETRY_HOURS=${RETRY_HOURS:-24}   # give up after a day of no free seat

mkdir -p "$RESULTS"

# refuse to stack a second launcher or a second solve on top of a running one
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "$(date '+%F %T') another launcher (pid $(cat "$LOCK")) is active; exiting" >> "$LOG"
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "launcher pid $$ started; sleeping ${DELAY}s (fires ~$(date -d "+${DELAY} seconds" '+%F %T'))"
sleep "$DELAY"

max_tries=$(( RETRY_HOURS * 3600 / RETRY_EVERY ))
for (( try=1; try<=max_tries; try++ )); do
    running=$(pgrep -u "$USER" -fc "MATLAB -batch" 2>/dev/null || echo 0)
    if matlab -batch "exit(0)" >/dev/null 2>&1; then
        log "license seat available (try $try, $running other -batch jobs); starting run_free_speed"
        cd "$REPO" || { log "cannot cd $REPO"; exit 1; }
        matlab -batch "run_free_speed" >> "$RESULTS/run_free_speed.log" 2>&1
        rc=$?
        log "run_free_speed exited rc=$rc"
        if [ -f "$RESULTS/timings.csv" ]; then
            log "timings:"; cat "$RESULTS/timings.csv" >> "$LOG"
        fi
        exit "$rc"
    fi
    log "no seat (try $try/$max_tries, $running other -batch jobs); retrying in ${RETRY_EVERY}s"
    sleep "$RETRY_EVERY"
done

log "gave up after ${RETRY_HOURS}h without a free license seat"
exit 1
