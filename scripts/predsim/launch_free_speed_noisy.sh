#!/bin/bash
# Launcher for run_free_speed_noisy.m (10 reps, noisy IG, speed free in [0.5,2]).
#
# MATLAB must be checked out against the iwse license project. Without
# LM_PROJECT=iwse the checkout lands in a pool the account's other -batch jobs
# have already saturated and fails with "Licensing error: -39,147"; with it, a
# seat is available immediately. Exported below so both the probe and the solve
# use it.
#
# The seat probe is kept anyway as a cheap guard, and it keeps retrying for
# RETRY_HOURS. Unlike launch_free_speed_delayed.sh there is no fixed up-front
# delay: it starts as soon as a seat is confirmed.
#
# run_free_speed_noisy.m is restartable -- reps already in timings.csv are
# skipped -- so if a seat is lost mid-set, re-running this picks up where it
# stopped.
#
# Launch detached so it outlives the ssh session:
#   setsid nohup ~/phd/predsim/launch_free_speed_noisy.sh >/dev/null 2>&1 &

set -u

REPO=/home/rzlin/ri94mihu/phd/predsim/PredSim
RESULTS=/home/rzlin/ri94mihu/phd/predsim/results/free_speed_noisy
LOG="$RESULTS/launcher.log"
LOCK="$RESULTS/.launcher.lock"

export LM_PROJECT=${LM_PROJECT:-iwse}

RETRY_EVERY=${RETRY_EVERY:-900}   # probe every 15 min
RETRY_HOURS=${RETRY_HOURS:-48}    # 10 sequential solves need a long window

mkdir -p "$RESULTS"

# refuse to stack a second launcher or a second solve on top of a running one
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "$(date '+%F %T') another launcher (pid $(cat "$LOCK")) is active; exiting" >> "$LOG"
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "launcher pid $$ started; probing for a MATLAB license seat (LM_PROJECT=$LM_PROJECT)"

max_tries=$(( RETRY_HOURS * 3600 / RETRY_EVERY ))
for (( try=1; try<=max_tries; try++ )); do
    running=$(pgrep -u "$USER" -fc "MATLAB -batch" 2>/dev/null || echo 0)
    if matlab -batch "exit(0)" >/dev/null 2>&1; then
        log "license seat available (try $try, $running other -batch jobs); starting run_free_speed_noisy"
        cd "$REPO" || { log "cannot cd $REPO"; exit 1; }
        matlab -batch "run_free_speed_noisy" >> "$RESULTS/run_free_speed_noisy.log" 2>&1
        rc=$?
        log "run_free_speed_noisy exited rc=$rc"
        if [ -f "$RESULTS/timings.csv" ]; then
            log "timings:"; cat "$RESULTS/timings.csv" >> "$LOG"
        fi
        # a lost seat mid-set exits non-zero with reps left; go back to probing
        # instead of giving up, since the script resumes where it stopped
        if [ "$rc" -eq 0 ]; then exit 0; fi
        log "non-zero exit; will re-probe and resume remaining reps"
        sleep "$RETRY_EVERY"
        continue
    fi
    log "no seat (try $try/$max_tries, $running other -batch jobs); retrying in ${RETRY_EVERY}s"
    sleep "$RETRY_EVERY"
done

log "gave up after ${RETRY_HOURS}h without completing"
exit 1
