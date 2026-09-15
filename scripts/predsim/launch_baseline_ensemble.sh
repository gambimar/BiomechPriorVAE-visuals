#!/bin/bash
# Launcher for run_baseline_ensemble.m (8 speeds x 10 reps, perturbed-standing
# IG, speed imposed at each of 0.8/1.0/1.2/1.4/1.6/2.5/3.5/4.5 m/s).
#
# MATLAB must be checked out against the iwse license project -- see
# launch_free_speed_noisy.sh / AGENTS.md for why. Exported below so both the
# probe and the solve use it.
#
# run_baseline_ensemble.m is restartable -- (speed,rep) combos already in
# timings.csv are skipped -- so if a seat is lost mid-set, re-running this
# picks up where it stopped. 80 solves, several of them cold-started running
# gaits, is expected to take days; RETRY_HOURS is set generously long.
#
# Launch detached so it outlives the ssh session:
#   setsid nohup ~/phd/predsim/launch_baseline_ensemble.sh >/dev/null 2>&1 &

set -u

REPO=/home/rzlin/ri94mihu/phd/predsim/PredSim
RESULTS=/home/rzlin/ri94mihu/phd/predsim/results/baseline_ensemble
LOG="$RESULTS/launcher.log"
LOCK="$RESULTS/.launcher.lock"

export LM_PROJECT=${LM_PROJECT:-iwse}

RETRY_EVERY=${RETRY_EVERY:-900}     # probe every 15 min
RETRY_HOURS=${RETRY_HOURS:-480}     # 80 sequential solves need a very long window

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
        log "license seat available (try $try, $running other -batch jobs); starting run_baseline_ensemble"
        cd "$REPO" || { log "cannot cd $REPO"; exit 1; }
        matlab -batch "run_baseline_ensemble" >> "$RESULTS/run_baseline_ensemble.log" 2>&1
        rc=$?
        log "run_baseline_ensemble exited rc=$rc"
        if [ -f "$RESULTS/timings.csv" ]; then
            log "timings:"; cat "$RESULTS/timings.csv" >> "$LOG"
        fi
        # a lost seat mid-set exits non-zero with combos left; go back to
        # probing instead of giving up, since the script resumes where it
        # stopped
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
