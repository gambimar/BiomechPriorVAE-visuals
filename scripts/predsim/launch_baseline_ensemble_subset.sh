#!/bin/bash
# Launcher for run_baseline_ensemble_subset.m -- runs ONE speed subset of the
# baseline ensemble (10 reps each) as its own detached process, so several
# subsets can run concurrently on separate license seats instead of one
# process working through all 8 speeds sequentially (see
# launch_baseline_ensemble.sh, superseded here for that reason).
#
# Usage:
#   launch_baseline_ensemble_subset.sh "1.2,1.4" v1214
#
# arg1: comma-separated speeds (m/s) this instance is responsible for.
# arg2: short tag used to namespace this instance's lock/log files so
#       multiple subset launches don't collide with each other or with a
#       plain launch_baseline_ensemble.sh run.
#
# All instances share resultsRoot/timings.csv -- run_baseline_ensemble_subset.m's
# restart-skip logic means whichever instance reaches a given (speed,rep)
# first "wins" and the others skip it, so overlapping speed sets across
# instances are wasteful but not unsafe.
#
# Launch detached so it outlives the ssh session:
#   setsid nohup ~/phd/predsim/launch_baseline_ensemble_subset.sh "1.2,1.4" v1214 >/dev/null 2>&1 &

set -u

SPEEDS_CSV=${1:?usage: launch_baseline_ensemble_subset.sh "SPEEDS_CSV" TAG}
TAG=${2:?usage: launch_baseline_ensemble_subset.sh "SPEEDS_CSV" TAG}

REPO=/home/rzlin/ri94mihu/phd/predsim/PredSim
RESULTS=/home/rzlin/ri94mihu/phd/predsim/results/baseline_ensemble_v2
LOG="$RESULTS/launcher_${TAG}.log"
LOCK="$RESULTS/.launcher_${TAG}.lock"

export LM_PROJECT=${LM_PROJECT:-iwse}

RETRY_EVERY=${RETRY_EVERY:-900}     # probe every 15 min
RETRY_HOURS=${RETRY_HOURS:-480}     # cold-started running gaits can take hours each

mkdir -p "$RESULTS"

# refuse to stack a second launcher for the SAME tag on top of a running one
# -- a different tag gets its own lock, so this does not block concurrent
# subsets the way launch_baseline_ensemble.sh's single global lock would.
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "$(date '+%F %T') another launcher (pid $(cat "$LOCK")) for $TAG is active; exiting" >> "$LOG"
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

MATLAB_SPEEDS="[$(echo "$SPEEDS_CSV" | tr ',' ' ')]"

log "launcher pid $$ started for speeds $MATLAB_SPEEDS (LM_PROJECT=$LM_PROJECT)"

max_tries=$(( RETRY_HOURS * 3600 / RETRY_EVERY ))
for (( try=1; try<=max_tries; try++ )); do
    running=$(pgrep -u "$USER" -fc "MATLAB -batch" 2>/dev/null || echo 0)
    if matlab -batch "exit(0)" >/dev/null 2>&1; then
        log "license seat available (try $try, $running other -batch jobs); starting run_baseline_ensemble_subset($MATLAB_SPEEDS)"
        cd "$REPO" || { log "cannot cd $REPO"; exit 1; }
        matlab -batch "run_baseline_ensemble_subset($MATLAB_SPEEDS)" >> "$RESULTS/run_baseline_ensemble_subset_${TAG}.log" 2>&1
        rc=$?
        log "run_baseline_ensemble_subset($MATLAB_SPEEDS) exited rc=$rc"
        if [ -f "$RESULTS/timings.csv" ]; then
            log "timings (shared across instances):"; tail -n 20 "$RESULTS/timings.csv" >> "$LOG"
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
