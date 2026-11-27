#!/bin/zsh
# Does a change that shouldn't change behavior (a cleanup, a merge of duplicates) leave flights as they were? The
# survey's flights (~/blender_tests/flights_survey/list.txt: a map and a plan of every engine), frame-stepped and
# seeded, flown with a repo's add-on and harness, two at a time (about 45 minutes for the 15):
#
#     git worktree add <scratch>/head HEAD
#     tools/same_flights.sh <scratch>/head <out>/before
#     tools/same_flights.sh . <out>/after [map names ...]
#     tools/same_flights.sh --compare <out>/before <out>/after
#
# --compare (tools/checks/same_flights.py, needs ffmpeg): each flight's camera path and events must match exactly; its
# frames are compared as decoded pixels. Not as files: a PNG holds the time it was written. And not every frame
# matches on the same code: a tick builds for a budget of real seconds, so what's built by a given frame varies a
# little from run to run. Fly the baseline twice to see how much, before reading a difference as the change's.
set -u
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
if [[ $1 == --compare ]]; then
    python=(/Applications/Blender.app/Contents/Resources/*/python/bin/python3.*([1]))
    exec $python ${0:A:h}/checks/same_flights.py $2 $3
fi
repo=${1:A}; out=${2:A}; shift 2
mkdir -p $out
caffeinate -i -w $$ &       # awake for as long as this runs (see keeping_up.sh)
fly() {
    name=$1; map=$HOME/blender_tests/$2; plan=${3/#\~/$HOME}
    $repo/tools/guard.sh 6000 $BLENDER -b $map --python $repo/tools/flight_sim.py -- $plan --out $out/$name \
        --every 30 --size 320x180 --addon working --timing --seed 1 > $out/$name.out 2>&1
    echo "$name: $(grep -c Traceback $out/$name.out) tracebacks, $(grep '^FLIGHT done' $out/$name.out | cut -c1-100)"
}
while read name map plan; do
    if (( $# )) && [[ ${@[(i)$name]} -gt $# ]]; then continue; fi
    fly $name $map $plan &
    while (( ${#jobstates} >= 2 )); do sleep 1; done
done < ~/blender_tests/flights_survey/list.txt
wait
