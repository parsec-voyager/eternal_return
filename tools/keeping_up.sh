#!/bin/zsh
# The keeping-up battery (2026-11-26): every way of flying the tests have been caught missing, on each map given, with
# the repo's add-on. For each map: plans made by tools/dive_back_plan.py (dive, glide, turns: each flown in, then full
# throttle backward), each flown --realtime with tools/probes/missing_view.py (the share of the view missing, live
# against settled), and the dive flown once more frame-stepped for its loops' and recenters' seams. One summary line a
# flight; the worst live|settled pictures are kept in the output folder, to be looked at.
#
#     tools/keeping_up.sh <out dir> <map.blend> [<map.blend> ...]
#     MODES="dive glide" DIVE=18 BACK=12 MEMORY=7000 tools/keeping_up.sh ...
#
# A 0.0% run counts only once its loops are checked (the drone may have flown off the map): every line has its loops.

set -u
cd "${0:A:h}/.."
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
out=${1:A}
shift
mkdir -p $out
modes=(${=MODES:-dive glide turns})
dive=${DIVE:-18}
back=${BACK:-12}
memory=${MEMORY:-7000}

for map in "$@"; do
    name=${map:t:r}
    for mode in $modes; do
        plan=$out/${name}_$mode.json
        $BLENDER -b $map --factory-startup --python tools/dive_back_plan.py -- $plan $dive $back $mode > $out/${name}_$mode.plan.out 2>&1
        if [[ ! -s $plan ]]; then
            echo "== $name $mode: no plan ($(grep dive_back_plan $out/${name}_$mode.plan.out | tail -1))"
            continue
        fi
        flight=$out/${name}_$mode
        mkdir -p $flight
        MISSING_OUT=$flight MISSING_EVERY=90 tools/guard.sh $memory $BLENDER -b $map --python tools/flight_sim.py -- \
            $plan --out $flight --every 100000 --size 800x450 --addon working --timing --realtime \
            --probe tools/probes/missing_view.py > $flight.out 2>&1
        echo "== $name $mode: $(grep -c LOOP $flight/events.txt 2>/dev/null) loops |" \
             "$(grep '^MISSING [0-9]' $flight.out | sed 's/MISSING [0-9]* samples: //') |" \
             "$(grep '^TIMING' $flight.out | sed 's/TIMING //' | cut -c1-120) | $(grep -c Traceback $flight.out) tracebacks"
        echo "   by time: $(grep '^MISSING t=' $flight.out | awk '{print $2, $4}' | tr '\n' ' ')"
    done
    # Seams: the dive frame-stepped, every loop and recenter rendered just before and after
    plan=$out/${name}_dive.json
    if [[ -s $plan ]]; then
        flight=$out/${name}_seams
        tools/guard.sh $memory $BLENDER -b $map --python tools/flight_sim.py -- $plan --out $flight --every 90 \
            --size 640x360 --addon working > $flight.out 2>&1
        echo "== $name seams: $(grep '^FLIGHT done' $flight.out | grep -o '[0-9]* loops.*jumps') |" \
             "$(grep -c Traceback $flight.out) tracebacks"
        grep "SEAM" $flight/events.txt 2>/dev/null | sed 's/^/   /' | head -5
        rm -f $flight/*.png(N)
    fi
done
