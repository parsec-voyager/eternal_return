#!/bin/zsh
# The keeping-up battery: every way of flying the tests have been caught missing, on each map given, with the repo's
# add-on. For each map: plans made by tools/dive_back_plan.py (dive, glide, turns: each flown in, then full throttle
# backward), each flown --realtime with tools/probes/missing_view.py (the share of the view missing, live against
# settled), and the dive flown once more frame-stepped for its loops' and recenters' seams. One summary line a flight;
# the worst live|settled pictures are kept in the output folder, to be looked at.
#
#     tools/keeping_up.sh <out dir> [<map.blend> ...]          (no maps: DEFAULT_MAPS, every kind)
#     MODES="dive glide" DIVE=18 BACK=12 MEMORY=7000 tools/keeping_up.sh ...
#
# A 0.0% run counts only once its loops are checked (the drone may have flown off the map): every line has its loops.

set -u
cd "${0:A:h}/.."
# Awake for as long as this runs: the Mac idles to sleep with nobody at the keyboard, however busy, and a battery left
# overnight then flies its realtime flights in the few seconds of each maintenance wake
caffeinate -i -w $$ &
# The maps flown when none are given: one of every kind of map the add-on draws, so a change that breaks one kind
# shows. (With no Penrose map with frames here, a pickling bug that broke every Penrose and Darb map with frames went
# out with the build processes, retrying a failed build for 2.5 hours on the first one flown.) A new kind of map gets a
# map here. (Sphere maps of sphere.py, cube and icosahedron, aren't surface maps: dive_back_plan.py can't plan them.)
DEFAULT_MAPS=(
    ~/blender_tests/tile_object/tile_monkey.blend               # laid-out quads, corners of 3 and 5, open edges
    ~/blender_tests/tile_object/tile_monkey_triangles.blend     # laid-out triangles (Loop), corners of 4 to 8
    ~/blender_tests/tile_object/tile_torus_quads.blend          # one hole, conformal: every corner regular
    ~/blender_tests/tile_object/tile_text.blend                 # thin letters, many odd corners
    ~/blender_tests/surface/smooth/knot_smooth.blend            # a knotted tube (Catmull-Clark)
    ~/blender_tests/tile_object/tile_monkey_penrose.blend       # Penrose rhombs on a laid-out shape, with frames
    ~/blender_tests/penrose_surface/parts/sphere_penrose_frames.blend   # Penrose sphere with frames
    ~/blender_tests/darb_surface/parts/darb_torus_frames.blend  # Darb-i Imam with frames
    ~/blender_tests/substitution_plane/plane_penrose_frames.blend       # the endless Penrose plane with frames
)
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
out=${1:A}
shift
mkdir -p $out
modes=(${=MODES:-dive glide turns})
dive=${DIVE:-18}
back=${BACK:-12}
memory=${MEMORY:-7000}
(( $# )) || set -- $DEFAULT_MAPS
for map in "$@"; do
    [[ -f $map ]] || { echo "== no map $map"; exit 1; }
done

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
