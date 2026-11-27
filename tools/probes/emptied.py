"""flight_sim probe: what makes a substitution or cells map build on the main thread in flight (EMPTY lines):
- a settle that leaves the map with no level drawn where it had some (the next settle then builds every level here
  and now, as at a flight's start), with what was drawn and the last 40 settles' counts
- a new state made for a map that had one (state_of())
- every level built in the tick (build_level() with wait, or no worker threads, or after a failed build)
"""
import functools
import importlib

HISTORY = []        # each settle's (world n, levels drawn, the coarsest, the finest) before it


def setup(m):
    package = m.__name__.rpartition(".")[0]
    ss = importlib.import_module(package + ".surface_substitution")
    surface = importlib.import_module(package + ".surface")
    background = importlib.import_module(package + ".background")
    original_state, original_settle, original_build = ss.state_of, ss.settle, ss.build_level

    @functools.wraps(original_state)
    def state_of(obj, top):
        st = ss._states.get(obj.name)
        out = original_state(obj, top)
        if st is not None and out is not st:
            print(f"EMPTY new state for {obj.name} (key {st.key} -> {out.key}, had {len(st.drawn)} drawn)", flush=True)
        return out

    # (functools.wraps: the harness reads settle's parameters, see hitches.py)
    @functools.wraps(original_settle)
    def settle(obj, *args, **kwargs):
        st = ss._states.get(obj.name)
        drawn = sorted(st.drawn) if st else []
        n_before = surface.world_of(obj).n if surface.is_surface(obj) else None
        records = {d: (r.coarse, r.cover, r.empty, round(float(r.window), 4), r.n)
                   for d, (_o, r) in st.drawn.items()} if st else {}
        HISTORY.append((n_before, len(drawn), drawn[0] if drawn else None, drawn[-1] if drawn else None))
        out = original_settle(obj, *args, **kwargs)
        st = ss._states.get(obj.name)
        if drawn and not (st and st.drawn):
            print(f"EMPTY settle emptied {obj.name}: n {n_before} -> {surface.world_of(obj).n}, had {records}; the "
                  f"last 40 settles (n, drawn, coarsest, finest): {HISTORY[-40:]}", flush=True)
        return out

    @functools.wraps(original_build)
    def build_level(*args, **kwargs):
        world, st, d, coarse = args[1], args[3], args[5], args[7]
        wait = kwargs.get("wait", True)
        if wait or not background.enabled or d in st.failed:
            print(f"EMPTY main-thread build n {world.n} d {d} coarse {coarse} (wait {wait}, workers "
                  f"{background.enabled}, failed {d in st.failed}, drawn {len(st.drawn)})", flush=True)
        return original_build(*args, **kwargs)

    ss.state_of, ss.settle, ss.build_level = state_of, settle, build_level
