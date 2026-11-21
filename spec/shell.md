# The add-on's shell: controls, operators, panels, sound, registration

This is a behavior spec for the parts of the Eternal Return add-on that started life as another author's code and
are being rewritten from scratch (clean room). Everything here says *what* the code must do. How to write it is up to
you; there is no reference code to look at, and there must not be (see [Rules](#rules)).

The add-on flies a virtual FPV drone through Blender scenes with a real RC radio or gamepad, and lets the pilot change
the drone's scale in flight. All of the flight itself, the physics, the HUD, the maps and the scaling tools already
exist as the add-on's own modules. What's missing is the shell around them: reading the controller, turning sticks into
controls, the Fly operator's modal loop, a few small operators, the side panel, the propeller sound and registration.

## Rules

- Read only: this spec, `spec/` in this repo, and the add-on's own modules listed under
  [What exists](#what-exists), `tools/`, `README.md` files, Blender's and PySDL2's own documentation.
- Never read, search or diff: `~/workspaces/dronecam-2.3.0` (anything in it), any file or folder outside this
  worktree, this repo's git history (`git log -p`, `git show`, `git diff` against older commits, `git blame`,
  reflog, stashes, other branches), installed copies of the add-on under `~/Library/Application Support/Blender/`
  or any `~/blender_tests/*/profile*` folder (they hold the old code), and `.zip` builds. If you are unsure whether
  something is allowed, don't open it.
- The files you write are new. Don't try to recreate any earlier version from memory.
- You may edit the add-on's own modules only where this spec says so (their calls into the shell).

## Words

- **Control**: one of the 11 things a stick axis can drive, in this order:
  `throttle, yaw, pitch, roll, scale, zoom, orbit, click, camera, lens_clip, brake`.
- **Flight controls**: `throttle, yaw, pitch, roll`. The physics core (`physics.py`, the `eternal_return_core` wheel)
  converts these.
- **One-way controls**: `throttle, orbit, click, brake`. On an RC radio their stick rests at the bottom (−1) and the
  control is off there; with *Centered* (gamepads) the stick rests at its center (0) and only the upper half does
  anything. Throttle in `3D` flying mode is the exception: its whole travel is used, below center is reverse thrust.
- **Factor controls**: `scale, zoom, camera, lens_clip, brake`. Their converted value is a factor per second:
  `factor ** x`, 1.0 at rest.
- **Raw axes**: the controller's axes, each normalized to [−1, 1) as SDL reports them divided by 32768.
- **Raw controls**: for each control, the raw axis value it is mapped to (or a rest value, below).
- **Converted controls**: what each control does: newtons for throttle, radians per second for yaw/pitch/roll, a
  per-second factor for the factor controls, 0..1 for orbit and click.

## What exists

These are the add-on's own modules, written for this add-on; read them freely. Package folder: `eternal_return/`.

- `flight.py`: the flight. `FlightFeatures` is a mixin class the Fly operator inherits from; it holds every flight
  method and the hooks below. Module globals hold the flight's state (`drone_obj`, `drone_camera`, `selected_obj`,
  `dict_controls` = the converted controls as the last tick left them, `record_start_frame`, `record_end_frame`, ...).
  It also defines the Reset operators (`ER_OT_ResetScale`, `..._ResetLocations`, `..._ResetCamera`,
  `..._ResetLensClip`, `..._ResetMinScale`, `..._ResetMaxScale`) and `fly_frame`, the frame handler.
- `physics.py`: the drone's physics on the core, its settings (`scene.drone_physics`), recordings
  (`physics.Recording`), `sticks_to_commands(raw_controls, scene)` (the core's conversion of the flight controls:
  returns `throttle` in N and `yaw`, `pitch`, `roll` in rad/s), `available()`, `stop()`, `replay(scene)`.
- `rig.py`: builds the drone and its camera (`build_drone(scene)`, `build_camera(scene, drone_obj, clip_start)`) and
  migrates old scenes (`migrate(scene, drone_obj, camera_obj, depsgraph)`).
- `hud.py`, `hud_draw.py`, `hud_render.py`: the HUD (`hud.legacy_hud`, `hud.delete_legacy`, `hud.migrate`,
  `hud_render.update/remove`, `scene.drone_hud`).
- `gamepad.py`: reads a gamepad's sticks for the HUD's gimbals (uses SDL directly).
- `scaling_settings.py`: `ScalingSettings`, a mixin of annotations (the scaling tools' settings) that the controls
  settings group must inherit, and `DEFAULT_MIN_SCALE` / `DEFAULT_MAX_SCALE`.
- `ui_features.py`: the side panel's own-feature parts: the `ER_PT_UI_HUD` ("UI") and `ER_PT_UI_Drone`
  ("Drone") subpanels, and `draw_scaling_tools(layout, context)` for the Controls panel's Scaling Tools section.
- Everything else in the package (maps, tiling, sphere, surface, ...) and their `register()` / `unregister()`.
- `tools/flight_sim.py`: the flight harness (see [Testing](#testing)).

## What to write

Five modules in `eternal_return/`:

| module | holds |
|---|---|
| `controls.py` | the controller and controls settings groups, SDL controller access, raw controls, conversion |
| `operators.py` | the Fly operator, Add Drone, Bake Animation, Store/Load Parameters, the add-on preferences, the References and Other settings groups |
| `panels.py` | the side panel and its Controls, Record and Extras subpanels |
| `sound.py` | the propeller sound |
| `__init__.py` | registration |

Keep every Blender identifier listed here exactly (operator `bl_idname`s, panel `bl_idname`s, scene attribute names,
property identifiers, enum item identifiers): scenes saved with the add-on store their settings under those names, and
other modules refer to them. Class names may be your own except where listed. Write your own labels, tooltips
(descriptions) and comments; the labels given here are the ones the user sees today and should stay as they are.

Code style: match the add-on's own modules (descriptive docstrings, comments that explain *why*, plain functions over
classes where a class adds nothing). Tables over repetition: the 11 controls share most settings, so generate what you
can from one table.

## controls.py

### Settings groups

Each is a `bpy.types.PropertyGroup` attached to `bpy.types.Scene` under the attribute name given.

**`scene.controls_axis_mapping`**: for each control, an enum property named after the control (`throttle`, `yaw`, ...)
choosing the raw axis it reads: items `'none'` (label empty, "not mapped") and `'0'` .. `'7'` (labels `0` .. `7`).
Defaults: throttle `'2'`, yaw `'0'`, pitch `'1'`, roll `'3'`, scale `'5'`, zoom `'none'`, orbit `'4'`, click `'7'`,
camera `'none'`, lens_clip `'6'`, brake `'none'`.

**`scene.joystick_axis_inputs`**: the raw axes, for display: float properties `joystick_axis0_input` ..
`joystick_axis7_input`, default 0.0, shown with 1 decimal.

**`scene.controls_parameters`** (inherits `scaling_settings.ScalingSettings` and `PropertyGroup`):

| property | type | default | notes |
|---|---|---|---|
| `flying_mode` | enum `ACRO`, `3D`, `HORIZON`, `ANGLE` | `3D` | labels "Acro (Hard)", "3D (Hard)", "Horizon (Easy)", "Angle (Easy)" |
| `<control>_invert` | bool, for every control | `pitch`, `scale`, `lens_clip` True, others False | |
| `<control>_deadzone` | float percent 0..100, for every control | 10.0 | subtype `PERCENTAGE`, 1 decimal |
| `<control>_centered` | bool, for each one-way control | False | |
| `scale_factor` | float, min 1 | 6.0 | |
| `zoom_factor` | float, min 1 | 2.0 | |
| `camera_factor` | float, min 1 | 2.0 | |
| `lens_clip_factor` | float, min 1 | 8.0 | |
| `brake_factor` | float, min 1 | 10.0 | |
| `autoleveling_strength` | float percent 0..100 | 20.0 | subtype `PERCENTAGE`, 0 decimals; Horizon and Angle modes |
| `angle_limit_value` | float angle, soft 0°..180° | 60° (in radians) | subtype `ANGLE`; Angle mode |

The scaling tools' settings come from the mixin; don't redefine them.

**`scene.throttle_control_power`**: `power`, float, default 30.0 (newtons at full throttle), soft range 0..100,
1 decimal. Label "Maximum Thrust".

**`scene.rotation_controls_actual_rates`**: for each of yaw, pitch, roll: `<axis>_center_sensitivity` (angle per
second, default 145°, soft 50°..1000°), `<axis>_max_rate` (default 533°, soft 50°..1000°), both subtype `ANGLE`,
1 decimal, stored in radians; `<axis>_expo` (float 0..1, default 0.5, 2 decimals).

**`scene.controls_output`**: one float property per control (named after it), default 0.0, holding the last converted
value; yaw, pitch, roll subtype `ANGLE`. Read-only in spirit (display only).

### SDL controller access

The SDL2 Python bindings are the bundled `sdl2` module (PySDL2 with its DLL wheel).

- `get_controller(report)`: open and return the controller to fly with, or None. `report(type_set, message)` is the
  operator's report function.
  - Before initializing SDL, set the hint `SDL_HINT_JOYSTICK_MFI` to `"0"`. Since macOS 27.2, Apple's
    GameController framework claims USB radios like the RadioMaster Pocket but never hands them to SDL; with the MFi
    driver off, SDL finds them through IOKit. The hint only takes effect before `SDL_Init`.
  - Initialize SDL with the joystick and game controller subsystems (the latter loads SDL's gamepad mappings, which
    `gamepad.open_gamepad()` needs).
  - Look through every joystick SDL lists, printing each one's index and name, and its axis count once opened. Fly
    with the first one whose name doesn't contain `vJoy` (report INFO `Controller found: <name>`). A vJoy device is a
    Windows virtual joystick: report WARNING `vJoy virtual controller detected.`, keep the first one as a fallback
    and close any others. If only vJoy was found, fly with it (INFO `No valid controllers found other than vJoy.
    Using vJoy.`). Close a kept vJoy fallback that ends up unused.
  - Nothing usable: report WARNING `No valid controllers found!` when there were joysticks, `No controllers found!`
    when there were none, and return None.
  - Never return a handle from an earlier flight: each call opens afresh.
- `read_axes(controller)`: update SDL's joystick state, then return every axis of the controller as a raw axis
  (`SDL_JoystickGetAxis(...) / 32768`), in axis order, as a list of floats. Also store the first eight into
  `scene.joystick_axis_inputs` for the panel.
- `close_controller(controller)` and `quit_sdl()`: close the joystick handle (if any) and shut SDL down.

### Raw controls

`map_controls(axes, scene) -> dict`: a raw value for each of the 11 controls. A control mapped to an axis that exists
gets that axis's value. A control that is unmapped, or mapped past the controller's last axis, gets the raw value that
converts to "at rest":

- throttle: 0.0 in `3D` mode, else −1.0; negated if `throttle_invert`;
- orbit, click, brake: −1.0, negated if the control's invert is on;
- all others: 0.0.

These raw controls are also what `physics.sticks_to_commands()` and the HUD read, so they must be exactly the stick
positions as described (no inversion or deadzone applied).

### Conversion

`convert(raw, scene) -> dict`: the converted value of each of the 11 controls.

- The four flight controls come from `physics.sticks_to_commands(raw, scene)` (the core: its own invert, deadzone,
  throttle range, thrust curve and rates, from the same scene settings). Call it once per conversion: it also keeps
  the settings the physics flies the frame with.
- Each other control, from its raw value `x`, in this order:
  1. **Invert**: `x = -x` if `<control>_invert`.
  2. **One-way range** (orbit, click, brake): without `<control>_centered`, map [−1, 1] to [0, 1]: `x = (x + 1) / 2`.
     Then, centered or not, clamp to [0, 1].
  3. **Deadzone** `z` = `<control>_deadzone / 100`, if `z > 0`:
     `x = sign(x) * clamp((|x| − z) / (1 − z), 0, 1)`; a stick inside the deadzone gives exactly 0. A deadzone of
     100% (z ≥ 1) leaves the control always 0 (never divide by zero).
  4. **Factor controls** (scale, zoom, camera, lens_clip, brake): `x = <control>_factor ** x`.
  5. Make sure the value is a plain Python float.
- Write every converted value into `scene.controls_output` (by control name).

`flight.py` calls `controls.map_controls()` through the operator's `read_controls()` and `controls.convert()` itself
every tick (see below); `hud.py` reads the settings above by name.

## operators.py

### Settings groups

**`scene.references`**:
- `drone_obj`, `camera_obj`: pointers to objects (the scene's drone and its camera).
- `world_offset`: float vector, size 3, subtype `TRANSLATION`, default zeros. Tooltip: "How far Keep Drone at Origin
  has moved the scene's objects. Reset Locations moves them back."

**`scene.other_parameters`**:
- `record`: bool, default False, label "Record": record the flight's inputs as keyframes (physics.py records them).
- `record_camera`: bool, default False, label "Record Camera": record Drone_Camera's transforms onto another camera,
  for rendering.
- `record_obj`: string, default empty, label "Camera Object": the name of that camera.
- `volume`: float percent 0..100, default 10.0, 0 decimals, label "Sound Volume".
- `drone_sound`: enum of sound files in the add-on's `sound/` folder (`sound.SOUNDS`); identifiers are the file
  names. Items, in order (identifier: label): `propellers.wav`: Drone Propellers; `spaceship.wav`: Spaceship;
  `ufo_8bit.wav`: 8-bit UFO; `breakbeat.wav`: Breakbeat (or, when that local file is installed,
  `oggggm-power_176-93513.mp3`: Amen Break, at speed 0.4); `alien.wav`: Alien Spaceship. Default `propellers.wav`.
  (The add-on's files are made by tools/make_sounds.py.)

**`bpy.types.WindowManager.er_fly_modal_running`**: bool, default False. True while flying. On the window
manager so it is never saved into a file. Other modules read it (the HUD draws flight marks only while it's True).

### Add-on preferences

An `AddonPreferences` class with `bl_idname = __package__` holding one pointer property for each of the four groups
`controls_axis_mapping`, `controls_parameters`, `throttle_control_power`, `rotation_controls_actual_rates` (same
attribute names as on the scene, same group classes).

### `er.fly` ("Fly Drone Camera"): the Fly operator

The operator class inherits `flight.FlightFeatures` first, then `bpy.types.Operator`. Class name
`ER_OT_Fly` (the flight harness looks it up by that name).

`FlightFeatures` provides these hooks, which the operator calls:

| hook | when | does |
|---|---|---|
| `prepare_flight(context) -> bool` | first, before the controller | finds the drone, gets the physics and maps ready; False (already reported) if it can't fly |
| `abandon_flight(context)` | if no controller after `prepare_flight` succeeded | undoes the physics' start |
| `start_flight(context, controller)` | once the controller is open | opens the gamepad view of it, adds the frame handler |
| `tick(context)` | on every timer event | one step of flight; it calls back `read_controls()` and `update_sound(context)` |
| `end_flight(context)` | in cancel, after playback stops | stops the physics, removes the frame handler, handles a recorded flight's return to its start |

The operator provides:

- `read_controls() -> (raw_controls, axes)`: `axes = controls.read_axes(self's controller)`,
  `raw_controls = controls.map_controls(axes, scene)`.
- `update_sound(context)`: the propeller sound follows the throttle: with `power` = `throttle_control_power.power`,
  the sound's throttle is `|flight.dict_controls["throttle"]| / power` (0 when power ≤ 0).

**execute** (the panel's button toggles flying):
1. If `er_fly_modal_running` is already True, set it False and finish with CANCELLED: the running operator
   sees the flag on its next event and stops itself.
2. `prepare_flight(context)`; on False, CANCELLED.
3. Open the controller (`controls.get_controller(self.report)`). None: shut SDL down, `abandon_flight(context)`,
   CANCELLED.
4. `start_flight(context, controller)`.
5. Remember the scene's `sync_mode` and set it to `'NONE'`: every frame must be played (and simulated), none skipped
   to keep up with real time.
6. Start the sound (`sound.PropellerSound()`, see below).
7. Start timeline playback if it isn't playing (`bpy.ops.screen.animation_play()` toggles, so only call it when
   stopped).
8. Add a 0.01 s window timer, add the operator as a modal handler, set `er_fly_modal_running` True, return
   RUNNING_MODAL.

**modal**: if the flag went False or the event is ESC, cancel and return CANCELLED. On a TIMER event, `tick(context)`.
If the tick raises, print the traceback, report ERROR `The drone stopped after an error. See the system console for
details.`, cancel and return CANCELLED: an exception escaping the modal would end the operator without cancel(),
leaving playback, the frame handler, SDL and the sound running. Every other event passes through (the pilot can still
use Blender while flying).

**cancel** (also called by Blender itself, e.g. when a file loads; everything may be partly set up):
1. Set the flag False; restore the remembered `sync_mode` (if one was remembered); remove the timer (if any); stop
   playback if it's playing.
2. `end_flight(context)`.
3. Close the controller and shut SDL down; stop the sound (if started).
4. Report INFO `Flying Stopped. Inputs recorded. Play the timeline to replay them.` when `other_parameters.record` is
   on, else `Flying Stopped.`

### `er.add_to_scene` ("Add Drone to Scene")

Gives the scene a fresh drone:
1. Remove DroneCam's old HUD wherever it is in the file: every object `obj` with `hud.legacy_hud(obj.parent) == obj`
   goes through `hud.delete_legacy(obj)`. Then `hud_render.remove(scene)`.
2. Remove every earlier drone from the file (`do_unlink`): mesh objects whose name starts with `Drone` and whose mesh
   name starts with `_ER_Drone`, and any leftover meshes named so; camera objects whose name and whose camera
   data's name start with `Drone_Camera`, and any leftover cameras named so; node groups whose name starts with
   `_DroneCam_`.
3. `drone = rig.build_drone(scene)`, set `references.drone_obj`; `camera = rig.build_camera(scene, drone,
   controls_parameters.default_clip_start)`, set `references.camera_obj`.
4. `scene.drone_hud.hud_read = True` (the drawn HUD reads the scene's settings), then `hud_render.update(scene)`.
5. Deselect everything, select the drone and make it active. Report INFO `Drone added to the scene.`

### `er.bake_animation` ("Bake Animation")

Turns the drone camera's flight into an ordinary keyframed camera the user can edit.

- Properties `start_frame`, `end_frame` (ints, min 1, defaults 1 and 250).
- Both invoke and execute first migrate the scene: `rig.migrate(scene, references.drone_obj, references.camera_obj,
  context.evaluated_depsgraph_get())` then `hud.migrate(scene, references.camera_obj)`.
- **invoke**: the frame range is the drone's recording's (`physics.Recording.of(drone_obj)` if
  `physics.available()`; its `start_frame` and `end_frame()`), else the last flight's (`flight.record_start_frame`,
  `flight.record_end_frame`). Then show the operator's properties dialog.
- **execute**: without `references.camera_obj`, report ERROR `No Drone Camera detected. Click Add Drone first` and
  CANCELLED. Otherwise add a camera object named `Editable Drone Camera` at the origin with the drone camera's lens,
  clip start and clip end; give it Copy Location and Copy Rotation constraints targeting the drone camera; make it the
  only selected object and the active one; update the view layer; bake it with `bpy.ops.nla.bake` over the frame
  range: selected only, visual keying, constraints cleared, object data, location and rotation channels. Report INFO
  `Baked Camera Animation. Location/Rotation keyframes can now be edited in the graph editor.`

Playback of the range replays the recording on the core as the bake steps through frames (physics.py's frame
handler), so nothing else is needed.

### `er.store_parameters` / `er.load_parameters`

Labels "Store Parameters in Addon Preferences" / "Load Parameters from Addon Preferences". Copy every add-on-defined
property of the four groups (`controls_axis_mapping`, `controls_parameters`, `throttle_control_power`,
`rotation_controls_actual_rates`) from the scene to the preferences, or back. "Add-on-defined" includes the
properties `controls_parameters` gets from its `ScalingSettings` mixin, and excludes Blender's own (`name`,
`rna_type`). Report INFO `Stored Parameters` / `Loaded Parameters`.

### Dropped

The old shell had a "Load Racetrack_01" operator opening a bundled demo map. Don't write it: the map is not ours to
ship.

## panels.py

All panels: space `VIEW_3D`, region `UI` (the N sidebar), category tab `Eternal Return`. Subpanels have
`bl_parent_id = "ER_PT_UI"` and `bl_context = "scene"`. Registration order (it's also the order on screen):
`ER_PT_UI`, `ER_PT_UI_HUD` and `ER_PT_UI_Drone` (both from `ui_features.py`),
`ER_PT_UI_Controls`, `ER_PT_UI_Record`, `ER_PT_UI_Extras`.

**`ER_PT_UI`** (label "Eternal Return"): two big buttons (rows scaled 1.5 high): "Add Drone"
(`er.add_to_scene`, icon `CON_CAMERASOLVER`), then the Fly toggle (`er.fly`): "Start Flying" with icon
`PLAY` while not flying, "STOP" with icon `PAUSE` while flying.

**`ER_PT_UI_Controls`** (label "Controls", open by default):
1. A box "Controls Parameters:" (icon `PREFERENCES`) with buttons "Store Parameters" and "Load Parameters" side by side.
2. A row (1.5 high) split in halves: "Fly Mode" (`flying_mode`); next to it "Auto-Level Strength" in Horizon and Angle
   modes, plus "Angle Limit" in Angle mode; an empty label in Acro and 3D (keeps the layout steady).
3. One box per control, stacked in one aligned column, in the order throttle, yaw, pitch, roll, scale, zoom, then the
   Scaling Tools section, then orbit, click, camera, lens_clip, brake. Each control's box: a title (icon below), then
   a row "Axis" (its mapping), "Invert", "Deadzone", then its own rows:

   | control | title, icon | own rows |
   |---|---|---|
   | throttle | Throttle, `UV_SYNC_SELECT` | "Maximum Thrust" (`throttle_control_power.power`); "Centered" next to it except in 3D mode |
   | yaw, pitch, roll | Yaw / Pitch / Roll, `FORCE_MAGNETIC` | "Center Sensitivity", "Max Rate", "Expo" (its rates) |
   | scale | Scale, `UV_SYNC_SELECT` | "Scale Factor"; then "Lens Clip" (`scale_lens_clip`), "Camera" (`scale_camera`), and a "Reset" button (`er.reset_scale`, icon `LOOP_BACK`) |
   | zoom | Zoom, `UV_SYNC_SELECT` | "Zoom Factor"; then "Lens Clip" (`scale_lens_clip_zoom`), "Camera" (`scale_camera_zoom`) |
   | orbit | Orbit, `FORCE_MAGNETIC` | "Centered", "Orbit Axes" (`orbit_axes`) |
   | click | Click, `RADIOBUT_ON` | "Centered", "Auto Deselect"; then "Click Override" (`selected_obj`) |
   | camera | Camera, `UV_SYNC_SELECT` | "Camera Factor"; then "Lens Clip" (`scale_lens_clip_with_camera`) and "Reset" (`er.reset_camera`, `LOOP_BACK`) |
   | lens_clip | Lens Clip, `UV_SYNC_SELECT` | "Lens Clip Factor"; then "Clip Start" (`default_clip_start`) and "Reset" (`er.reset_lens_clip`, `LOOP_BACK`); then "Maintain Ratio", "Bounded" |
   | brake | Brake, `FORCE_DRAG` | "Brake Factor"; then "Centered", "Auto Brake" (`auto_brake`) |

   The Scaling Tools section is a box holding a collapsible layout panel (`layout.panel("ER_scaling_tools",
   default_closed=True)`) whose header reads "Scaling Tools" (icon `LOOP_BACK`); its body is drawn by
   `ui_features.draw_scaling_tools(body, context)` when open.
4. Below the boxes, a label "Controls Outputs:" (icon `GRAPH`) and a box with every control's `controls_output` value
   in a 2-column grid.
5. A label "Raw Controller Axis Inputs:" (icon `CONSOLE`) and a box with the eight `joystick_axis_inputs`, 4 columns,
   labelled "Axis 0:" .. "Axis 7:".

**`ER_PT_UI_Record`** (label "Record"): a row with the "Record" toggle (icon `RECORD_ON` when on, `RECORD_OFF`
when off) and the "Bake Animation" button (`er.bake_animation`, icon `AUTOMERGE_OFF`); a row "Record Camera" and
"Camera Object" (`record_obj`).

**`ER_PT_UI_Extras`** (label "Extras"): "Drone Sound" (`drone_sound`), then "Sound Volume" (`volume`, as a
slider).

## sound.py

`PropellerSound()`: plays the scene's chosen `other_parameters.drone_sound` file from the add-on's `sound/` folder,
looping forever, on Blender's `aud` audio device.
- Starts silent (device and sound volume 0) until the first `update()`.
- `update(throttle)`, throttle 0..1 (values outside are clamped): device volume = `other_parameters.volume` (the
  percent value used directly as aud's volume factor, so the default 10 plays loud; keep it so, the user's settings
  are tuned to it); sound volume = throttle; pitch = (0.5 + 2 × throttle) × the sound's speed (0.5 at idle, 2.5 at full, at speed 1). If the chosen
  sound changed since it started, switch to the new one first.
- `stop()`: stop everything it plays.

## \_\_init\_\_.py

`register()`:
1. Register the classes: every settings group (before anything that points to it), the operators (`operators.py`
   and the Reset operators from `flight.py`), the add-on preferences, then the panels in the order above.
2. `physics.register()`, `hud.register()`, `hud_render.register()`.
3. The scene attributes of every settings group (names above) as `PointerProperty`s, and the window manager's
   `er_fly_modal_running`.
4. `fractal.follow_eyes` appended to `bpy.app.handlers.frame_change_pre` if it isn't there (endless fractals' eyes
   follow the drone camera once a frame, flying or not).
5. `solid_sky.register()`, `mandelbrot.register()`, `sphere.register()`, `parts.register()`, `surface.register()`,
   `hud_draw.register()`.

`unregister()` undoes all of it (classes in reverse order).

## Calls the add-on's own modules make into the shell

These exist today and must keep working:
- `flight.py`: `controls.convert(raw, scene)` (in `fly_tick`), and the hooks above.
- `hud.py`: reads `controls_axis_mapping` and `controls_parameters` settings by name.
- `physics.py`: reads `controls_parameters`, `throttle_control_power`, `rotation_controls_actual_rates` by name.
- `hud_draw.py`, `ui_features.py`: read `er_fly_modal_running`, `references`.
- `tools/flight_sim.py` replaces `controls.read_axes` with a virtual controller and calls the operator's methods on
  a plain object (it collects the methods of `operators.ER_OT_Fly` and its mixins); keep `read_controls()`
  calling `controls.read_axes` and `controls.map_controls` through the module (not a copied reference), so that
  replacement takes effect.

## Testing

- The add-on runs in Blender 5.2: `/Applications/Blender.app/Contents/MacOS/Blender`. Python 3.13 is Blender's own.
- **Install into a test profile, never the user's**: `tools/install_test.sh <profile dir>` builds the package and
  installs it into that profile (`BLENDER_USER_RESOURCES`). Use a new profile folder in the scratchpad.
- **The flight harness** flies a JSON plan headlessly through the real flight code with a virtual controller:
  `Blender -b <map.blend> --python tools/flight_sim.py -- <plan.json> --addon working --every 4 --out <dir>`. It writes
  `log.csv` (the drone's path every frame) and `events.txt`. The task gives you maps, a plan and baseline logs from
  the shell you are replacing: your `log.csv` files must be identical to them.
- **A real-operator check** in a UI Blender (`ui_check.py`, given with the task) runs `er.fly` for real with only
  the controller and sound faked, plus every operator above; its log must show the same lifecycle states as the
  baseline's (flight positions differ between runs: it runs in real time).
