## Eternal Return

Eternal Return is a Blender add-on for flying a virtual FPV drone through scenes with an RC radio or gamepad, at any scale. Changing scale continuously, in flight, is the way to move: dive into a fractal, climb out of a planet, and travel between any two points of any size in seconds, on a logarithmic path like the patterns of nature.

It began as a fork of Globglob's (Arthur Blaquart) DroneCam add-on and has since been rewritten on a flight core of its own. Every feature below can be assigned controls on your controller, and there are plenty of smaller additions as well.

### Install

Eternal Return needs Blender 5.2 or later, on macOS (Apple silicon), Windows (x64) or Linux (x64), and a game controller: an RC radio that connects over USB as a joystick (like a RadioMaster on EdgeTX), or a gamepad (Xbox and others), read through SDL. Download the zip for your platform from the [releases](https://github.com/parsec-voyager/eternal_return/releases), then in Blender use Edit > Preferences > Get Extensions > Install from Disk. The Eternal Return tab is in the 3D viewport's sidebar (N).

To build it from source, `eternal_return/install.sh` builds and installs it into Blender on macOS, and `tools/build_release.sh` builds one zip per platform into `dist/`. The core's wheels come from its own repository's releases ([eternal_return_core](https://github.com/parsec-voyager/eternal_return_core), `tools/update_core_wheel.sh <tag>`).

### Physics

The drone flies on the Eternal Return core, a compiled module bundled with the add-on, in double precision. Each flight goes on from where the last one stopped, at the same position, orientation and speed, so stopping and starting again doesn't move the view. The physics settings (mass, gravity, ground, drag, air density and camera tilt) are the scene's, in the Drone panel. Add Drone builds the drone and its camera.

Record (in the Record panel) records a flight's inputs: the sticks, and everything else the physics are given each frame, along with the Drone's own scaling, orbits and moves, keyed on the Drone. Afterwards the timeline goes back to where the flight started, and playing it replays the flight exactly. The keyed sticks (Thrust, Pitch, Roll and Yaw Rate under Drone Recording) can be edited in the Graph Editor, and the replay flies the edited sticks from the next jump of the timeline on. Bake Animation bakes the replay into an editable camera. A flight without Record keeps the last recording in the file but takes it off the Drone; assign it back (as the Drone's action) to replay it again. Record Camera, which keys the camera itself every tick, is still the way to record everything else the add-on does, like loops of an endless map.

### Scale

This allows you to change the scale of the drone as you fly. This will change the scale of the drone object itself, so all of the physics scale proportionally. In other words, if you scale up the drone to twice the size, it will travel twice as fast and have twice as much momentum before coming to a stop. You can set the rate the drone scales with the "Scale Factor" input. If Scale Factor is set to 2, the scale of the drone will change by a factor of 2 every second (double or divide in half, depending on if being scaled up or down).

### Zoom

This feature also changes the scale of the drone, but it does so relative to the origin of whatever object is selected in your scene. This means if you scale the drone up to twice the size, the drone's position will also change so that its distance from the object origin doubles. This has the effect of allowing the drone to zoom in and out of the selected object. You can set the rate with "Zoom Factor", which works the same as "Scale Factor".

On a sphere or surface map, like the Earth or the torus, the object's origin is only the map's anchor, which follows the drone, so Zoom and Orbit use the map's true center instead: the planet's core, or the mesh's origin. That's the same as a planet mesh's origin, so zooming in near the ground about the Earth's center takes you toward the core, under the surface.

### Orbit

Orbit is a feature that can be toggled on and off from the controller. When it is active, the yaw, pitch and roll inputs from the controller are used to rotate the drone around the selected object. This makes it appear as though the selected object is rotating in place in front of the drone, when in reality the object is stationary and the drone is revolving around it.

### Tidal Lock

Tidal Lock is to Orbit what Auto Scale is to Zoom. When it's on, the drone turns as it flies around the selected object, so the object always stays at the same angle in its view, the way a tidally locked moon always shows its planet the same face. Fly a quarter of the way around a planet and the drone turns a quarter turn, so the horizon stays put and you keep facing the surface. You don't need the orbit switch for this. Your yaw, pitch and roll still rotate the drone on top of it.

It has its own Mesh and Split Axes settings, separate from Auto Scale's. By default it keeps the object's origin at the same angle. With Mesh on, it keeps the closest point on the surface instead, so it follows the surface's curve and doesn't turn at all over a flat floor (but over bumpy ground it turns with every bump). With Split Axes on two axes, it keeps the line along the third axis at the same angle, so around a cylinder it turns about the cylinder's axis, and Rotation tilts those axes for a tilted cylinder. (On one axis the direction to a plane never turns, so that does nothing, and all three is the same as the origin.) On a sphere or surface map, the origin is the map's true center, as for Orbit and Zoom. Over a mesh whose many faces approximate a curve, like a torus, leave Mesh off or Tidal Lock off: the closest point's direction swings at every edge, a bumpy ride along the surface. Because the settings are separate, Auto Scale can measure to a bumpy planet's surface so you shrink near the ground, while Tidal Lock turns smoothly about the planet's center. The drone turns about the camera, so the camera doesn't move. The drone's momentum turns with it, so if you hold speed along the surface you keep circling it. Auto leveling levels to the surface. Clicking a new object, or toggling Mesh or Split Axes, starts over from there without a jump. A recording of inputs keys the turns with the Drone, as it keys Orbit's.

### Horizon

Turn on Horizon to draw a horizon line on the HUD, level with whatever is down. With an object selected, down is toward it: to its origin, or with Horizon's own Mesh on, to the closest point on its surface, and with Auto Scale's Split Axes, along the chosen axes only. Mesh is separate from Auto Scale's, because on a faceted mesh like a low-poly sphere, the direction to the closest point stays on one face's normal and then swings across the edge to the next face, so the line steps. With it off, Auto Scale can still measure to the surface while the line stays level with the center. On a sphere or surface map, the origin is the map's true center (a planet's core, a torus's middle), as for Orbit and Zoom. So over a floor whose config splits Auto Scale on Z, the line is level with the floor, and around a planet it's level with the ground below you all the way around. With nothing selected, like after Auto Deselect lets go of the click, down is the way the drone's gravity pulls (along the Drone object's Z axis, so it turns with Tidal Lock and Orbit). In zero gravity with nothing selected, the line hides until you select something. It stays true at any pitch, roll and lens. Like an aircraft's artificial horizon, it shows level, not the planet's edge: the higher you fly, the further below it the edge sits, as the surface curves away. Scaling configs can set it with "horizon", and Mesh with "horizon_mesh" under "horizon_config".

Two thinner circles, 45 degrees either side of it, work like latitude lines with the horizon as their equator: one is centered on the selected object's origin, the other on the point straight opposite it. A dot marks each of those two points, the way to the object and the way straight away from it. Looking down at a planet you see a ring around its center, and looking up, a ring around the other dot. They're drawn the same exact way as the line, so they stay true at any pitch, roll and lens, and they're drawn inside the HUD's ring too. A click goes through them to whatever is behind.

The horizon and the velocity marker are drawn like the rest of the HUD (see HUD), only while you fly.

### Velocity

Velocity (on by default) marks where the drone is actually flying, as opposed to where the camera looks: a small ring with wings and a tail, like an aircraft HUD's flight path marker, on the point of the view the drone's velocity points at. Flying straight ahead, it sits in the middle of the HUD's ring. In a turn, momentum carries the drone on the old way for a while, and the marker slides off toward it, then comes back as the flight follows the turn. Strafing or drifting, it shows which way. When the velocity points behind the camera, like flying backward, it marks the point straight opposite instead, the way you're flying away from, with an X across the ring. It shows your flight, the velocity the drone's physics move it by, so scaling, loops, wraps and Keep Drone at Origin don't move it. Orbit and Zoom move the drone too, around or toward the selected object, but they aren't flight, so the marker leaves them out, as Tidal Lock does. It hides while the drone is nearly still, or flying too far to the side to be in view. It's exact at any pitch, roll and lens, for the very frame drawn, and a click goes through it.

### Solid Fog

Solid Fog (on by default) draws a map's fog in Solid mode too, so far things fade into the fog color the way they do in the Rendered viewport, at the scene's fog_distance times the drone's scale: the fog grows and shrinks with the drone, and loops look the same. Blender's Solid mode has no fog of its own, so the add-on lays see-through layers of fog over the view at growing distances, each only over what's farther away, and draws the sky with its sun behind them. It's tuned to look like the Rendered fog, with far things going dark blue before they're gone. Maps without fog are left as they are, and so are the Earth's, whose fog is the Earth's real air.

### HUD

The UI subpanel (in the Eternal Return side panel, above Controls) has every HUD display together: Crosshairs, Ring, Gimbals, Gimbal Values, Altitude, Speed, Scale and Scale Speed (each scene saves its own), plus Horizon with its Mesh, Velocity, and Solid Fog. Gimbals shows where each stick is, in a square under the ring (throttle and yaw on the left, pitch and roll on the right); Gimbal Values adds their positions in percent, and Deadzones shades, on each stick axis, the positions the deadzone of the control mapped to that axis takes in (from the Controls settings: about the center, or for a throttle, orbit, click or brake on the whole stick, at its low end). The squares show the controller's axes 0-3 (a radio's AETR order: right stick across and up, left stick up and across), so a band follows your axis mapping: with Yaw mapped to axis 0, Yaw's deadzone is on the right stick's across.

The HUD is drawn straight into the view looking through the drone camera (draw handlers, like the sphere maps' floors), from the very view Blender draws each frame with, just past the lens clip, the same size on screen at any scale. It's not an object, so it costs the scene nothing to evaluate, and its text is drawn as text, not made into meshes every frame. Its lines are flat, light grey in Solid mode and dark in Material Preview and Rendered. Like any draw handler, it needs the viewport's overlays on.

**In Renders** shows the HUD in renders through the drone camera too: it's built as objects under the camera (Drone HUD and its text), disabled in the viewport, where the drawn HUD is, and remade before every frame. A recording keeps what the HUD showed on every frame (the sticks and the readouts), so a recorded flight renders, and plays back in the viewport, with its HUD as you flew it. Turning it on locks the interface while rendering (Render > Lock Interface), which changing objects from a frame handler needs. Bake Animation's Baked Drone Camera has no HUD.

### Readouts

While flying, Altitude, Speed, Scale and Scale Speed measure the world you fly through, not Blender's coordinates:

- **Speed** is how fast you really fly, in meters per second. Wraps, loops, Keep Drone at Origin and the endless maps' recenters move the drone in Blender's coordinates but not in the world, so they're left out, instead of showing as bursts of speed. Orbit, Zoom and gliding along a surface count, since they really move you.
- **Scale** is your true scale. A loop scales the drone by the loop factor to the same view of a finer or coarser world, so it doesn't change it: dive through a thousand loops of 10 and it reads 1.00e-1000x. Reset Scale sets it back to 1.
- **Scale Speed** is how fast your true scale changes, per second.
- **Altitude** is your height over the selected object, measured the way Auto Scale measures it (its Mesh and Split Axes settings, which each map's config sets for its ground: a floor's plane, a planet's surface). With nothing selected, it's your height over a sphere or surface map in the scene, or otherwise over the world's Z = 0.

On the Earth (a sphere map with a planet image or Google 3D tiles), every readout is in the Earth's real meters: the planet is 6,371 km in radius, so you see your real height over the ground, how fast you'd be crossing the Earth, and how many times a real drone's size you are. Elsewhere a Blender unit is a meter (times the scene's Unit Scale), relabeled by every loop. Lengths use SI prefixes (nm, µm, km, Mm) to three figures, and powers of ten past them. They're updated ten times a second, and stay as they were when you stop.

### Gravity

The Drone's Gravity (in the Drone panel, also shown in Scaling Tools) pulls the drone toward the selected object while you fly, in meters per second squared, negative pulling it in (-9.81 is Earth's gravity), the way its negative pulls down: toward the object's origin, or with Gravity's Mesh on, the closest point on its surface, along Auto Scale's Split Axes, like the horizon's down. So over a floor whose config splits Auto Scale on Z, you fall straight down to the floor, and on a sphere or surface map (the Earth included) you fall to the ground below you. Positive pushes you away. With nothing selected (like between clicks with Auto Deselect), it pulls along the Drone object's Z axis as it always has. It's in the drone's own units, like its thrust, so it handles the same at any scale. With Auto Scale on, the drone shrinks as it falls toward a surface, and falls ever slower: it falls into the surface forever without reaching it, and with loops on, into a fractal without end. Auto Brake (or the brake) adds drag, which holds a falling drone nearly still while the throttle is at zero. Scaling configs can set Mesh with "gravity_mesh" under "gravity_config". A recording keys the pull every frame, so it replays as flown.

### Click

This feature allows the user to do a mouse click in the center of the viewport with the controller, which allows you to select different objects as you fly around. This allows the zoom and orbit features to be performed on multiple objects in a single flight. There is also an input box, "Click Override", where you can input an object name, and the drone will zoom and orbit around the inputted object rather than the selected object.

The click never selects the drone itself. The Drone and its camera are skipped, and the HUD is drawn, not an object, so it picks whatever is behind them. Clicking empty sky leaves nothing selected. With Auto Deselect on, letting go of the click does the same, so nothing is selected between clicks.

### Lens Clip

The lens clip lengths of the drone camera are automatically scaled proportionally when scaling the drone, but you can also scale them up or down independently with the controller. The rate of scaling is determined by "Lens Clip Factor", which scales the lens clip lengths by the inputted amount each second, just like Zoom and Scale.

### Brake

The brake is very handy when flying in zero gravity. When the brake is active, it increases the drag of the drone smoothly and exponentially, bringing the drone to a stop. The brake will only activate when throttle is zero, because applying the throttle with high drag makes things glitchy. Enable "Auto Brake", to automatically activate the brake as soon as throttle reaches zero.

### License

Eternal Return is free software under the GNU General Public License, version 3 or later (LICENSE). The Eternal Return core it flies on is MIT licensed. The drone sounds are the add-on's own, synthesized by tools/make_sounds.py, which makes them again.
