#!/bin/zsh
src=${0:A:h}
out=${src:h}/eternal_return.zip
bb=/Applications/Blender.app/Contents/MacOS/Blender

# The add-on was DroneCam's fork, installed as "dronecam", until it became Eternal Return (2026-11-20). Both keep their
# settings under the same scene attributes, so the old one goes before the new one is enabled.
if $bb -b --factory-startup --command extension list 2>/dev/null | grep -q "^  dronecam \[installed\]"; then
  $bb -b --command extension remove user_default.dronecam
fi

# --enable also installs the wheels: a reinstall without it leaves out wheels new since the last install
$bb -b --factory-startup --command extension build --source-dir $src --output-filepath $out && \
$bb -b --command extension install-file -r user_default --enable $out
