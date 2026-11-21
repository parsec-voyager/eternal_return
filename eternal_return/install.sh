#!/bin/zsh
src=${0:A:h}
out=${src:h}/eternal_return.zip
bb=/Applications/Blender.app/Contents/MacOS/Blender

# --enable also installs the wheels: a reinstall without it leaves out wheels new since the last install
$bb -b --factory-startup --command extension build --source-dir $src --output-filepath $out && \
$bb -b --command extension install-file -r user_default --enable $out
