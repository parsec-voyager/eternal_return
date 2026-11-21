#!/bin/zsh
# Build the add-on for release: one zip per platform in the manifest's platforms, each carrying only that platform's
# wheels, in <repo>/dist (eternal_return-<version>-<platform>.zip). Run update_core_wheel.sh <core tag> first, so
# every platform's core wheel comes from the same tested core release. install.sh stays the everyday build: one zip
# with every wheel, of which Blender installs the ones for its platform.
repo=${0:A:h:h}
bb=/Applications/Blender.app/Contents/MacOS/Blender

rm -rf $repo/dist && mkdir -p $repo/dist && \
$bb -b --factory-startup --command extension build --split-platforms --source-dir $repo/eternal_return \
  --output-dir $repo/dist && ls -l $repo/dist
