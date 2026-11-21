#!/bin/zsh
# Put the Eternal Return core's wheels (~/workspaces/eternal_return_core) into the add-on's wheels, for install.sh and
# build_release.sh to bundle.
#
#   update_core_wheel.sh          rebuild the macOS wheel from the core's working tree, for Blender's Python (the
#                                 development loop: the other platforms' wheels stay as they were)
#   update_core_wheel.sh v0.1.0   download every platform's wheel from that core release, which the core's CI built
#                                 and tested on each platform (for a release of the add-on)
#
# A new core version changes the wheels' names: list them in blender_manifest.toml's wheels in place of the old ones.
# Blender has to restart to load a new one (a native module can't be reloaded).
core=~/workspaces/eternal_return_core
py=/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13
wheels=${0:A:h:h}/eternal_return/wheels

if [[ -n $1 ]]; then
  gh release download $1 --repo parsec-voyager/eternal_return_core --pattern '*.whl' --dir $wheels --clobber && \
  ls $wheels
else
  cd $core && .venv/bin/maturin build --release -i $py --out target/wheels && \
  cp target/wheels/eternal_return_core-*-cp311-abi3-macosx_11_0_arm64.whl $wheels/ && ls $wheels
fi
