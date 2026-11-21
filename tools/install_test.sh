#!/bin/zsh
# install_test.sh <profile dir>: build the add-on in this repo and install it into a test Blender profile
# (BLENDER_USER_RESOURCES=<profile dir>), never the user's own. Prints the install status and whether the core wheel
# made it into the profile.
B=/Applications/Blender.app/Contents/MacOS/Blender
repo=${0:A:h:h}
profile=${1:?usage: install_test.sh <profile dir>}
mkdir -p $profile
zip=$profile/build.zip
$B -b --factory-startup --command extension build --source-dir $repo/eternal_return --output-filepath $zip > /dev/null 2>&1 || { echo "build failed"; exit 1; }
BLENDER_USER_RESOURCES=$profile $B -b --command extension install-file -r user_default --enable $zip 2>&1 | grep STATUS
ls $profile/extensions/.local/lib/python3.13/site-packages 2>/dev/null | grep -i eternal_return_core
