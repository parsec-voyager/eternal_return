#!/bin/zsh
# guard.sh <most MB> <command...>: run the command, and kill it if its resident memory passes that many MB (a new
# engine's first test once walked a tree for ever and took 14 GB before the terminal was closed), or if it has run for
# GUARD_SECONDS (7200 unless set; the slowest seam flight takes about an hour). A flight that's stuck never ends, and
# nothing says so: a build that failed and was asked for again each time flew one flight for 2.5 hours. The seconds
# are those awake: a Mac that slept through the night hasn't run the flight any longer.
zmodload zsh/datetime
most=$1; shift
limit=${GUARD_SECONDS:-7200}
"$@" &
pid=$!
typeset -F ran=0 last=$EPOCHREALTIME now
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
  if [[ -n $rss && $rss -gt $((most * 1024)) ]]; then
    echo "GUARD: killed at ${rss} KB"; kill -9 $pid; break
  fi
  now=$EPOCHREALTIME
  (( ran += (now - last < 2 ? now - last : 0), last = now ))
  if (( ran > limit )); then
    echo "GUARD: killed after ${limit} s (stuck?)"; kill -9 $pid; break
  fi
  sleep 0.3
done
wait $pid 2>/dev/null
