#!/bin/zsh
# guard.sh <most MB> <command...>: run the command, and kill it if its resident memory passes that many MB (a new
# engine's first test once walked a tree for ever and took 14 GB before the terminal was closed, 2026-11-24)
most=$1; shift
"$@" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
  if [[ -n $rss && $rss -gt $((most * 1024)) ]]; then
    echo "GUARD: killed at ${rss} KB"; kill -9 $pid; break
  fi
  sleep 0.3
done
wait $pid 2>/dev/null
