#!/bin/bash
set -u
LINUX=reidsurmeier@100.127.125.127
echo "[$(date)] resume claude-mem + Prompts"
rsync -az --info=progress2 --partial "$LINUX:.claude-mem/" ~/.claude-mem/
rsync -az --info=stats1 "$LINUX:Prompts/" ~/Prompts/
echo "[$(date)] DONE"
du -sh ~/.claude ~/.codex ~/.claude-mem ~/Prompts 2>/dev/null
