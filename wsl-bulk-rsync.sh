#!/bin/bash
# Run as user reidsurmeier inside WSL2 Ubuntu on Windows.
# Pulls config + Claude Code state + ebooks from Linux box (100.127.125.127).
set -u
LINUX=reidsurmeier@100.127.125.127
LOG=/tmp/wsl-bulk-rsync.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] START"

cd ~

echo "==> Phase 1: secrets + auth state"
rsync -az --info=stats1 "$LINUX:.gnupg/" ~/.gnupg/ || true
rsync -az --info=stats1 "$LINUX:.secrets/" ~/.secrets/ || true
rsync -az --info=stats1 \
  "$LINUX:.netrc" "$LINUX:.env.migration" \
  ~/ || true
chmod 700 ~/.gnupg ~/.secrets 2>/dev/null
chmod 600 ~/.netrc ~/.env.migration 2>/dev/null

echo "==> Phase 2: shell + dotfiles"
rsync -az --info=stats1 \
  "$LINUX:.bashrc" "$LINUX:.profile" "$LINUX:.tmux.conf" \
  "$LINUX:.gitconfig" "$LINUX:.npmrc" \
  "$LINUX:.mcp.json" "$LINUX:.claude.json" \
  ~/ || true
mkdir -p ~/.docker
rsync -az "$LINUX:.docker/config.json" ~/.docker/ || true

echo "==> Phase 3: Claude Code state (excluding regeneratables)"
mkdir -p ~/.claude
rsync -az --info=stats1 \
  --exclude='file-history/' --exclude='session-env/' --exclude='bash-log.txt' \
  --exclude='cache/' --exclude='telemetry/' --exclude='paste-cache/' --exclude='_logs/' \
  "$LINUX:.claude/" ~/.claude/ || true

echo "==> Phase 4: codex"
rsync -az --info=stats1 "$LINUX:.codex/" ~/.codex/ || true

echo "==> Phase 5: gh CLI auth"
mkdir -p ~/.config/gh
rsync -az "$LINUX:.config/gh/" ~/.config/gh/ || true

echo "==> Phase 6: huggingface token"
mkdir -p ~/.cache/huggingface
rsync -az "$LINUX:.cache/huggingface/token" ~/.cache/huggingface/ 2>/dev/null || true

echo "==> Phase 7: claude-mem corpora (481MB)"
rsync -az --info=progress2 "$LINUX:.claude-mem/" ~/.claude-mem/ || true

echo "==> Phase 8: openclaw skills + plugins (small)"
mkdir -p ~/.openclaw
rsync -az --info=stats1 \
  --exclude='workspace/' --exclude='plugin-runtime-deps/' --exclude='memory/' \
  "$LINUX:.openclaw/" ~/.openclaw/ || true

echo "==> Phase 9: Prompts (workflow plans)"
rsync -az --info=stats1 "$LINUX:Prompts/" ~/Prompts/ || true

echo "==> Phase 10: Epub + PDF library (28GB) — this takes time"
rsync -az --info=progress2 "$LINUX:'Epub + PDF/'" ~/Epub-PDF/ || true

echo "[$(date)] DONE"
df -h ~ | tail -1
du -sh ~/.claude ~/.codex ~/.claude-mem ~/.openclaw ~/Prompts ~/Epub-PDF 2>/dev/null
