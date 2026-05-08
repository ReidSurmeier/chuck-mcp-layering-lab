#!/bin/bash
# Install dev tooling on WSL2 Ubuntu. Run as user reidsurmeier.
set -u
log() { echo "==> $*"; }

log "apt update + base packages"
sudo apt-get update -qq
sudo apt-get install -yq build-essential pkg-config curl wget git tmux jq unzip rsync ca-certificates gnupg ripgrep fzf python3-pip python3-venv

log "Node.js 22 (NodeSource)"
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -yq nodejs
fi
node -v && npm -v

log "bun"
if [ ! -x ~/.bun/bin/bun ]; then
  curl -fsSL https://bun.sh/install | bash
fi
~/.bun/bin/bun --version 2>/dev/null || true

log "rustup + cargo"
if [ ! -x ~/.cargo/bin/cargo ]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
~/.cargo/bin/rustc --version 2>/dev/null || true

log "uv (fast Python)"
if [ ! -x ~/.local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
~/.local/bin/uv --version 2>/dev/null || true

log "gh CLI"
if ! command -v gh >/dev/null; then
  (type -p wget >/dev/null || sudo apt-get install -yq wget) \
    && sudo mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt-get update -qq \
    && sudo apt-get install -yq gh
fi
gh --version | head -1
gh auth status 2>&1 | head -3

log "hf CLI"
pip3 install --user --upgrade --break-system-packages "huggingface_hub[cli]" 2>/dev/null || pip3 install --user --upgrade "huggingface_hub[cli]"
~/.local/bin/hf whoami 2>/dev/null || hf whoami 2>/dev/null || echo "hf installed (may need PATH)"

log "Tailscale"
if ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

log "Claude Code CLI"
if ! command -v claude >/dev/null && [ ! -x ~/.local/bin/claude ]; then
  npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
fi
claude --version 2>/dev/null || echo "claude not on PATH yet"

log "DONE"
