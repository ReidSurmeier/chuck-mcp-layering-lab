#!/bin/bash
set -u
F=/home/reidsurmeier/.bashrc
sed -i 's|^source "/home/reidsurmeier/.openclaw/completions/openclaw.bash"|[ -f "$HOME/.openclaw/completions/openclaw.bash" ] \&\& source "$HOME/.openclaw/completions/openclaw.bash"|' "$F"
grep -n openclaw.bash "$F"
echo ===
# Ensure .profile sources .bashrc for login shells (Ubuntu standard pattern)
P=/home/reidsurmeier/.profile
if ! grep -q 'BASH_VERSION' "$P"; then
  cat >> "$P" <<'EOF'

# source .bashrc for login bash sessions (Ubuntu standard)
if [ -n "$BASH_VERSION" ]; then
  if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
  fi
fi
EOF
fi
tail -8 "$P"
