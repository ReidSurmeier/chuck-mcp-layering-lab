#!/bin/bash
set -e
USER_NAME=reidsurmeier
if ! id "$USER_NAME" >/dev/null 2>&1; then
  useradd -m -s /bin/bash -G sudo "$USER_NAME"
  echo "Created user $USER_NAME"
else
  echo "User $USER_NAME already exists"
fi
if getent group docker >/dev/null 2>&1; then
  usermod -aG docker "$USER_NAME"
fi
echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$USER_NAME-nopasswd"
chmod 440 "/etc/sudoers.d/$USER_NAME-nopasswd"
passwd -d "$USER_NAME" >/dev/null 2>&1 || true
sed -i "s/^default=.*/default=$USER_NAME/" /etc/wsl.conf
id "$USER_NAME"
grep '^default=' /etc/wsl.conf
