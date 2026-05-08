#!/bin/bash
set -e
USER_NAME=reidsurmeier
USER_HOME="/home/$USER_NAME"
mkdir -p "$USER_HOME/.ssh"
cp /mnt/c/seed/id_ed25519 /mnt/c/seed/id_ed25519.pub /mnt/c/seed/known_hosts /mnt/c/seed/authorized_keys "$USER_HOME/.ssh/"
chown -R "$USER_NAME:$USER_NAME" "$USER_HOME/.ssh"
chmod 700 "$USER_HOME/.ssh"
chmod 600 "$USER_HOME/.ssh/id_ed25519" "$USER_HOME/.ssh/authorized_keys"
chmod 644 "$USER_HOME/.ssh/id_ed25519.pub" "$USER_HOME/.ssh/known_hosts"
ls -la "$USER_HOME/.ssh"
sudo -u "$USER_NAME" ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 reidsurmeier@100.127.125.127 "echo SSH_TO_LINUX_OK; hostname"
