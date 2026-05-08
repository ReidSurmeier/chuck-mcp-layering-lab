#!/bin/bash
# Run at every WSL boot via /etc/wsl.conf [boot] command=
# Idempotent: starts docker, sets Windows netsh portproxy for :8004, brings up colorsep stack
set -u
log() { echo "[colorsep-boot] $*" >&2; }

systemctl is-active docker >/dev/null 2>&1 || systemctl start docker.service
for i in 1 2 3 4 5 6 7 8 9 10; do
  docker info >/dev/null 2>&1 && break
  sleep 1
done

WSL_IP=$(hostname -I | awk '{print $1}')
log "WSL IP $WSL_IP"

NETSH=/mnt/c/Windows/System32/netsh.exe
"$NETSH" interface portproxy reset >/dev/null 2>&1 || true
"$NETSH" interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8004 connectaddress="$WSL_IP" connectport=8004 >/dev/null
log "portproxy 0.0.0.0:8004 -> $WSL_IP:8004"

cd /mnt/c/colorsep
docker compose -f docker-compose.local.yml up -d >&2
log "compose up done"
