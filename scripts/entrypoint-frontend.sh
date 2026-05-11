#!/bin/sh
set -e

# Locate server.js — exclude node_modules to avoid picking up Next.js internals
SERVER_JS=$(find /app -name server.js -not -path '*/node_modules/*' -maxdepth 6 | head -1)

if [ -z "$SERVER_JS" ]; then
  echo "FATAL: server.js not found in /app (excluding node_modules)" >&2
  echo "Listing /app (depth 4):" >&2
  find /app -maxdepth 4 -type f 2>/dev/null | head -60 >&2
  exit 1
fi

echo "starting: node $SERVER_JS"
exec node "$SERVER_JS" --max-http-header-size=65536
