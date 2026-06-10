#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

cd "${REPO_ROOT}"

if [ -f ".env" ]; then
   set -a
   source ".env"
   set +a
fi

if [ $# -eq 0 ]; then
   echo "Usage: $0 <command> [args...]"
   exit 1
fi

exec "$@"
