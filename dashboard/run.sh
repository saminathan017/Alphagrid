#!/bin/bash

DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/runtime/run_local.sh"
