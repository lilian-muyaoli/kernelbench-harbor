#!/bin/bash
set -e
if [ -f /app/solution/oracle_solution.py ]; then
  cp /app/solution/oracle_solution.py /app/solution.py
  echo "oracle solution installed"
else
  echo "no oracle provided for this task" >&2
  exit 1
fi
