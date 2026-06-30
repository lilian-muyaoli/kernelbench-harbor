#!/bin/bash
mkdir -p /logs/verifier
python /app/kb_verifier.py /app/reference.py /app/solution.py /app/config.json /logs/verifier/reward.json
# also emit reward.txt fallback (1/0) from reward.json
python -c "import json; print(int(json.load(open('/logs/verifier/reward.json'))['reward']))" > /logs/verifier/reward.txt 2>/dev/null || echo 0 > /logs/verifier/reward.txt
