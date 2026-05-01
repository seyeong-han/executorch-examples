#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

python3 - <<'PY'
import json
import subprocess
import sys

required = {
    "ENABLE_USER_SCRIPT_SANDBOXING": "NO",
    "DEAD_CODE_STRIPPING": "YES",
    "ENABLE_HARDENED_RUNTIME": "YES",
    "DEVELOPMENT_TEAM": "XSDG2DKA58",
    "CODE_SIGN_STYLE": "Automatic",
}

raw = subprocess.check_output(["xcodegen", "dump", "--type", "json"], text=True)
spec = json.loads(raw)

def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)

def normalize(value):
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value)

seen = {}
for node in walk(spec):
    for key, expected in required.items():
        if key in node:
            seen.setdefault(key, set()).add(normalize(node[key]))

errors = []
for key, expected in required.items():
    values = seen.get(key, set())
    if expected not in values:
        errors.append(f"{key} expected {expected}, saw {sorted(values) or '<missing>'}")

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    sys.exit(1)

print("Project settings verified")
PY
