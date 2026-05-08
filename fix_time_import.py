#!/usr/bin/env python3
"""Add missing 'import time' to separate_v20.py."""
with open('/app/separate_v20.py', 'r') as f:
    code = f.read()
code = code.replace('import argparse\n', 'import argparse\nimport time\n', 1)
with open('/app/separate_v20.py', 'w') as f:
    f.write(code)
print("Added 'import time' to separate_v20.py")
