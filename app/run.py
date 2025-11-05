import os
import platform
import subprocess
import sys

def run_command(cmd):
    print(f"\n>>> {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)

if not os.path.exists(".venv"):
    run_command(f"{sys.executable} -m venv .venv")

if platform.system() == "Windows":
    pip = ".venv\\Scripts\\pip"
    python = ".venv\\Scripts\\python"
else:
    pip = ".venv/bin/pip"
    python = ".venv/bin/python"

run_command(f"{pip} install --upgrade pip")
run_command(f"{pip} install -r requirements.txt")
run_command(f"{python} app.py")
