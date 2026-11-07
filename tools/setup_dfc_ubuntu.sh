#!/usr/bin/env bash
# Run inside the dedicated Ubuntu 22.04 distro, after WSL2 is available.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: sudo bash setup_dfc_ubuntu.sh /absolute/path/to/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl" >&2
    exit 2
fi
if [ "$(id -u)" -ne 0 ]; then echo "Run via sudo" >&2; exit 2; fi
. /etc/os-release
if [ "$ID" != ubuntu ] || [ "$VERSION_ID" != 22.04 ]; then echo "Expected Ubuntu 22.04" >&2; exit 2; fi
if [ "$(uname -m)" != x86_64 ]; then echo "Expected x86_64" >&2; exit 2; fi
wheel=$(realpath "$1")
if [ "$(basename "$wheel")" != hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl ]; then
    echo "Unexpected DFC wheel" >&2; exit 2
fi
if [ -e /opt/ntru-dfc ]; then echo "Refusing to overwrite /opt/ntru-dfc" >&2; exit 2; fi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv python3.10-dev build-essential graphviz libgraphviz-dev pkg-config
python3.10 -m venv /opt/ntru-dfc
/opt/ntru-dfc/bin/python -m pip install --upgrade pip setuptools wheel
/opt/ntru-dfc/bin/python -m pip install "$wheel"
/opt/ntru-dfc/bin/python -m pip check
/opt/ntru-dfc/bin/python -m pip freeze > /opt/ntru-dfc/requirements-resolved.txt
sha256sum "$wheel" > /opt/ntru-dfc/compiler-wheel.sha256
/opt/ntru-dfc/bin/python -c 'from hailo_sdk_client import ClientRunner; print("DFC import OK")'
