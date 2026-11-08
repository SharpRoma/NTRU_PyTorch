#!/usr/bin/env bash
# Dedicated DFC 3.33 environment; does not install a driver or update firmware.
set -euo pipefail
expected_sha=788fddd66d1efbdcb8749b37b06f082f0eefc17dac78dab467a3c96096953155
check_only=false
resume=false
case "${1:-}" in
    --check) check_only=true; shift ;;
    --resume) resume=true; shift ;;
esac
if [ "$#" -ne 1 ]; then
    echo "Usage: [sudo] bash $0 [--check|--resume] /absolute/path/to/DFC-3.33.0.whl" >&2
    exit 2
fi
. /etc/os-release
if [ "$ID" != ubuntu ] || [ "$VERSION_ID" != 24.04 ]; then
    echo "Expected Ubuntu 24.04; no system changes made" >&2; exit 2
fi
if [ "$(uname -m)" != x86_64 ]; then echo "Expected x86_64" >&2; exit 2; fi
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
wheel=$(realpath "$1")
if [ "$(basename "$wheel")" != hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl ]; then
    echo "Unexpected DFC wheel" >&2; exit 2
fi
actual_sha=$(sha256sum "$wheel")
if [ "${actual_sha%% *}" != "$expected_sha" ]; then
    echo "DFC wheel SHA256 mismatch; no system changes made" >&2; exit 2
fi
if [ -e /opt/ntru-dfc ]; then
    if ! "$resume"; then
        echo "Refusing to overwrite /opt/ntru-dfc; use --resume for this verified installation" >&2
        exit 2
    fi
    recorded_sha=$(cut -d ' ' -f 1 /opt/ntru-dfc/compiler-wheel.sha256)
    if [ "$recorded_sha" != "$expected_sha" ]; then
        echo "Existing environment has no matching DFC provenance" >&2; exit 2
    fi
    /opt/ntru-dfc/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)'
elif "$resume"; then
    echo "No previous /opt/ntru-dfc installation to resume" >&2; exit 2
fi
if "$check_only"; then
    echo "Preflight OK: Ubuntu 24.04 x86_64, Python 3.12, verified DFC 3.33.0 wheel"
    exit 0
fi
if [ "$(id -u)" -ne 0 ]; then echo "Run via sudo (or use --check)" >&2; exit 2; fi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12-venv python3.12-dev python3-tk build-essential cmake \
    graphviz libgraphviz-dev pkg-config dkms "linux-headers-$(uname -r)"
if ! "$resume"; then python3.12 -m venv /opt/ntru-dfc; fi
exec > >(tee -a /opt/ntru-dfc/install.log) 2>&1
trap 'echo "Installation failed at line $LINENO; preserve /opt/ntru-dfc for diagnosis"' ERR
sha256sum "$wheel" > /opt/ntru-dfc/compiler-wheel.sha256
cp /etc/os-release /opt/ntru-dfc/os-release.txt
uname -a > /opt/ntru-dfc/kernel.txt
/opt/ntru-dfc/bin/python --version
export CUDA_VISIBLE_DEVICES=-1
export PIP_CACHE_DIR=/opt/ntru-dfc/pip-cache
/opt/ntru-dfc/bin/python -m pip install --upgrade pip setuptools wheel
/opt/ntru-dfc/bin/python -m pip install "$wheel"
# Imported by the SDK optimizer, but omitted from the vendor wheel metadata.
/opt/ntru-dfc/bin/python -m pip install typeguard==2.13.3
/opt/ntru-dfc/bin/python -m pip check
/opt/ntru-dfc/bin/python -m pip freeze > /opt/ntru-dfc/requirements-resolved.txt
/opt/ntru-dfc/bin/python -c \
    'from importlib.metadata import version; from hailo_sdk_client import ClientRunner; assert version("hailo-dataflow-compiler") == "3.33.0"; print("DFC import OK")'
echo "DFC ready. Hardware driver, HailoRT and firmware have not been changed."
