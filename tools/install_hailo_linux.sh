#!/usr/bin/env bash
# Install the locally built, pinned HailoRT and DKMS driver. Never flashes a board.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
driver="$root/results/toolchain/hailort-drivers-4.23.0"
runtime="$root/results/toolchain/hailort-4.23.0"
build="$root/results/toolchain/hailort-4.23.0-build"
firmware="$root/results/toolchain/hailort-4.23.0-firmware/hailo8_fw.4.23.0.bin"
fw_sha=1ba9528972091ec17bebc0dc7ea2e6f4449efe70664890f6387ccbc7b60626ee
. /etc/os-release
[ "$ID" = ubuntu ] && [ "$VERSION_ID" = 24.04 ] && [ "$(uname -m)" = x86_64 ]
[ "$(git -c safe.directory="$driver" -C "$driver" rev-parse HEAD)" = ce1087bfe8132c99b41374e3128fc78612a3f492 ]
[ "$(git -c safe.directory="$runtime" -C "$runtime" rev-parse HEAD)" = 08f088d3b443c7846af067269ce998c6d5d91449 ]
git -c safe.directory="$driver" -C "$driver" diff --quiet
git -c safe.directory="$runtime" -C "$runtime" diff --quiet
actual=$(sha256sum "$firmware")
[ "${actual%% *}" = "$fw_sha" ]
[ "$(modinfo -F version "$driver/linux/pcie/hailo_pci.ko")" = 4.23.0 ]
vermagic=$(modinfo -F vermagic "$driver/linux/pcie/hailo_pci.ko")
[ "${vermagic%% *}" = "$(uname -r)" ]
[ -f "$build/hailort/libhailort/src/libhailort.so.4.23.0" ]
if [ -e /lib/firmware/hailo/hailo8_fw.bin ]; then
    installed=$(sha256sum /lib/firmware/hailo/hailo8_fw.bin)
    if [ "${installed%% *}" != "$fw_sha" ]; then
        echo "Existing firmware file differs: refusing to replace it" >&2; exit 2
    fi
fi
if [ -e /usr/src/hailo_pci-4.23.0 ]; then
    echo "Existing DKMS source found: inspect it before retrying" >&2; exit 2
fi
if [ "$(id -u)" -ne 0 ]; then echo "Preflight OK. Run via sudo to install." >&2; exit 2; fi
operator_uid=${SUDO_UID:-}
if [[ ! "$operator_uid" =~ ^[0-9]+$ ]] || [ "$operator_uid" = 0 ]; then
    echo "Run through sudo as the user who will access Hailo" >&2; exit 2
fi
getent passwd "$operator_uid" >/dev/null
exec > >(tee "$root/results/toolchain/hailort-4.23.0-system-install.log") 2>&1
cmake --install "$build" --prefix /usr/local
ldconfig
# Version-matched boot firmware is loaded to device RAM by the PCIe driver.
# No fw-control update, flash write, reboot or Secure Boot change is performed.
install -d /lib/firmware/hailo
install -m 644 "$firmware" /lib/firmware/hailo/hailo8_fw.bin
# Restrict the device to the invoking user rather than upstream's world-writable rule.
printf 'SUBSYSTEM=="hailo_chardev", KERNEL=="hailo[0-9]*", OWNER="%s", MODE="0600"\n' \
    "$operator_uid" > /etc/udev/rules.d/51-hailo-udev.rules
if mokutil --sb-state | grep -q 'SecureBoot enabled'; then
    SHIM_NOTRIGGER=y update-secureboot-policy --new-key
fi
install -d /usr/src/hailo_pci-4.23.0
git -c safe.directory="$driver" -C "$driver" archive HEAD common linux | tar -x -C /usr/src/hailo_pci-4.23.0
sed 's/@PCIE_DRIVER_VERSION@/4.23.0/g' \
    "$driver/linux/pcie/dkms.conf.in" > /usr/src/hailo_pci-4.23.0/dkms.conf
dkms add -m hailo_pci -v 4.23.0
dkms build -m hailo_pci -v 4.23.0 -k "$(uname -r)"
dkms install -m hailo_pci -v 4.23.0 -k "$(uname -r)"
udevadm control --reload-rules
modinfo hailo_pci
if mokutil --sb-state | grep -q 'SecureBoot enabled'; then
    [ -n "$(modinfo -F signer hailo_pci)" ]
    # Some mokutil versions return 0 even for "is not enrolled". Require the
    # affirmative C-locale message; an unknown result must never load the module.
    enrollment=$(LC_ALL=C mokutil --test-key /var/lib/shim-signed/mok/MOK.der 2>&1 || true)
    echo "$enrollment"
    if ! printf '%s\n' "$enrollment" | grep -q ' is already enrolled$'; then
        echo "Driver installed and signed. MOK enrollment is required before loading."
        echo "Run: sudo mokutil --import /var/lib/shim-signed/mok/MOK.der"
        echo "Choose a one-time password in your terminal; enroll the key in MOK Manager on your next reboot."
        exit 0
    fi
fi
modprobe hailo_pci
udevadm settle
/usr/local/bin/hailortcli scan
/usr/local/bin/hailortcli fw-control identify
