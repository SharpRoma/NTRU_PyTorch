# Восстановление toolchain Raspberry Pi: известный рабочий маршрут

Дата фиксации: 9 ноября 2026. Это запись реально выполненного маршрута, не предложение повторить установку на уже работающей плате. Актуальное состояние и незавершённые задачи — [CODEX_HANDOFF_2026-11-09.md](CODEX_HANDOFF_2026-11-09.md).

Все команды, если не указано иное, выполняются пользователем sharproma в SSH на Pi. Отключать питание перед монтажом PCIe/M.2/радиаторов; никаких hot-plug операций в ходе установки не выполнялось. Пароль SSH/sudo остаётся у пользователя.

## 1. Исходная система и сеть

На ПК через Raspberry Pi Imager записана Samsung microSD64GB: Raspberry Pi OS Lite Legacy 64-bit, Bookworm. Hostname ntru-pi, пользователь sharproma, SSH/password включён, Ethernet к LAN роутера. Монитор/клавиатура не требовались. Наблюдавшийся IP192.168.1.96 выдаёт DHCP, он не является постоянной настройкой.

На ПК:

```bash
ssh sharproma@ntru-pi.local
```

На Pi подтвердили Debian12, Python3.11.2, uname -m aarch64 и ядро6.12.109+rpt-rpi-2712. Проверили:

```bash
cat /etc/os-release
python3 --version
uname -r
free -h
df -h /
vcgencmd measure_temp
vcgencmd get_throttled
```

Первый free:991MiB total,803MiB available,511MiB swap (0used); root59GB,54GB свободно. Значения не являются гарантией для будущей установки/нагрузки. Вентилятор корпуса не сочетается с HAT, установлены отдельные радиаторы. Длительную нагрузку сопровождать наблюдением температуры/throttle.

Установленные инструменты:

```bash
sudo apt update
sudo apt install git build-essential cmake python3-venv python3-dev python3-pip pciutils
```

apt upgrade не выполнялся. При необходимости заголовков сначала проверить совпадение с текущим ядром:

```bash
ls -ld /lib/modules/$(uname -r)/build
dpkg -l | grep -Ei 'hailo|dkms|linux-headers'
apt-cache policy hailort hailo-dkms hailo-all dkms linux-headers-rpi-2712
```

Заголовки6.12.109 уже были установлены. Apt предлагал meta-headers6.12.115 и Hailo4.20; эти версии не ставили. Нельзя использовать mismatched headers или обновлять ядро только ради этой инструкции.

## 2. Драйвер 4.23.0 и DKMS

Модуль установлен в обесточенный M.2 адаптер, затем lspci показал Hailo1e60:2864. Одно PCI name «Hailo-8» ещё не устанавливает архитектуру8L — позднее это подтвердил identify.

Для чистой установки:

```bash
mkdir -p ~/ntru-toolchain
git clone --depth 1 --branch v4.23.0 https://github.com/hailo-ai/hailort-drivers.git ~/ntru-toolchain/hailort-drivers-4.23.0
git -C ~/ntru-toolchain/hailort-drivers-4.23.0 rev-parse HEAD
# ce1087bfe8132c99b41374e3128fc78612a3f492
```

Предварительная сборка без загрузки:

```bash
(set -o pipefail; make -C ~/ntru-toolchain/hailort-drivers-4.23.0/linux/pcie all -j1 2>&1 | tee ~/ntru-toolchain/driver-build.log)
modinfo ~/ntru-toolchain/hailort-drivers-4.23.0/linux/pcie/hailo_pci.ko
```

Требовать version4.23.0, vermagic текущего ядра и aarch64. Не продолжать по ошибке сборки.

```bash
sudo apt install --no-install-recommends dkms
sudo mkdir -p /usr/src/hailo_pci-4.23.0
git -C ~/ntru-toolchain/hailort-drivers-4.23.0 archive HEAD common linux | sudo tar -x -C /usr/src/hailo_pci-4.23.0
sed 's/@PCIE_DRIVER_VERSION@/4.23.0/g' ~/ntru-toolchain/hailort-drivers-4.23.0/linux/pcie/dkms.conf.in | sudo tee /usr/src/hailo_pci-4.23.0/dkms.conf >/dev/null
sudo dkms add -m hailo_pci -v 4.23.0
sudo dkms build -m hailo_pci -v 4.23.0 -k "$(uname -r)" -j 1
sudo dkms install -m hailo_pci -v 4.23.0 -k "$(uname -r)"
dkms status
modinfo hailo_pci | grep -E 'filename:|version:|vermagic:'
```

На текущей Pi эти действия уже выполнены. Не удалять/перерегистрировать рабочее DKMS-дерево без причины.

## 3. Firmware: файл для загрузки в RAM

```bash
mkdir -p ~/ntru-toolchain/firmware-4.23.0
cd ~/ntru-toolchain/firmware-4.23.0
wget -O hailo8_fw.4.23.0.bin https://hailo-hailort.s3.eu-west-2.amazonaws.com/Hailo8/4.23.0/FW/hailo8_fw.4.23.0.bin
```

Проверка и установка с отказом от замены другого существующего файла:

```bash
(
    set -e
    fw_src="$HOME/ntru-toolchain/firmware-4.23.0/hailo8_fw.4.23.0.bin"
    printf '%s  %s\n' \
        '1ba9528972091ec17bebc0dc7ea2e6f4449efe70664890f6387ccbc7b60626ee' \
        "$fw_src" | sha256sum -c -
    if [ -e /lib/firmware/hailo/hailo8_fw.bin ]; then
        sudo cmp "$fw_src" /lib/firmware/hailo/hailo8_fw.bin
    else
        sudo install -D -m 644 "$fw_src" /lib/firmware/hailo/hailo8_fw.bin
    fi
)
```

Никаких fw-control update/flash commands. Параметр4096 лучше задать **до первого inference**:

```bash
sudo modprobe hailo_pci force_desc_page_size=4096
```

Если модуль уже загружен с другим значением, сначала закрыть процессы Hailo, затем `sudo modprobe -r hailo_pci` и загрузить с4096. Если module in use — диагностировать удерживающий процесс, не использовать force unload.

## 4. Udev и сохранение DMA-параметра

```bash
printf '%s\n' 'SUBSYSTEM=="hailo_chardev", KERNEL=="hailo[0-9]*", OWNER="sharproma", MODE="0600"' | sudo tee /etc/udev/rules.d/51-hailo-udev.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --subsystem-match=hailo_chardev
sudo udevadm settle
ls -l /dev/hailo0
```

Для загрузок после reboot:

```bash
printf '%s\n' 'options hailo_pci force_desc_page_size=4096' | sudo tee /etc/modprobe.d/ntru-hailo.conf
sudo update-initramfs -u -k "$(uname -r)"
```

Только после успешного update-initramfs был выполнен ручной reboot. После входа проверены sysfs4096, владелец sharproma и identify. Это выполнено, не pending task.

## 5. HailoRT C++: скрытая сборка во время configure

```bash
git clone --depth 1 --branch v4.23.0 https://github.com/hailo-ai/hailort.git ~/ntru-toolchain/hailort-4.23.0
git -C ~/ntru-toolchain/hailort-4.23.0 rev-parse HEAD
# 08f088d3b443c7846af067269ce998c6d5d91449
```

Важная особенность этой версии: CMake configure не только скачивает зависимости — он уже компилирует Protobuf/protoc. В `hailort/cmake/execute_cmake.cmake` число jobs берётся из CPU count, а helper запускается в очищенной env. Один внешний CMAKE_BUILD_PARALLEL_LEVEL не решает эту вспомогательную параллельность.

На Pi локально заменена ровно одна строка:

```cmake
# Было
set(CMAKE_EXTRA_BUILD_ARGS -- -j${cores_count})
# Стало
set(CMAKE_EXTRA_BUILD_ARGS -- -j1)
```

Оригинал сохранён рядом как execute_cmake.cmake.before-j1. Это настройка сборки, не изменение вычислительных исходников HailoRT. Для новой установки сделать такую же проверенную замену до configure; не выдавать checkout за полностью чистый после этого.

```bash
(
    set -o pipefail
    cmake -S ~/ntru-toolchain/hailort-4.23.0 \
        -B ~/ntru-toolchain/hailort-4.23.0-build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DHAILO_BUILD_EXAMPLES=OFF -DHAILO_BUILD_GSTREAMER=OFF \
        -DHAILO_BUILD_SERVICE=OFF -DHAILO_BUILD_UT=OFF \
        2>&1 | tee ~/ntru-toolchain/hailort-configure-j1.log
)
```

Наблюдались GCC12.2 Protobuf warnings Wstringop-overread и CMake dev warning PUBLIC_HEADER DESTINATION. В выполненной сборке они не являлись fatal errors: Configuring done/Generating done получены. Не подавлять все warning/error и не считать подобный текст доказательством успешной сборки без итогового статуса.

```bash
(set -o pipefail; cmake --build ~/ntru-toolchain/hailort-4.23.0-build --parallel 1 2>&1 | tee ~/ntru-toolchain/hailort-build-j1.log)
sudo cmake --install ~/ntru-toolchain/hailort-4.23.0-build
sudo ldconfig
cd ~
/usr/local/bin/hailortcli --version
/usr/local/bin/hailortcli scan
/usr/local/bin/hailortcli fw-control identify
```

Ожидания:4.23.0, PCIe0001:01:00.0, HAILO8L, firmware4.23.0. Board Name Hailo-8 при Device Architecture HAILO8L нормален. Warning Cannot create log file при cwd=/ устраняется переходом `cd ~`, а не sudo-пуском проекта.

## 6. Python runtime и сеть

```bash
cd ~
python3 -m venv ~/ntru-runtime
```

Обновление pip зависало на дополнительном www.piwheels.org; оно не было условием продолжения. После Ctrl+C pip23.0.1 оставался работоспособен. PyPI поIPv4 отвечал200. Для одной команды отключали config/env индексов:

```bash
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  ~/ntru-runtime/bin/python -m pip --disable-pip-version-check install \
  --index-url https://pypi.org/simple --timeout 15 --retries 0 \
  --only-binary=:all: setuptools wheel numpy==1.26.4
```

Точные версии setuptools/wheel и полный pip freeze с Pi пока не перенесены в checkpoint; не заявлять полную побайтовую воспроизводимость всех build dependencies. Снять версии в следующем разрешённом сеансе, без их переустановки.

PyHailoRT собирали из того же checkout, в отдельный wheel:

```bash
mkdir -p ~/ntru-toolchain/pyhailort-wheels
(
    set -o pipefail
    env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
        CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 \
        ~/ntru-runtime/bin/python -m pip --disable-pip-version-check wheel \
        --verbose --no-deps --no-build-isolation \
        --index-url https://pypi.org/simple --timeout 15 --retries 0 \
        --wheel-dir ~/ntru-toolchain/pyhailort-wheels \
        ~/ntru-toolchain/hailort-4.23.0/hailort/libhailort/bindings/python/platform \
        2>&1 | tee ~/ntru-toolchain/pyhailort-build.log
)
```

Результат hailort-4.23.0-cp311-cp311-linux_aarch64.whl, SHA2565eeaf6f9462b2112902030e2f9871fddccbd907159211a8ebe90425a159b43a0. Установка с зависимостями:

```bash
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  ~/ntru-runtime/bin/python -m pip --disable-pip-version-check install \
  --index-url https://pypi.org/simple --timeout 15 --retries 0 \
  ~/ntru-toolchain/pyhailort-wheels/hailort-4.23.0-cp311-cp311-linux_aarch64.whl
~/ntru-runtime/bin/python -m pip check
```

Python `hp.Device.scan()` и `with hp.Device(id): device.control.identify()` отработали без sudo, вернули HAILO8L/4.23.0, устройство освобождено. При Python exception не менять device architecture check ради продолжения.

## 7. Проект, evidence и квалификация

На ПК свежий NumPy-only wheel создаёт tools/build_runtime_wheel.py; локальный комплект results/pi-transfer-2026-11-08-001/ntru-pi-bundle.tar.gz уже подготовлен. Он включает fixed N701 artifact **целиком**, matching experimental private key и SHA256SUMS.

На ПК (в отдельном терминале, не SSH):

```bash
scp /home/sharproma/PycharmProjects/NTRU_PyTorch/results/pi-transfer-2026-11-08-001/ntru-pi-bundle.tar.gz sharproma@ntru-pi.local:~/
```

На Pi для первой распаковки:

```bash
cd ~
tar -xzf ntru-pi-bundle.tar.gz
cd ~/ntru-pi-bundle
sha256sum -c SHA256SUMS
~/ntru-runtime/bin/python -m pip install --no-index --no-deps \
  ~/ntru-pi-bundle/wheel/ntru_pytorch-0.1.0+edge-py3-none-any.whl
~/ntru-runtime/bin/python -m pip check
```

Не распаковывать повторно поверх изменённого bundle без сравнения, не удалять numerical evidence. На PC проверенный HEF имеет те же hashes, что на Pi; перекомпиляция не выполнялась.

На Pi уже прошли:

```bash
~/ntru-runtime/bin/python ~/ntru-pi-bundle/pi_ntru_smoke.py \
  --output ~/ntru-pi-results/smoke-002.json
~/ntru-runtime/bin/python ~/pi_ntru_qualify.py --bundle ~/ntru-pi-bundle \
  --cases 1000 --output ~/ntru-pi-results/qualification-001.json
```

Файлы output намеренно не перезаписываются. Для нового запуска выбрать новый номер. qualification script на Pi был отдельно скопирован из tools/pi_ntru_qualify.py; checksum a86ca60fb3fc9a97c1841479af45d39b33b03de119c7944dff6f893a0c93b146.

Перенос оригинальных отчётов на ПК уже выполнен:

```bash
scp -r sharproma@ntru-pi.local:~/ntru-pi-results \
  /home/sharproma/PycharmProjects/NTRU_PyTorch/results/raspberry-pi-2026-11-09-001/
```

Небольшие копии этих originals включены в docs/evidence/checkpoint-2026-11-09/pi с SHA256 provenance. Результат квалификации — точность, не performance. Swap на Pi не отключался, cgroup-лимит768MiB там ещё не проверялся. ARM liboqs и полноценные CPU/NPU benchmarks остаются следующими этапами.
