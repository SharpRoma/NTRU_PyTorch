# Первый запуск Raspberry Pi 5 1 ГБ

Состояние 9 ноября 2026, по журналам, присланным пользователем из SSH-сеанса Raspberry Pi. Оригинальные JSON находятся на плате в `~/ntru-pi-results` и уже скопированы пользователем на ПК в `results/raspberry-pi-2026-11-09-001/ntru-pi-results`. Их versioned копии с SHA256: [evidence/checkpoint-2026-11-09/pi](evidence/checkpoint-2026-11-09/pi). Эти данные не являются замером производительности.

- Raspberry Pi OS Lite Legacy 64-bit, Debian Bookworm 12; Python 3.11.2.
- Ядро `6.12.109+rpt-rpi-2712`, ARM64, системные страницы 16384 байта. Заголовки точно соответствуют ядру.
- Hailo PCIe `0001:01:00.0`, HAILO8L; драйвер DKMS, HailoRT и firmware 4.23.0.
- Драйвер из `ce1087bfe8132c99b41374e3128fc78612a3f492`; HailoRT из `08f088d3b443c7846af067269ce998c6d5d91449`.
- HailoRT собран Release, без service/GStreamer/examples/UT. В `hailort/cmake/execute_cmake.cmake` вспомогательная сборка Protobuf ограничена `-j1`; основная сборка и PyHailoRT также в один поток.
- PyHailoRT `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`, SHA256 `5eeaf6f9462b2112902030e2f9871fddccbd907159211a8ebe90425a159b43a0`.
- Окружение `~/ntru-runtime`, NumPy 1.26.4; pip check успешен. При загрузке пакетов piwheels был недоступен; использован PyPI с `PIP_CONFIG_FILE=/dev/null` и очисткой переменных индексов на одну команду.
- Udev `/etc/udev/rules.d/51-hailo-udev.rules`: владелец `/dev/hailo0` — `sharproma`, MODE 0600.

## Размер DMA-дескриптора

Первая попытка N701 остановилась на настройке сети SDK: `max_desc_page_size given 16384 is bigger than hw max desc page size 4096`. Численный допуск HEF до этого прошёл. Исправление — штатный параметр драйвера `force_desc_page_size=4096`; размер системных страниц, HEF, доказательство и криптографическая арифметика сохраняются.

Рабочая настройка закреплена в `/etc/modprobe.d/ntru-hailo.conf`:

```text
options hailo_pci force_desc_page_size=4096
```

Выполнен `sudo update-initramfs -u -k "$(uname -r)"`. После перезагрузки пользователь подтвердил параметр 4096 через sysfs, права доступа и HAILO8L / firmware 4.23.0 через identify. Вслед за проверкой identify после перезагрузки выполнена расширенная NTRU-квалификация, описанная ниже.

## Аппаратный smoke до перезагрузки

Комплект ПК: `results/pi-transfer-2026-11-08-001/ntru-pi-bundle.tar.gz`; на Pi распакован в `~/ntru-pi-bundle`. Содержит облегчённый wheel, весь артефакт и соответствующий существующий экспериментальный private key. Бинарные файлы x86 и DFC не переносились.

`smoke-001.json` фиксирует ошибку DMA. `smoke-002.json` — успешный реальный NTRU N701 на Hailo-8L: 19 проб, 77 кадров, три сообщения (включая Unicode/бинарное многоблочное), совпадение с независимой INT64-свёрткой и SparseNTRU, три открытия сессии и восстановление после исключения. Torch/SymPy не импортированы. Peak RSS 71778304 байта (68,5 МиБ), проверка артефакта 109769975838 нс, весь smoke 110323954236 нс. Эти времена не являются задержкой шифрования или временем NPU. Температура после smoke 63,1 °C, throttled=0x0.

HEF SHA256: `a8db2b54e0eb7f519959f808f1c0312bc820567eb5bf95e7a40cc01617b6af63`.
Канонический public key SHA256: `54e0c7e244eb1c93c4761b4b99a98e0e8123a13f17663754931650ca08042f03`.

## Расширенная проверка

Подготовлен `tools/pi_ntru_qualify.py`, который использует установленный runtime и перенесённый bundle. Его надо скопировать на плату отдельно от исходного bundle, чтобы сохранить SHA256SUMS комплекта.

```bash
~/ntru-runtime/bin/python ~/pi_ntru_qualify.py \
  --bundle ~/ntru-pi-bundle --cases 1000 \
  --output ~/ntru-pi-results/qualification-001.json
```

Использует существующий `validate_hardware`: всего 2500 проб при cases=1000, точные частичные суммы и независимая INT64-свёртка. Затем проверяет четыре вида сообщений с обоими NumPy CPU kernels (shift-sum/cached-shifts), кросс-расшифрование и восстановление сессии. Статус относится к указанной области аппаратной квалификации. Сравнение с Torch/SymPy непосредственно на ARM, сборки liboqs и бенчмарки — отдельные последующие этапы. Пользователь выполнил скрипт после перезагрузки и прислал успешный qualification-001.json; результаты приведены ниже.


## Расширенная проверка после перезагрузки: прошла

По присланному пользователем `qualification-001.json`: passed=true, hardware_tested=true, hardware_qualification=true в явно указанной области NPU/INT64/SparseNTRU/session recovery. Скрипт SHA256 `a86ca60fb3fc9a97c1841479af45d39b33b03de119c7944dff6f893a0c93b146` совпадает с `tools/pi_ntru_qualify.py` на ПК.

- Реальное HAILO8L `0001:01:00.0`, HailoRT/firmware 4.23.0.
- 2500 полиномов, включая 1000 случайных, seed=42; точные частичные суммы и независимая INT64-свёртка без расхождений.
- Пустые, Unicode, бинарные с нулями и многоблочные сообщения: точное совпадение шифротекстов и CPU-расшифрование обоими SparseNTRU kernels (shift-sum/cached-shifts).
- Три открытия устройства, восстановление после контролируемого исключения.
- page_size=16384, force_desc_page_size=4096, active16-v1, optimized host.
- Peak RSS 85540864 байта (81,6 МиБ), Torch/SymPy не импортированы. Это RSS процесса, не потребление всей платы и не cgroup-проверка 768 МиБ.
- Полное время 118858596162 нс включает численный допуск/проверку артефакта и все проверки. Не является измерением задержки шифрования или времени NPU.
- После выполнения 55,4 °C, throttled=0x0.

Оригинальные отчёты скопированы пользователем на ПК: `results/raspberry-pi-2026-11-09-001/ntru-pi-results/`; небольшие побайтовые копии включены в Git в `docs/evidence/checkpoint-2026-11-09/pi/`. ARM liboqs, Torch/SymPy на ARM и сравнение производительности ещё не выполнены.
