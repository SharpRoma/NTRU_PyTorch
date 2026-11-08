# Передача проекта NTRU_PyTorch другому чату Codex

**Актуальный checkpoint: 9 ноября 2026, Europe/Moscow. Начинать продолжение с этого файла.** Пользователь остановил вычислительную работу для оформления документации и индекса Git. Ниже разделены выполненные этапы, исторические измерения и будущие задачи. Пароли и закрытые ключи здесь не хранятся.

## 1. Назначение проекта и договорённости

Исследуется точное NTRUEncrypt-style шифрование и перенос фиксированного произведения `r·h mod (x^N−1,q)` на Hailo-8L. Проект должен оставаться полезным на Ubuntu ПК для экспериментов/РИНЦ; перспективная работа ВАК К1–К2 — сравнение CPU и того же NPU на Raspberry Pi 5 с 1 ГБ RAM. Публикационный статус, безопасность параметров и ускорение на Pi не заявлены.

Исходная схема проекта не равна полному NTRU-HRSS KEM. Названия preset исторические, не уровни криптостойкости. Существующие ключи, шифротексты, byte codec, SymPy/PyTorch API и проверенные HEF сохраняются. Смена публичного ключа требует своего HEF и нового допуска: текущий HEF не универсален.

Основные договорённости:

- Только реальный запуск NTRU на NPU подтверждает аппаратную работоспособность. SoftwareHailoEncryptor, ONNX/DFC emulation, компиляция и ResNet этого не подтверждают.
- Точное равенство, без допуска в ciphertext/частичных суммах. Для affine-профиля доказанная ошибка строго меньше 1/2; неизвестные SDK-семантики отклоняются. Не ослаблять контракт ради прохождения тестов.
- Сохранять прошлые каталоги и не перезаписывать отчёты. Компиляцию и измерения выполнять последовательно, без параллельных тестов/генерации ключей. Явно разделять полное encrypt_bytes, фазы, старт, compilation, C microbenchmark и KEM.
- Энергопотребление не измерялось. SDK+PCIe не является внутренним временем NPU. 119 секунд квалификации на Pi не являются временем шифрования.
- Secure Boot и TPM на ПК намеренно отключены пользователем. Не возвращаться к их настройке. Ядро и firmware не обновлять автоматически.
- Пользователь впервые работает с Raspberry Pi, но знаком с STM32/ESP. Давать небольшие последовательные шаги, явно писать «на ПК»/«на малине». Подключение без экрана, через Ethernet и SSH. Пароль вводит пользователь, он не передавался агенту.

## 2. Что подтверждено к checkpoint

| Этап | Состояние |
|---|---|
| DFC 3.33.0 на Ubuntu 24.04/Python 3.12 | установлен, ClientRunner импортируется, pip check; компиляция на CPU |
| HailoRT/драйвер/firmware ПК | 4.23.0; исторические аппаратные эксперименты N17/N107/N701 прошли |
| N=17,107,701 на ПК | affine HEF, численный допуск и реальные partial sums/ciphertexts/cross-decryption; optimized host также квалифицирован |
| active16-v1 и LUT cache | реализованы и проверены, прежняя упаковка сохранена |
| host_processing optimized | реализован, точен, повторяемое ускорение против reference на ПК |
| SparseNTRU | два точных NumPy ядра, проверены на ПК; использованы на Pi для сверки/расшифрования |
| liboqs на x86_64 | pinned portable clean/native AVX2, арифметика и отдельный HRSS KEM, KAT/1000-case/interop/sanitizer/бенчмарки прошли |
| Runtime с ограничением памяти на ПК | отдельный NumPy wheel, 768 МиБ cgroup, swap=0, реальный Hailo 128 блоков; не эмуляция всей Pi |
| Raspberry Pi Hailo-8L | установлен и работает, N701 smoke и расширенная квалификация после reboot прошли |
| liboqs на ARM | **ещё не собирался и не проверялся** |
| PyTorch/SymPy на Pi | **ещё не установлены/не проверены** |
| CPU/NPU performance на Pi | **ещё не измерено** |

Исходные отчёты малины уже скопированы пользователем на ПК. Их побайтовые копии в Git: [evidence/checkpoint-2026-11-09/pi](evidence/checkpoint-2026-11-09/pi). Это оригинальные файлы выполнения, а не вручную перепечатанные JSON. Ограничение qualification_scope в qualification-001 обязательно учитывать: NPU, независимая INT64-свёртка, оба SparseNTRU kernels, session recovery. Это не подтверждение Torch/SymPy/liboqs на ARM.

## 3. Сравнение с последним коммитом

База сравнения — `14c24e7de56669295cdf71b6d858886edad5f101` от 7 ноября 2026: `feat(hailo): добавить экспериментальный конвейер NTRU для Hailo-8L`. Он уже содержал ONNX, software-модель, заготовку DFC worker и контракты допуска. Текущий checkpoint переводит этот каркас в реально проверенный конвейер и добавляет следующие этапы:

| Область | Изменения после 14c24e7 |
|---|---|
| Ubuntu toolchain | установка DFC под 24.04/3.12 с SHA256/resume, CMake/typeguard, Linux runtime/DKMS installer |
| DFC extraction | настоящий узкий SDK-экстрактор, полная calibration, native outputs/HAR/HN/traceback, effective weights/bias и rounding/accumulator evidence |
| Допуск | реальные 16/32-bit accumulators и pre-rounding bounds, проверка metadata самого HEF через SDK; unit-контракт не ослаблен |
| Упаковка | active16-v1, owners/signs, нулевой аппаратный кадр, прежний position16-v1 |
| Host pipeline | точная LUT cache/compact UINT8 LUT, grouped INT64 restoration, preallocated packing, batch reduction и NumPy byte codec |
| Измерения | новые CLI flags, старт/фазы/payload bytes/s, варианты и Sparse comparison; независимые worker-процессы OQS-эксперимента |
| Runtime memory | lazy imports, отдельный edge wheel, потоковый SHA256 и построчное чтение numerical evidence без полной строковой матрицы |
| CPU baseline | SparseNTRU shift-sum/cached-shifts, prepare(key), ограничения INT64/памяти и тесты |
| liboqs | два C-моста, ctypes, pinned builds, проверка ELF/hash/ABI/SONAME, qualification/interop/KEM и arithmetic benchmarks |
| Pi | standalone smoke/qualification tools, реальный перенос, диагностика DMA 16K→4K, сохранённые отчёты |
| Документация | журналы этапов и настоящий handoff с versioned небольшими свидетельствами |

Основная модульная структура и форматы v1 сохранены. Следующий коммит пользователя будет новым HEAD; 14c24e7 останется исторической базой, не инструкцией откатить рабочее дерево. Подготовленный conventional commit — [COMMIT_MESSAGE_2026-11-09.txt](COMMIT_MESSAGE_2026-11-09.txt). Агент коммит не создавал.

## 4. Два компьютера и окружения

### ПК Ubuntu

Корень: `/home/sharproma/PycharmProjects/NTRU_PyTorch`. Ubuntu 24.04 x86_64, kernel исторических экспериментов `7.0.0-42-generic`; AMD Ryzen 5 5600, около 16 ГБ RAM, RTX 3060 в этих экспериментах не используется. Основная `.venv`: Python 3.12.3, NumPy 1.26.4, PyTorch 2.15.0, SymPy 1.14.0, HailoRT 4.23.0. DFC — отдельный `/opt/ntru-dfc`, 3.33.0, CPU. Разрешённые зависимости/журналы локально в results/toolchain и results/ubuntu-implementation-2026-11-07.

**Hailo-модуль теперь физически находится на Raspberry Pi. Не предполагать наличие NPU на ПК и не запускать hardware tests там без нового scan.** Исторические PC hardware reports остаются свидетельствами именно тех запусков.

### Raspberry Pi

- Raspberry Pi 5, 1 ГБ RAM, Samsung microSD 64 ГБ, Raspberry Pi OS Lite Legacy 64-bit (Debian 12 Bookworm), Python 3.11.2.
- Kernel `6.12.109+rpt-rpi-2712`, AArch64, glibc 2.36, **системные страницы 16384 байта**.
- Ethernet к роутеру; SSH с ПК: `ssh sharproma@ntru-pi.local`; наблюдавшийся DHCP IP `192.168.1.96` может измениться.
- Питание 5,1 В / 5 А. Наклеены отдельные радиаторы. Вентилятор корпуса не помещается с PCIe–M.2 адаптером; работа пока пассивная. Во всех присланных проверках throttled=0x0; после квалификации 55,4 °C. Под будущей длительной нагрузкой проверить заново.
- Swap при первом запуске 511 МиБ, не отключался. Нельзя писать «Pi протестирована без swap» на основании PC cgroup-эксперимента.
- PCIe Hailo `0001:01:00.0`, устройство `/dev/hailo0`, HAILO8L; driver/runtime/PyHailoRT/firmware 4.23.0.
- Python env `~/ntru-runtime`; NumPy 1.26.4, PyHailoRT cp311/aarch64. Torch/SymPy отсутствуют. Проект установлен из edge wheel с --no-index --no-deps.
- `~/ntru-pi-bundle`: artifact/, private.json, wheel/, SHA256SUMS, pi_ntru_smoke.py. Ключ экспериментальный и соответствует HEF.
- `~/pi_ntru_qualify.py`: отдельный скрипт, вне исходного bundle, чтобы не менять SHA256SUMS комплекта.
- `~/ntru-pi-results`: smoke-001.json (неудача DMA), smoke-002.json и qualification-001.json (успех).
- `~/ntru-toolchain`: исходники driver/runtime, build dirs и логи, firmware, pyhailort-wheels/.
- HailoRT установлен в `/usr/local`; не через пакет hailort из apt. Apt Bookworm предлагал только 4.20.0. `hailo-all` не устанавливался.

Агент работал на ПК; команды Pi выполнял пользователь и присылал вывод. Не предполагать настроенный SSH-ключ или возможность безпарольного автоматического SSH.

## 5. Критичная настройка Raspberry Pi: DMA 4096

Драйвер по умолчанию взял PAGE_SIZE=16384; первый N701 configure завершился:

```text
max_desc_page_size given 16384 is bigger than hw max desc page size 4096
HAILO_INTERNAL_FAILURE(8)
```

Численный допуск к этому моменту прошёл. Исправление — штатный module parameter, не изменение системного PAGE_SIZE/HEF/арифметики:

```text
/etc/modprobe.d/ntru-hailo.conf:
options hailo_pci force_desc_page_size=4096
```

Выполнен `sudo update-initramfs -u -k "$(uname -r)"`, reboot и подтверждение sysfs/identify. После reboot выполнена расширенная квалификация. Контроль на Pi:

```bash
cat /sys/module/hailo_pci/parameters/force_desc_page_size   # 4096
ls -l /dev/hailo0                                        # owner sharproma, mode 0600
/usr/local/bin/hailortcli --version                       # 4.23.0
/usr/local/bin/hailortcli fw-control identify              # HAILO8L / 4.23.0
```

Udev `/etc/udev/rules.d/51-hailo-udev.rules`:

```text
SUBSYSTEM=="hailo_chardev", KERNEL=="hailo[0-9]*", OWNER="sharproma", MODE="0600"
```

Не заменять права на 0666 и не запускать проект через sudo для обхода доступа. Порядок установки и устранённые ошибки: [pi-first-run.md](pi-first-run.md), полное восстановление toolchain: [pi-toolchain-runbook.md](pi-toolchain-runbook.md).

## 6. Закреплённые версии и артефакты

| Компонент | Версия / происхождение |
|---|---|
| DFC | 3.33.0 x86_64; SHA256 wheel `788fddd66d1efbdcb8749b37b06f082f0eefc17dac78dab467a3c96096953155` |
| Hailo driver | v4.23.0, `ce1087bfe8132c99b41374e3128fc78612a3f492` |
| HailoRT | v4.23.0, `08f088d3b443c7846af067269ce998c6d5d91449` |
| Hailo8 boot firmware | `hailo8_fw.4.23.0.bin`, SHA256 `1ba9528972091ec17bebc0dc7ea2e6f4449efe70664890f6387ccbc7b60626ee` |
| PyHailoRT на Pi | cp311/aarch64, SHA256 wheel `5eeaf6f9462b2112902030e2f9871fddccbd907159211a8ebe90425a159b43a0` |
| liboqs | 0.16.0, `5a1a854b0dc9f2141bdc771c555ee60c37950183` |

Firmware-файл драйвер загружает в RAM ускорителя при probe; команды flash update не выполнялись. DFC не нужен на Pi: используется готовый HEF.

Проверенные PC artifact каталоги относительно корня проекта:

| N | p/q/d | Cin/Cout | Artifact | Private key (локальный, вне Git) |
|---:|---|---|---|---|
| 17 | 3/256/1 | 32/144 | results/admitted-n17-ubuntu-affine-003 | results/compiler-experiments/n17/private.json |
| 107 | 3/64/5 | 112/656 | results/admitted-n107-ubuntu-affine-001 | results/compiler-experiments/n107/private.json |
| 701 | 3/8192/55 | 704/9120 | results/admitted-n701-ubuntu-affine-001 | results/compiler-experiments/n701-active16-2026-11-08/private.json |

HEF SHA256:

```text
N17  8a00dd51f08a47854833947070c927f0c41cf7476d1be22a96daca0897837ed6
N107 2cf4e9b28f32600ecdd8dfa516aa50275d2588357860a929a31b2d774af198b2
N701 a8db2b54e0eb7f519959f808f1c0312bc820567eb5bf95e7a40cc01617b6af63
```

N701 canonical public key SHA256 `54e0c7e244eb1c93c4761b4b99a98e0e8123a13f17663754931650ca08042f03`; private JSON file SHA256 `cf9006df32d5d76b0b4416a792266998f5b3ac8f91af22053c713e90e8a93920`. Это разные виды хеша, не путать file SHA с canonical key hash.

Artifact directory содержит manifest.json, public.json, model.onnx, weights.npy, vectors.npz, model.hef, numeric-evidence.npz, compiler-report.json. Нельзя переносить только HEF или исключать «большой NPZ»: все эти файлы связаны проверками manifest. N701 evidence занимает около 515 МиБ на диске, но читается построчно, не целиком в RAM.

PC→Pi bundle: `results/pi-transfer-2026-11-08-001/ntru-pi-bundle.tar.gz`, SHA256 `a6a65f110c2ec1199a1403c180fb5cb1b96e2e4783d49417076407cac7164529`, около 13 МиБ gzip / 560 МиБ распаковано. Он содержит экспериментальный private key — **не добавлять архив в Git**.

## 7. Численная модель и пути исполнения

Hailo ускоряет произведение с фиксированным h. Канал l*N+k кодирует бит l циклически сдвинутого h; padding каналов кратен 16. На NPU идёт одиночный бинарный Conv1x1, кадр имеет не более 16 единиц, ожидаемые partial sums 0–16. На CPU остаются sampling/codec, упаковка, affine encode/decode, восстановление знака и битовых плоскостей в INT64, modulo q, сообщение, центрирование, Ciphertext и вся расшифровка.

`unit-v1` не прошёл строгий unit numerical contract, даже при нулевой sample-error эмулятора; сохраняется compiled-unverified. Рабочий отдельный `binary-circulant-affine-v1` выводится из реальных MAC/APU SDK-параметров. Точная рациональная LUT восстанавливает целые суммы. Проверка bounds учитывает 16/32-bit accumulator, clipping, rounding и saturation. Доверие к экстрактору/семантике SDK остаётся явной границей доказательства, это не аппаратная аттестация.

`hailo-admit` проверяет также реальную HEF metadata (порты/scales/zero points/dtype), но не открывает NPU. Только qualification после этого даёт hardware evidence.

Совместимые значения по умолчанию сохранены:

```text
packing="position16-v1"
host_processing="reference"
cache_affine_lut=True
```

Для текущих опытов явно выбирать active16-v1 + optimized + cached LUT. active16-v1 группирует положительные и отрицательные ненулевые позиции отдельно, максимум по 16, сохраняя исходные каналы/owners/signs. Нулевой полином отправляет один нулевой кадр. Штатный r N107 (5+5) — 2 кадра вместо 14; N701 (55+55) — 8 кадров вместо 88. Для произвольного ternary r число кадров иное.

N701 на полином: 5632 входных и 72960 выходных UINT8 байт. На 128 полиномов: 1024 кадра, вход 720896, выход 9338880 байт. Это VStreams payload, не все байты на физическом PCIe.

Optimized host не меняет HEF/доказательство: компактная LUT 256×UINT8 с invalid marker255, grouped INT64 restoration без расширения всех кадров, fallback для переставленных owners, preallocated frames, NumPy unpackbits big-endian и batch reduction. Публичные проверки и Ciphertext constructor не обходятся. Вложенные таймеры не суммировать повторно; unaccounted_ns — остаток непересекающихся верхних фаз.

## 8. CPU-ориентиры и liboqs

SparseNTRU — NumPy-only точная тернарная арифметика, shift-sum O(N) prepared memory и cached-shifts O(N²). Для N701 подготовленные массивы соответственно 16824 и 3936816 байт. prepare(key) кэширует один immutable key, проверяет INT64 bound/лимит памяти; новый ключ заменяет кэш. Это не обещание constant-time или оптимального CPU ceiling.

liboqs имеет **два независимых уровня**:

1. `OqsArithmeticEncryptor`: неизменённый PQClean NTRU-HRSS701 `poly_Rq_mul`, циклическое произведение modulo x^701−1 и8192, затем наш codec/message/centering/Ciphertext. Только N701,p3,q8192; тернарный r, -1→8191, uint16 canonical buffers. Для AVX2 padding704/alignment32, batch — один FFI call. Не использовать poly_Sq_mul: Phi-редукция меняет задачу. Raw unreduced trace отклоняется. Это encryption-only adapter; decrypt выполняют существующие CPU engines.
2. `OqsKem`: публичный NTRU-HRSS-701 KEM; bytes keys/ciphertext/shared secret, C-owned struct layout, ctypes. Keypair/encapsulate/decapsulate — отдельные замеры, не коэффициент ускорения нашего encrypt_bytes.

Source checkout и profile directories: `results/native/oqs/0.16.0/source`, затем `{x86_64,aarch64}/{portable,native}`. Portable — clean C. Native x86 требует AVX2/BMI2; native ARM также clean, но compiler target Cortex-A76. Отдельного HRSS NEON kernel не заявлено. Build Release/shared, only HRSS701/common infrastructure, OpenSSL/GPU OFF, без глобальной установки/ldconfig и без изменения desktop venv. На Pi --jobs1.

Python импорт oqs ничего не скачивает/не собирает/не загружает. Явный verified manifest, SHA/ELF/version/ABI checks. Один KEM profile на процесс из-за SONAME; close не делает смену профиля допустимой. Arithmetic bridge отдельно компилирует upstream Rq source, не зависит от экспорта private symbols liboqs. При несовпадении provenance builder отказывается перезаписывать сборку.

KAT single/all сравниваются с закреплёнными upstream SHA256, детерминированный RNG только в отдельном kat_kem subprocess. Обычный KEM использует системный RNG liboqs, seed42 им не управляет. Неверные длины отклоняются до FFI; повреждённый ciphertext может успешно возвращать другой secret (implicit rejection). Нельзя требовать обязательного исключения.

## 9. Измерения: что можно цитировать

### Последнее прямое сравнение ПК

`results/oqs-integration-2026-11-08-001/arithmetic/`: общий сохранённый корпус, seed42, разные messages/r на каждое измерение, batch1/8/32/128, 3 warmups, 20 samples, 3 отдельных worker-процесса каждого backend, изменяемый порядок. Torch6 threads. Каждый measured ciphertext сверяется вне таймера с независимым INT64 oracle; SymPy/Torch encryption и cross-decryption проверены до timing для каждого batch. Hailo active16/optimized/LUT, повторные аппаратные пробы. Основной таймер — полный encrypt_bytes, key/r generation и IO снаружи.

Медианы объединённых 60 измерений на ячейку, мс (by-run median/p95 сохранены отдельно):

| Блоков | Torch | Sparse shift | Sparse cached | OQS clean | OQS AVX2 | Hailo |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.051 | 0.533 | 0.466 | 0.431 | 0.414 | 1.501 |
| 8 | 23.869 | 3.905 | 3.314 | 3.083 | 2.931 | 7.097 |
| 32 | 95.502 | 15.410 | 13.045 | 12.098 | 11.534 | 25.444 |
| 128 | 380.574 | 61.650 | 51.810 | 48.335 | 45.869 | 102.322 |

На 128 Hailo быстрее Torch в 3.72×, **OQS AVX2 быстрее Hailo в 2.23×**. На всех четырёх проверенных партиях оба OQS arithmetic профиля быстрее Hailo на этом ПК. Переносить вывод на ARM нельзя. Нет доверительных интервалов/утверждения оптимального CPU.

Отдельный C Rq microbenchmark на128: clean2.6637 мс, AVX2 0.3010 мс; boundary включает native multiplication, canonical stores и checksum, исключает Python/input preparation. Он не является полным encrypt_bytes. Версионируемые сводки: [OQS arithmetic](evidence/checkpoint-2026-11-09/pc/oqs-arithmetic-summary.json).

### Отдельный HRSS KEM на ПК

3 processes/profile, 3 warmups и20 windows≥100ms для каждой операции и API; KAT inputs, штатная случайность. Медианы, мкс/op:

| Профиль/API | keypair | encapsulate | decapsulate |
|---|---:|---:|---:|
| clean C | 969.72 | 33.49 | 76.39 |
| clean Python | 971.17 | 36.36 | 79.83 |
| AVX2 C | 57.98 | 10.76 | 12.35 |
| AVX2 Python | 61.06 | 13.67 | 15.77 |

Не делить эти значения на Hailo encrypt_bytes и не нормировать KEM по байтам нашего сообщения. [KEM summary](evidence/checkpoint-2026-11-09/pc/oqs-kem-summary.json).

### Исторические этапы

| Этап/каталог | Содержание и границы |
|---|---|
| results/optimization-2026-11-08 | N107 position/active × LUT cached/uncached; controlled run с6 threads, selected active+LUT; на128 30.011ms Hailo против42.123ms Torch. До host optimized. |
| results/n701-benchmark-2026-11-08-001 | N701 reference host: на128 Hailo265.695ms противTorch381.979ms. Это исходная точка оптимизаций, не текущий самый быстрый путь. |
| results/n701-host-optimization-2026-11-08-001 | reference→optimized→optimized→reference; на128 pooled259.426→104.868ms, выигрыш59.89%/59.45% в двух раундах; приняты все этапы host оптимизации, default reference сохранён. |
| results/sparse-cpu-baseline-2026-11-08-001 | два раунда: на128 Torch386.262ms, Sparse shift62.832, cached53.838, Hailo105.812. Это отдельный запуск, не объединять его значения с OQS experiment. |
| results/edge-preparation-2026-11-08-001 | streamed evidence/lazy edge runtime, PC memory diagnostics; до ARM запуска. |
| results/oqs-integration-2026-11-08-001 | текущий external CPU comparison, KEM, profiles, corpus/raw JSON/CSV, hardware reports, source archives, 768MiB/swap0 runtime. |
| results/raspberry-pi-2026-11-09-001/ntru-pi-results | оригинальные smoke001/002 и qualification001 с Pi; **не performance benchmark**. |

Основные JSON/CSV и phase boundaries описаны в профильных документах. Медианы фаз не обязаны складываться в медиану total; разные N/разные дни/разные процессы не считать одним контролируемым экспериментом.

### Реальные результаты Pi

Smoke002: 19 polynomial probes,77frames,3messages,3opens, cleanup after host exception; RSS68.5MiB, artifact validation≈109.77s. После исправления DMA и последующего reboot qualification001: 2500probes/1000random, four message types, both Sparse kernels,3opens/recovery; RSS81.6MiB, весь процесс118.86s, throttled0. Нет ARM latency/throughput measurements, power measurements или cgroup/swap0 verification на Pi.

## 10. Карта исходников

| Путь | Назначение |
|---|---|
| src/ntru_pytorch/models.py, formats.py, codec.py | frozen models, JSON v1 и старые read-only readers, 8-byte big-endian length header/bit codec |
| tensor.py, torch_backend.py, sympy_backend.py | прежние независимые CPU implementations; default API |
| __init__.py | lazy public backend imports; SparseNTRU доступен явно, не добавлен в старый wildcard __all__ |
| array_api.py | exact NumPy normalize/encode, общий Sparse/Hailo host code |
| sparse_backend.py | SparseNTRU, два kernels, подготовка одного ключа, dense INT64 decrypt |
| hailo/scheme.py | binary weights, packing/map validation, reconstruction и software oracle |
| hailo/host.py | optimized pack/reconstruct; fallback для unsorted owners |
| hailo/proof.py | exact Fraction proof и affine LUT, compact decoder |
| hailo/evidence.py | потоковое чтение NPY в NPZ, no pickle, bounds/shape/dtype checks |
| hailo/artifacts.py | preparation/checksums/admission; НЕ исключать evidence checks ради старта |
| hailo/build.py, dfc_worker.py, dfc_extract.py | Linux/WSL DFC orchestration, worker и narrow observed SDK extractor |
| hailo/runtime.py, engine.py | actual HailoRT session + stream metadata checks, device identity, exact host processing |
| hailo/validation.py | hardware probes, independent conv, PC cross-backend qualification, mismatch diagnostics |
| hailo/benchmark.py, cli.py | legacy compatible experiment/CLI, фазовые метрики и host/packing variants |
| oqs/loader.py, kem.py, arithmetic.py | verified loading/ABI, KEM public bytes API, Rq encryption adapter |
| oqs/qualification.py, benchmark.py, cli.py | KAT/native checks, isolated workers, persisted corpus/windows, commands |
| native/oqs/{kem_bridge.c,rq_bridge.c,CMakeLists.txt,smoke.c} | собственные C-мосты и standalone sanitizer test; upstream source не vendored |
| tools/setup_dfc_ubuntu.sh | только Ubuntu24.04 x86_64 DFC setup, SHA/resume; не запускать на Pi |
| tools/install_hailo_linux.sh | только Ubuntu24.04 x86_64 system runtime/DKMS setup; не запускать на Pi |
| tools/build_runtime_wheel.py | тот же src → отдельный NumPy-only edge wheel, Python3.11–3.12, no build isolation |
| tools/build_oqs.py, check_oqs_interop.py | pinned local native profiles / per-process interoperability |
| tools/benchmark_hailo_host.py | программный фазовый microbenchmark, не NPU evidence |
| tools/benchmark_sparse_cpu.py | CPU-only fixed-key comparison, optional Torch |
| tools/qualify_oqs_hailo.py | PC real Hailo vs both OQS profiles + SymPy/Torch decrypt; на Pi пока не использовать с отсутствующими зависимостями |
| tools/profile_hailo_memory.py, profile_oqs_memory.py | memory diagnostics; последний ожидает external cgroup768MiB/swap0 |
| tools/pi_ntru_smoke.py | небольшой N701 device smoke и recovery, initial bundle copy |
| tools/pi_ntru_qualify.py | 1000+random device checks, lean NumPy references, progress, JSON/diagnostics |

Тесты: test_hailo{,_build,_packing,_host}.py, test_edge_runtime.py, test_sparse_backend.py, test_oqs.py плюс прежние CPU/model/codec tests. Mock/software tests не аппаратные отчёты. Native OQS tests пропускаются, если заранее не существует verified build; автоматически ничего не скачивают.

## 11. Команды продолжения и проверки

На ПК из корня, без устройства можно выполнить:

```bash
.venv/bin/python -m pytest -q -m 'not hailo_hardware'
.venv/bin/python -m ruff check src tests tools
bash -n tools/setup_dfc_ubuntu.sh tools/install_hailo_linux.sh
.venv/bin/python -m pip check
```

Финальный прогон checkpoint и статус каждого check записаны в [checkpoint-validation.md](checkpoint-validation.md). Один hardware-marked test намеренно исключён; это не отменяет отдельных реальных отчётов PC/Pi. Не повторять PC benchmarks/qualification автоматически на машине без физического Hailo.

Создание свежего edge wheel на ПК (новый output):

```bash
.venv/bin/python tools/build_runtime_wheel.py --output results/NEW-EDGE-WHEEL
```

Wheel pure-Python, содержит тот же src, имя пакета ntru-pytorch, версия0.1.0+edge. Не ставить его поверх desktop venv; только в отдельный runtime env. Он не содержит драйвер/firmware/SDK и не устанавливает Torch/SymPy. Base desktop pyproject продолжает зависеть от Torch/SymPy. NumPy1.x runtime ограничен Python<3.13; для более нового Python нужно отдельно решать совместимость, не брать первый попавшийся новый OS image.

На Pi уже установлен проект. Повторная квалификация, только если нужна и с новым output:

```bash
~/ntru-runtime/bin/python ~/pi_ntru_qualify.py \
  --bundle ~/ntru-pi-bundle --cases 1000 \
  --output ~/ntru-pi-results/qualification-NEW.json
```

Важно: старый `hailo-check` CLI безусловно делает ONNX Runtime validation; старые hailo-demo/benchmark cross-checks требуют SymPy/Torch. Не считать их готовыми lean ARM benchmark commands. Использовать Pi standalone tools либо заранее адаптировать новый процессный benchmark, сохраняя проверки.

Pinned OQS на ПК:

```bash
python3 tools/build_oqs.py --profile portable --jobs 2
python3 tools/build_oqs.py --profile native --jobs 2
.venv/bin/python -m ntru_pytorch oqs-info --oqs-profile portable
.venv/bin/python -m ntru_pytorch oqs-check --oqs-profile portable \
  --cases 1000 --output results/NEW-OQS-CHECK
```

Если передать --public-key/--private-key в oqs-check, дополнительно включаются тяжёлые SymPy/Torch checks. Для KEM benchmark обязательно задать --qualification-root с check-portable/check-native, не рассчитывать на дефолт results/oqs-qualification, если существующие результаты лежат в oqs-integration-2026-11-08-001. Подробнее в liboqs-integration.md.

### Сеть pip на Pi

www.piwheels.org таймаутился, но `wget -4 --spider ... https://pypi.org/simple/numpy/` вернул200. Это не доказательство геоблокировки. Рабочая одноразовая конфигурация:

```bash
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  ~/ntru-runtime/bin/python -m pip --disable-pip-version-check install \
  --index-url https://pypi.org/simple --timeout 15 --retries 0 PACKAGE
```

Не использовать один --isolated как гарантию исключения системного piwheels config. Для готовых NumPy/setuptools/wheel можно --only-binary=:all:, но не применять его ко всем зависимостям PyHailoRT: netifaces может требовать source build. Для локального project wheel --no-index --no-deps.

## 12. Данные вне Git и воспроизводимость

results/, ubuntu-transfer/, .venv/, *.hef, vendor tools и ключи намеренно игнорируются. Новый clone **не содержит** проверенных HEF/private keys/native libraries/DFC wheel. Нужны отдельные локальные артефакты по путям выше либо их копии с Pi. Если они отсутствуют, сначала запросить у пользователя их перенос; не создавать новый ключ и не выдавать новый HEF за тот же эксперимент.

В Git включены небольшие побайтовые копии отчётов и сводок: [evidence/checkpoint-2026-11-09/README.md](evidence/checkpoint-2026-11-09/README.md). manifest.json фиксирует оригинальный путь, размер и SHA256. Эти файлы позволяют другому чату прочитать свидетельства без старой машины; для повторного исполнения всё равно нужен внешний bundle. Данные numeric evidence/weights/ключи в этот каталог не копировались.

Полные результаты OQS в results/oqs-integration-2026-11-08-001: raw worker JSON/config/log, measurements.csv, summary/phase/native/cold-start CSV, corpus NPZ+hashes, KAT corpus и all/single rsp, hardware/sanitizer/memory reports, source archives. benchmark-sources.tar.gz — точные исходники измерений; integration-sources-final.tar.gz — финальная диагностическая версия. После измерений менялись диагностика/экспорт таблиц, не arithmetic path. sha256-manifest.json покрывает локальный комплект. KAT corpus содержит стандартные KAT secrets, но в Git он всё равно не включён; interop.json с экспериментальными KEM keys также остаётся вне Git.

Исходные четыре статьи и transfer-комплект локально: ubuntu-transfer/README.md, ubuntu-transfer/payload/articles/{Kravtsov_RN_a1.docx,Kravtsov_RN_a2.docx,Кравцов РН Винокур АИ Статья.docx,Кравцов РН Статья.docx}. Они не считаются текущими аппаратными свидетельствами. Локальные `ubuntu-transfer/CONTEXT.md` и `ubuntu-transfer/verify.py` существуют. При переносе старого payload: `python3 ubuntu-transfer/verify.py`. README/CONTEXT этого комплекта относятся к 7 ноября до Ubuntu-установки; указания там про ещё не адаптированный DFC и source-snapshot исторические. Не открывать https://CONTEXT.md как сайт и не восстанавливать старый snapshot поверх текущих исходников.

## 13. Ближайший следующий этап (ещё не выполнен)

1. Проверить git status и прочитать Pi qualification report; не повторять уже выполненную установку драйвера/перепрошивку/DFC. Убедиться, что пользователь действительно возобновляет работу.
2. Подготовить на Pi дерево tools/build_oqs.py + native/oqs/ в структуре проекта: builder определяет ROOT как родителя tools/. Один отдельно скопированный build_oqs.py в ~/ не является готовым комплектом.
3. Собрать **portable** liboqs0.16.0 на AArch64 с --jobs1, проверить manifest/ELF/ABI/KAT/≥1000 round-trips/арифметику. Затем native Cortex-A76 clean, записав, что отдельного NEON kernel нет. Результаты source/native x86 не переносить как ARM binaries.
4. На ARM сверить OQS arithmetic с существующим key/HEF и независимой свёрткой/Hailo. Torch/SymPy можно проверять позднее в отдельных окружениях/процессах; не загружать их в Hailo runtime для основного memory comparison.
5. Спроектировать и выполнить lean ARM benchmark, потому что текущий oqs-arith-benchmark orchestrator безусловно включает Torch и вызывает SymPy/Torch prechecks. Он пока ориентирован на ПК. Не просто пропускать проверки или ставить heavy libs в 1GB runtime ради запуска без изменений.
6. Перед измерениями решить охлаждение/повторяемость: имеющийся fan не совмещается с HAT, пока passively cooled. Зафиксировать thread count по реальному Pi CPU (не копировать 6 threads Ryzen), governor/temperature/throttle, swap и memory boundary, не менять ОС без необходимости.
7. Общий корпус key/messages/r, batch1/8/32/128, seed42,3warmups,20samples,3process runs/backend, порядок меняется; отдельно start/prep/packing/SDK/decode/recovery/total, validation вне таймера. Сохранять JSON/CSV, hashes/versions/RSS. KEM отдельно, мощность не выводить из времени.

Этап первого переноса на Pi уже успешен. Нельзя писать, что на Pi осталось только повторить установку, что она уже быстрее CPU или что текущие 119 секунд — скорость NPU.

## 14. Карта документации

| Документ | Зачем читать |
|---|---|
| Этот handoff | актуальные границы, состояние, пути, различия с14c24e7 и следующий этап |
| pi-first-run.md | хронология запуска/ошибки/реальные Pi результаты |
| pi-toolchain-runbook.md | точные известные рабочие команды установки ARM toolchain и восстановления |
| liboqs-integration.md | два уровня liboqs, профили, API/commands/measurement protocol |
| sparse-cpu-baseline.md | NumPy kernels и CPU/Hailo comparison |
| edge-runtime.md | separate wheel, streaming evidence, PC bounded-memory preparation |
| hailo-host-processing.md | detailed exact CPU optimization и paired acceptance |
| hailo-optimization.md | active16/LUT выбор, N107 и переход кN701 |
| hailo-n701-benchmark.md | историческая reference-host baseline265.7ms |
| hailo-ubuntu-run.md | Ubuntu DFC extraction/admission, исторический system installation журнал |
| hailo-research.md | математика/контракты и ранняя Windows/WSL история |
| ubuntu-checkpoint-2026-11-07.md | исторический аудит до установки DFC; НЕ актуальный список незавершённых задач |
| checkpoint-validation.md | финальные software checks этого checkpoint и сравнение staged diff |

Короткий запрос для нового чата: «Продолжаем NTRU_PyTorch. Сначала прочитай docs/CODEX_HANDOFF_2026-11-09.md, docs/pi-toolchain-runbook.md и versioned qualification-001.json. Проверь текущее дерево и локальные артефакты. N701 на Pi уже прошёл2500 аппаратных проб с4096 DMA; следующий этап — ARM liboqs и затем CPU/NPU performance, сохраняя PC-работоспособность и строгую точность». Это текст для передачи пользователем, не автоматический запуск следующих задач.
