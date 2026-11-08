# liboqs: два независимых CPU-ориентира

> На 9 ноября N701 runtime уже прошёл аппаратную квалификацию на Raspberry Pi. **Сборки liboqs и их производительность на ARM ещё не проверены.** Актуальный контекст: [handoff](CODEX_HANDOFF_2026-11-09.md).

Закреплены liboqs 0.16.0 и commit `5a1a854b0dc9f2141bdc771c555ee60c37950183`. Нативные библиотеки не входят в Python wheel: они собираются явно для соответствующей машины. Импорт `ntru_pytorch.oqs` не скачивает исходники, не собирает и не загружает библиотеки, не импортирует NumPy, Torch, SymPy или Hailo SDK.

`OqsKem` вызывает публичный NTRU-HRSS-701 KEM. Ключи и шифротексты — отдельные `bytes`; это другой протокол, его время нельзя делить на время нашего `encrypt_bytes` и называть ускорением NTRU.

`OqsArithmeticEncryptor` использует неизменённое upstream `poly_Rq_mul` для **циклического** произведения modulo `(x^701−1,8192)`. Редукция по Φ₇₀₁ не добавляется. Это внешний C-компонент нашего прежнего протокола при N=701,p=3,q=8192. Существующие PublicKey/Ciphertext, кодирование и дешифрование CPU сохраняются. Необрезанный trace произведения недоступен и отклоняется явно. Полный KEM этим адаптером не реализуется. Входы проверяются до сужения; партия проходит через один FFI-вызов. AVX2 работает с выровненными 32-байтовыми полиномами и нулевым padding до 704 коэффициентов.

## Сборка на Ubuntu x86_64

Нужны Git, CMake, GCC, Make и Python 3.11/3.12. Глобальная установка библиотек и `ldconfig` не используются. Проектная `.venv` не меняется.

```bash
python3 tools/build_oqs.py --profile portable --jobs 2
python3 tools/build_oqs.py --profile native --jobs 2
```

Выход: `results/native/oqs/0.16.0/x86_64/{portable,native}`. Checkout общий: `results/native/oqs/0.16.0/source`. Сборки Release/shared, OpenSSL и GPU отключены, минимальный алгоритм — HRSS-701. `portable` использует clean C с общим целевым CPU. `native` требует AVX2/BMI2; при их отсутствии завершается ошибкой, не подменяет профиль clean. Clean-умножение upstream использует Toom-4/Karatsuba.

`manifest.json` содержит commit, архитектуру, реализацию, ABI мостов, SHA256 библиотек/адаптеров/лицензий/сборочных файлов, CMake options и compiler version. Полный список SHA256 исходников checkout хранится отдельно. Сохраняются compile_commands и CMakeCache. Повторный запуск проверяет готовую сборку; несовместимую или повреждённую не перезаписывает. При изменении C-мостов старую сборку следует сохранить в другом каталоге до новой сборки. Python загружает только явный путь из проверенного manifest, проверяет ELF и фактический путь зависимости liboqs.

Один процесс может открывать только один профиль KEM: после закрытия контекста DSO остаётся загруженным, поэтому смена профиля требует нового процесса. Арифметические мосты независимы от liboqs SONAME.

```python
from ntru_pytorch.oqs import OqsKem, OqsArithmeticEncryptor

with OqsKem("results/native/oqs/0.16.0/x86_64/portable") as kem:
    pk, sk = kem.keypair()
    ct, secret = kem.encapsulate(pk)
    assert kem.decapsulate(sk, ct) == secret

# public — существующий PublicKey N701, rs — по одному тернарному r на блок.
with OqsArithmeticEncryptor("results/native/oqs/0.16.0/x86_64/native") as engine:
    engine.prepare(public)
    ciphertext = engine.encrypt_bytes(public, b"example", r_polys=rs)
```

KEM использует обычный RNG liboqs. Детерминированный генератор включается исключительно внутри отдельного upstream `kat_kem` процесса. Повреждённый KEM ciphertext обрабатывается upstream implicit rejection: успешный код и другой секрет допустимы. Неверные размеры отклоняются до вызова C. Временные ctypes-буферы секретов очищаются; возвращаемые Python `bytes` остаются обычными неизменяемыми объектами Python.

## Квалификация

```bash
.venv/bin/python -m ntru_pytorch oqs-info --oqs-profile portable
.venv/bin/python -m ntru_pytorch oqs-check --oqs-profile portable \
  --output results/oqs-check-clean-new --cases 1000 \
  --public-key results/admitted-n701-ubuntu-affine-001/public.json \
  --private-key results/compiler-experiments/n701-active16-2026-11-08/private.json
# Повторить для --oqs-profile native с другим output.
```

Полная проверка сохраняет исходный вывод upstream KAT и сверяет SHA256 одиночного и полного набора из 100 векторов с закреплёнными эталонами upstream. Проверяются 1000 случайных KEM round-trip, 1000 случайных тернарных произведений, 1412 граничных проб, полные остатки 0–8191, API-ошибки, повторное открытие и шифрование/расшифрование с SymPy, Torch и обоими SparseNTRU. Независимый арифметический эталон — INT64 `np.convolve` с циклическим сворачиванием и modulo q.

Для взаимной совместимости двух готовых qualification-каталогов:

```bash
.venv/bin/python tools/check_oqs_interop.py \
  --oqs-root results/native/oqs/0.16.0/x86_64 \
  --qualification-root results/oqs-integration-2026-11-08-001
```

Sanitizer отдельно от измерений:

```bash
python3 tools/build_oqs.py --profile portable --sanitize --jobs 2
results/native/oqs/0.16.0/x86_64/portable-sanitizer/bridge-build/bridge_smoke
```

ASan/UBSan-сборка не загружается в обычный Python. Standalone executable проверяет 1000 KEM round-trip и границы арифметического моста. LeakSanitizer нужно запускать вне ptrace-песочницы. Upstream исходники ради sanitizer не меняются.

Реальная NPU-квалификация:

```bash
.venv/bin/python tools/qualify_oqs_hailo.py \
  --artifact results/admitted-n701-ubuntu-affine-001 \
  --private-key results/compiler-experiments/n701-active16-2026-11-08/private.json \
  --oqs-root results/native/oqs/0.16.0/x86_64 \
  --output results/oqs-hardware-new
```

Она сравнивает оба арифметических OQS-профиля с реальным Hailo, проверяет точные частичные суммы, оба CPU-дешифрования и повторное открытие после исключения. Software-проверка не считается NPU-подтверждением.

## Измерения

```bash
.venv/bin/python -m ntru_pytorch oqs-kem-benchmark \
  --qualification-root results/oqs-integration-2026-11-08-001 \
  --output results/oqs-kem-new --runs 20 --warmup 3
.venv/bin/python -m ntru_pytorch oqs-arith-benchmark \
  --public-key results/admitted-n701-ubuntu-affine-001/public.json \
  --private-key results/compiler-experiments/n701-active16-2026-11-08/private.json \
  --artifact results/admitted-n701-ubuntu-affine-001 --hardware \
  --output results/oqs-arith-new --seed 42 --runs 20 --warmup 3
```

Каталог output должен быть новым. Арифметический бенчмарк создаёт один NPZ-корпус с разными сообщениями/r для каждого прогрева и измерения, ожидаемыми шифротекстами и SHA256. Он проверяет публичный/секретный ключ, шифрование и CPU-кросс-расшифрование на каждом размере партии до запуска worker. Каждый из шести backend запускается в трёх отдельных процессах с изменяемым порядком, шесть потоков Torch. Hailo использует active16-v1, optimized host и кэш LUT; перед каждым аппаратным worker повторяется 1000 случайных и граничные пробы. Все измеренные результаты сравниваются с эталоном вне таймера. Передача — восемь кадров на полином, UINT8 5632 входных и 72960 выходных байт на полином.

Сохраняются JSON worker, configs/logs, общий отчёт, сырые CSV, медианы/p95 по каждому запуску и объединённые, payload bytes/s и blocks/s, исходные хеши, версии, подготовка ключа, стартовые времена, фазы и peak RSS. Для OQS отдельно измеряется C-only Rq-цикл: включает канонизацию результата и volatile checksum, исключает подготовку входных буферов. Это микробенчмарк, а не полное время приложения.

KEM имеет отдельные C/Python показатели keypair/encapsulate/decapsulate, три процесса на профиль, три прогрева и двадцать окон не короче 100 мс. Фиксированные KAT-входы валидируются перед измерениями; внутренняя случайность и keypair не исключаются из соответствующей операции. Единицы — ns/op и op/s. Не проводится сопоставление KEM с полезными байтами нашего сообщения, измерение мощности или вывод внутреннего времени NPU.

## Подготовка Raspberry Pi 5

На 64-битной Linux AArch64 используется тот же скрипт:

```bash
python3 tools/build_oqs.py --profile portable --jobs 1
python3 tools/build_oqs.py --profile native --jobs 1
```

Результаты попадут в `.../aarch64/{portable,native}`. Оба профиля HRSS-701 используют clean C; native дополнительно задаёт Cortex-A76 для компилятора. Отдельное NEON-ядро HRSS не предполагается. Архитектурные библиотеки x86 на Pi переносить нельзя. Облегчённый Python wheel собирается существующим `tools/build_runtime_wheel.py`; для арифметики нужна NumPy, для KEM достаточно stdlib и проверенных DSO. Torch/SymPy требуются для расширенной исследовательской проверки на ПК и не импортируются runtime автоматически.

Инструкция и ветвление сборки подготовлены, но ARM-сборка и работа на Raspberry Pi считаются подтверждёнными только после реального выполнения на ней.

## Результат текущего ПК

Полный отчёт и таблицы: [results/oqs-integration-2026-11-08-001/RESULTS.md](../results/oqs-integration-2026-11-08-001/RESULTS.md). На 128 блоках медианы полного шифрования: PyTorch 380,6 мс, Sparse cached 51,8 мс, OQS clean 48,3 мс, OQS AVX2 45,9 мс, Hailo 102,3 мс. Новый внешний CPU-ориентир быстрее Hailo на всех четырёх проверенных партиях. KEM приведён в отдельной таблице.

Пройдены 137 программных тестов, KAT и квалификация обоих профилей, совместимость между ними, ASan/UBSan и реальная проверка Hailo. В облегчённом wheel без установленных Torch/SymPy выполнены KEM и OQS/Hailo 128 блоков под cgroup-лимитом 768 МиБ без swap; пик RSS 99.7 МиБ, OOM нет. Cgroup учитывает разделяемые страницы иначе, чем RSS. Это проверка процесса на x86_64, не всей ОС Raspberry Pi.

Воспроизведение ограниченного runtime после установки wheel, NumPy и PyHailoRT в отдельное окружение:

```bash
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 \
  /path/to/edge/bin/python tools/profile_oqs_memory.py \
  --oqs-root results/native/oqs/0.16.0/x86_64 \
  --artifact results/admitted-n701-ubuntu-affine-001 \
  --output results/oqs-memory-new.json
```

Снимок исходников измерений сохранён в benchmark-sources.tar.gz; SHA256 каждого модуля записан в отчётах worker.
