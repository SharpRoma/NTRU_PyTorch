# Проверки checkpoint 9 ноября 2026

Проверяется рабочее дерево относительно `14c24e7de56669295cdf71b6d858886edad5f101`. Сам коммит этим этапом не создаётся. Главный документ: [CODEX_HANDOFF_2026-11-09.md](CODEX_HANDOFF_2026-11-09.md).

## Свежие программные проверки на Ubuntu ПК

| Проверка | Результат |
|---|---|
| `.venv/bin/python -m pytest -q -m 'not hailo_hardware'` | **137 passed, 1 deselected**, 103.16 s |
| `.venv/bin/python -m ruff check src tests tools` | All checks passed |
| `bash -n tools/setup_dfc_ubuntu.sh tools/install_hailo_linux.sh` | exit 0 |
| `.venv/bin/python -m compileall -q src/ntru_pytorch tools` | exit 0 |
| `git diff --check` | exit 0 |

Исходные логи — `results/documentation-checkpoint-2026-11-09-001/`; их побайтовые versioned копии: [pytest](evidence/checkpoint-2026-11-09/pc/checkpoint-pytest.log), [Ruff](evidence/checkpoint-2026-11-09/pc/checkpoint-ruff.log), SHA256 в [manifest](evidence/checkpoint-2026-11-09/manifest.json).

Это программный прогон ПК, не новый аппаратный эксперимент. Единственный hardware-marked test намеренно исключён: модуль Hailo сейчас физически находится на Pi. Во время оформления checkpoint новые бенчмарки, DFC-компиляции и установки на Pi не выполнялись.

## Аппаратные и native свидетельства

PC N17/N107/N701: отдельные реальные historical qualification reports в [evidence/pc](evidence/checkpoint-2026-11-09/pc). Квалификация OQS portable/native, upstream KAT, interop, sanitizer и memory/cgroup также сохранены отдельно. Оригинальные полные сырые эксперименты остаются в results/, а в Git попали небольшие отчёты/сводки/provenance.

Pi: original smoke001/002/qualification001 пользователь выполнил по SSH и перенёс через SCP. Свежие локальные копии прочитаны агентом, параметры/хеши сопоставлены с текущими исходниками. Квалификация N701:2500 проб/1000random, exact=true, оба Sparse kernels, сообщения и session recovery. Отчёт явно не подтверждает Torch/SymPy на Pi и не является performance benchmark. [Оригинальные отчёты в Git](evidence/checkpoint-2026-11-09/pi).

## Состав индекса Git

Включаются все изменения реализации этого этапа: README, pyproject, src, native/oqs, tools, tests, профильные docs, handoff/runbook и небольшие evidence JSON/logs. Собственные C-мосты включены; checkout upstream liboqs не vendored.

Исключены: .venv, /opt toolchain, results целиком, vendor binaries/checkouts, DFC wheel, HEF/HAR/NPZ, private keys, KEM interop keys, bundle с экспериментальным private key, исходные статьи и ubuntu-transfer. Для нового clone внешние артефакты всё равно надо переносить отдельно. `git add -f results` не применялся.

Текст Conventional Commit: [COMMIT_MESSAGE_2026-11-09.txt](COMMIT_MESSAGE_2026-11-09.txt). Он описывает весь diff после 14c24e7, а не только текущую документацию. Перед фактическим commit пользователь может просмотреть `git diff --cached --stat` и `git diff --cached`.
