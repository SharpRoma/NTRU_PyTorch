import time
import numpy as np
import torch
import logging
from logger import logger
from sympy import Poly, symbols, rem
from sympy.polys.domains import ZZ
import textwrap
from ntru import generate_keys, encrypt as encrypt_sympy, decrypt as decrypt_sympy
from NTRU_torch import encrypt_torch, decrypt_torch
from utils_torch import genRand10_pytorch, padArr_pytorch, linear_convolve_pytorch_direct, reduce_poly_mod_ideal_pytorch
from utils import *
from benchmark import debug_encrypt_sympy, debug_encrypt_pytorch, run_performance_benchmark, benchmark_sympy_mult, benchmark_pytorch_mult
from ntru import PARAM_SETS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger.setLevel(logging.INFO)
generate_keys("key", mode="moderate", skip_check=True, debug=False, check_time=False)
MESSAGE = "test2testtest2testtest2testtest2testtest2testtest2testtest2testtest2testtest2testtest2test"

logger.info("--- Шифруем с помощью Sympy ---")
enc_sympy1 = encrypt_sympy("key", MESSAGE, debug=True)
# logger.info(f"Sympy шифротекст: {enc_sympy1[:100]}...")

logger.info("--- Расшифровка шифротекста от Sympy с помощью PyTorch ---")
dec_from_sympy_1 = decrypt_torch("key", enc_sympy1, debug=True)
logger.info(f"Результат: '{dec_from_sympy_1}'")
if dec_from_sympy_1 == MESSAGE: logger.info("УСПЕХ!\n")
else: logger.error("ОШИБКА!\n")

logger.info("--- Шифруем с помощью PyTorch ---")
enc_torch2 = encrypt_torch("key", MESSAGE, debug=True)
# logger.info(f"PyTorch шифротекст: {enc_torch2[:100]}...")

logger.info("--- Расшифровка шифротекста от PyTorch с помощью Sympy ---")
dec_from_sympy_2 = decrypt_sympy("key", enc_torch2, debug=True)
logger.info(f"Результат: '{dec_from_sympy_2}'")
if dec_from_sympy_2 == MESSAGE: logger.info("УСПЕХ!\n")
else: logger.error("ОШИБКА!\n")

logger.info("--- Шифруем с помощью Sympy ---")
enc_sympy3 = encrypt_sympy("key", MESSAGE, debug=True)
# logger.info(f"Sympy шифротекст: {enc_sympy3[:100]}...")

logger.info("--- Расшифровка шифротекста от Sympy с помощью Sympy ---")
dec_from_sympy_3 = decrypt_sympy("key", enc_sympy3, debug=True)
logger.info(f"Результат: '{dec_from_sympy_3}'")
if dec_from_sympy_3 == MESSAGE: logger.info("УСПЕХ!\n")
else: logger.error("ОШИБКА!\n")

logger.info("--- Шифруем с помощью PyTorch ---")
enc_torch4 = encrypt_torch("key", MESSAGE, debug=True)
# logger.info(f"PyTorch шифротекст: {enc_torch4[:100]}...")

logger.info("--- Расшифровка шифротекста от PyTorch с помощью PyTorch ---")
dec_from_sympy_4 = decrypt_torch("key", enc_torch4, debug=True)
logger.info(f"Результат: '{dec_from_sympy_4}'")
if dec_from_sympy_4 == MESSAGE: logger.info("УСПЕХ!\n")
else: logger.error("ОШИБКА!\n")
time.sleep(3)

generate_keys("key", mode="moderate", skip_check=True, debug=False, check_time=False)

h_numpy, N, p, q, dr = read_key_params_from_pub("key.pub")
h_torch = torch.from_numpy(h_numpy).to(dtype=torch.int64)

MESSAGE = "test"
m_bits = str2bit(MESSAGE)
m_numpy = np.zeros(N, dtype=np.int64)
m_numpy[:len(m_bits)] = m_bits
m_torch = torch.from_numpy(m_numpy).to(dtype=torch.int64)

np.random.seed(42)
r_numpy = genRand10(N, dr, dr)
r_torch = torch.from_numpy(r_numpy).to(dtype=torch.int64)

print("--- Проведение пошагового сравнения шифрования ---")
sympy_steps = debug_encrypt_sympy(h_numpy, r_numpy, m_numpy, N, q)
pytorch_steps = debug_encrypt_pytorch(h_torch, r_torch, m_torch, N, q, 'cpu')

steps_to_compare = [
    ('step1_rh', 'r*h (в Z[x])'),
    ('step2_rh_mod_q', '(r*h)_coeffs % q'),
    ('step3_plus_m', '(r*h)_coeffs % q + m'),
    ('step4_mod_ideal', '(...) % (x^N-1)'),
    ('step5_final_mod_q', '(...) % q (финальный)'),
    ('step6_final_centered_e', 'Финальный центрированный e')
]

# --- Формирование вывода для таблицы ---
print("\n" + "=" * 130)
print("Таблица 1: Сравнение промежуточных результатов шифрования для Sympy и PyTorch")
print("=" * 130)
print(
    f"{'Шаг вычисления e':<35} | {'Результат Sympy (первые 8 коэфф.)':<40} | {'Результат PyTorch (первые 8 коэфф.)':<40} | {'Совпадение':<10}")
print("-" * 130)

for key, name in steps_to_compare:
    sympy_res = sympy_steps[key]
    pytorch_res = pytorch_steps[key]

    # Приводим к спискам обычных Python int для сравнения и вывода
    sympy_res_list = [int(c) for c in sympy_res]
    pytorch_res_list = [int(c) for c in pytorch_res]

    # Сравниваем только общую часть
    min_len = min(len(sympy_res_list), len(pytorch_res_list))
    is_equal = sympy_res_list[:min_len] == pytorch_res_list[:min_len]

    # Берем срез для вывода
    sympy_res_to_print = sympy_res_list[:8]
    pytorch_res_to_print = pytorch_res_list[:8]

    # Передаем в textwrap строки, созданные из списков обычных int
    sympy_str = textwrap.shorten(str(sympy_res_to_print), width=38, placeholder="...")
    pytorch_str = textwrap.shorten(str(pytorch_res_to_print), width=38, placeholder="...")

    print(f"{name:<35} | {sympy_str:<40} | {pytorch_str:<40} | {'Да' if is_equal else 'НЕТ'}")

print("=" * 130)
time.sleep(3)

MESSAGE_FOR_BENCHMARK = "test2testtest2testtest2testtest2testtest2test"
run_performance_benchmark(MESSAGE_FOR_BENCHMARK, n_runs=100, n_warmup=10)
time.sleep(3)

logger.info("\n")
logger.info(f"--- Проведение анализа масштабируемости ---")

# Количество прогонов для усреднения
NUM_RUNS = 10

results = {}
DEVICE = 'cpu'
PARAM_SETS_FOR_PLOT = {
        "moderate": {"N": 107}, "high": {"N": 167}, "highest": {"N": 503},
        "dead": {"N": 701}, "dead2": {"N": 821}
    }

for mode, params in PARAM_SETS.items():
    N = params["N"]
    logger.info(f"--- Тестирование для N = {N} (режим '{mode}') ---")

    # Прогрев (первый вызов может быть медленнее)
    benchmark_sympy_mult(N)
    benchmark_pytorch_mult(N, device=DEVICE)

    # Замер Sympy
    sympy_times = []
    for _ in range(NUM_RUNS):
        sympy_times.append(benchmark_sympy_mult(N))
    avg_sympy_time = sum(sympy_times) / NUM_RUNS

    # Замер PyTorch
    pytorch_times = []
    for _ in range(NUM_RUNS):
        pytorch_times.append(benchmark_pytorch_mult(N, device=DEVICE))
    avg_pytorch_time = sum(pytorch_times) / NUM_RUNS

    results[N] = {
        "mode": mode,
        "sympy_time": avg_sympy_time,
        "pytorch_time": avg_pytorch_time,
        "speedup": avg_sympy_time / avg_pytorch_time
    }

# --- Вывод результатов в виде красивой таблицы ---
logger.info("\n")
logger.info("=" * 80)
logger.info("Результаты анализа масштабируемости (среднее время умножения полиномов, сек)")
logger.info("=" * 80)
logger.info(f"{'N':<6} | {'Режим':<10} | {'Время Sympy':<15} | {'Время PyTorch':<15} | {'Ускорение (раз)':<20}")
logger.info("-" * 80)

# Сортируем результаты по N для наглядности
for N in sorted(results.keys()):
    res = results[N]
    logger.info(
        f"{N:<6} | {res['mode']:<10} | {res['sympy_time']:<15.6f} | {res['pytorch_time']:<15.6f} | {res['speedup']:<20.2f}x")

logger.info("=" * 80)
logger.info("\n")

logger.info("--- Построение графика производительности ---")

# 1. Подготовка данных для графика с помощью Pandas DataFrame
plot_data = []
# Переводим ключи словаря на русский для легенды
for N, res in results.items():
    plot_data.append({'Реализация': 'Sympy', 'Степень полинома (N)': N, 'Время выполнения (с)': res['sympy_time']})
    plot_data.append({'Реализация': 'PyTorch', 'Степень полинома (N)': N, 'Время выполнения (с)': res['pytorch_time']})

df = pd.DataFrame(plot_data)

# 2. Настройка стиля и шрифта для поддержки кириллицы
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# Указываем шрифт, который поддерживает кириллицу.
# 'DejaVu Sans' обычно установлен с matplotlib. 'Arial' тоже хороший вариант, если он есть в системе.
# Если шрифта нет, matplotlib может выдать предупреждение и использовать стандартный.
plt.rcParams['font.family'] = ['DejaVu Sans']

# 3. Создание фигуры и осей
fig, ax = plt.subplots(figsize=(10, 6))  # Немного увеличим ширину для длинных русских надписей

# 4. Построение графика
sns.lineplot(
    data=df,
    x='Степень полинома (N)',
    y='Время выполнения (с)',
    hue='Реализация',  # Разделение по цвету
    style='Реализация',  # Разделение по стилю линии
    markers=True,
    dashes=False,
    ax=ax
)

# 5. Настройка осей и заголовков на РУССКОМ языке
ax.set_title('Зависимость производительности умножения полиномов от степени N', fontsize=14, weight='bold')
ax.set_xlabel('Степень полинома (N)', fontsize=12)
ax.set_ylabel('Время выполнения (с, логарифмическая шкала)', fontsize=12)
ax.set_yscale('log')  # !!! ИСПОЛЬЗУЕМ ЛОГАРИФМИЧЕСКУЮ ШКАЛУ для Y !!!
# Это критично, чтобы были видны обе линии.

# Настройка легенды
ax.legend(title='Реализация')  # Заголовок легенды

# Улучшаем внешний вид
plt.tight_layout()

# 6. Сохранение графика в файл
output_filename = "ntru_performance_scaling_rus"
try:
    fig.savefig(f"{output_filename}.png", dpi=300, bbox_inches='tight')
    logger.info(f"График успешно сохранен в файлы {output_filename}.png")
except Exception as e:
    logger.error(f"Не удалось сохранить график: {e}")
# plt.show()

