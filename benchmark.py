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
from utils import *
from utils_torch import genRand10_pytorch, padArr_pytorch, linear_convolve_pytorch_direct, reduce_poly_mod_ideal_pytorch, multiply_poly_pytorch_circulant
from ntru import PARAM_SETS

# --- Функция для пошагового Sympy шифрования ---
def debug_encrypt_sympy(h_np, r_np, m_np, N, q):
    results = {}
    x = symbols('x')

    # --- НОВЫЙ, БОЛЕЕ НАДЕЖНЫЙ СПОСОБ СОЗДАНИЯ POLY ---
    # h_np, r_np, m_np УЖЕ в порядке [a0, a1, ..., aN-1]
    P_r_ZZ = Poly(dict(enumerate(r_np)), x, domain=ZZ)
    P_h_ZZ = Poly(dict(enumerate(h_np)), x, domain=ZZ)
    P_m_ZZ = Poly(dict(enumerate(m_np)), x, domain=ZZ)
    # Для x^N - 1 коэффициенты: 1 при x^N и -1 при x^0
    P_I_ZZ = Poly({N: 1, 0: -1}, x, domain=ZZ)

    # --------------------------------------------------

    def to_np_a0_order(poly_sympy, target_len):
        # Эта функция теперь должна быть более надежной
        # all_coeffs() возвращает [старший, ..., младший]
        coeffs_sympy_order = poly_sympy.all_coeffs()
        degree = poly_sympy.degree()

        np_arr = np.zeros(target_len, dtype=np.int64)

        # Если полином - ноль
        if poly_sympy.is_zero:
            return np_arr

        # Заполняем numpy массив
        for i, coeff in enumerate(coeffs_sympy_order):
            current_degree = degree - i
            if current_degree < target_len:
                np_arr[current_degree] = coeff
        return np_arr

    # 1. r*h (в Z[x])
    rh_poly_full_ZZ = P_r_ZZ * P_h_ZZ
    # Длина результата будет N-1 + N-1 + 1 = 2N-1
    results['step1_rh'] = to_np_a0_order(rh_poly_full_ZZ, 2 * N - 1)

    # 2. (r*h)_coeffs % q
    term1_coeffs_mod_q = [c % q for c in rh_poly_full_ZZ.all_coeffs()]
    P_term1_mod_q = Poly(term1_coeffs_mod_q, x, domain=ZZ)  # Временно в ZZ
    results['step2_rh_mod_q'] = to_np_a0_order(P_term1_mod_q, 2 * N - 1)

    # 3. (r*h)_coeffs % q + m
    sum_intermediate = P_term1_mod_q + P_m_ZZ
    results['step3_plus_m'] = to_np_a0_order(sum_intermediate, 2 * N - 1)

    # 4. (...) % (x^N-1)
    sum_coeffs_reduced = rem(sum_intermediate, P_I_ZZ, domain=ZZ)
    results['step4_mod_ideal'] = to_np_a0_order(sum_coeffs_reduced, N)

    # 5. (...) % q (финальный)
    e_mod_q_positive = [c % q for c in sum_coeffs_reduced.all_coeffs()]
    results['step5_final_mod_q'] = to_np_a0_order(Poly(e_mod_q_positive, x), N)

    # 6. Финальный центрированный e
    e_centered_coeffs = []
    for c_val in e_mod_q_positive:
        if 2 * c_val >= q:
            e_centered_coeffs.append(c_val - q)
        else:
            e_centered_coeffs.append(c_val)
    results['step6_final_centered_e'] = to_np_a0_order(Poly(e_centered_coeffs, x), N)

    return results


# --- Функция для пошагового PyTorch шифрования ---
def debug_encrypt_pytorch(h_torch, r_torch, m_torch, N, q, device):
    results = {}

    h_comp = h_torch.to(device=device, dtype=torch.int64)
    r_comp = r_torch.to(device=device, dtype=torch.int64)
    m_comp = m_torch.to(device=device, dtype=torch.int64)

    # 1. r*h (в Z[x])
    rh_poly_full = linear_convolve_pytorch_direct(r_comp, h_comp)
    results['step1_rh'] = rh_poly_full.cpu().numpy()

    # 2. (r*h)_coeffs % q
    term1_coeffs_mod_q = torch.remainder(rh_poly_full, q)
    results['step2_rh_mod_q'] = term1_coeffs_mod_q.cpu().numpy()

    # 3. ... + m
    m_comp_padded = torch.nn.functional.pad(m_comp, (0, rh_poly_full.shape[0] - N), value=0)
    sum_intermediate = term1_coeffs_mod_q + m_comp_padded
    results['step3_plus_m'] = sum_intermediate.cpu().numpy()

    # 4. ... % (x^N-1)
    sum_coeffs_reduced = reduce_poly_mod_ideal_pytorch(sum_intermediate, N)
    results['step4_mod_ideal'] = sum_coeffs_reduced.cpu().numpy()

    # 5. (...) % q (финальный)
    e_mod_q_positive = torch.remainder(sum_coeffs_reduced, q)
    results['step5_final_mod_q'] = e_mod_q_positive.cpu().numpy()

    # 6. Финальный центрированный e
    e_centered = e_mod_q_positive.clone()
    mask_q = 2 * e_centered >= q
    e_centered[mask_q] -= q
    results['step6_final_centered_e'] = e_centered.cpu().numpy()

    return results

def run_performance_benchmark(message: str, n_runs: int = 100, n_warmup: int = 10):
    """
    Проводит сравнительный анализ производительности.
    n_warmup: количество "прогревочных" запусков.
    """
    logger.setLevel(logging.INFO)
    logger.info("\n")
    logger.info("=" * 50)
    logger.info("НАЧАЛО СРАВНИТЕЛЬНОГО АНАЛИЗА ПРОИЗВОДИТЕЛЬНОСТИ")
    logger.info(f"Количество прогревочных запусков: {n_warmup}")
    logger.info(f"Количество замеров для усреднения: {n_runs}")
    logger.info(f"Сообщение: '{message}'")
    logger.info("=" * 50)

    generate_keys("key", mode="moderate", skip_check=True)

    # --- Sympy-реализация ---
    logger.info("--- Тестирование Sympy-реализации ---")
    # Прогрев
    for _ in range(n_warmup):
        _ = encrypt_sympy("key", message, check_time=False)
    # Замер
    sympy_encrypt_times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        enc_s = encrypt_sympy("key", message, check_time=False)
        end_time = time.perf_counter()
        sympy_encrypt_times.append(end_time - start_time)
    # Расшифровка
    sympy_decrypt_times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = decrypt_sympy("key", enc_s, check_time=False)
        end_time = time.perf_counter()
        sympy_decrypt_times.append(end_time - start_time)

    sympy_avg_encrypt_time = np.mean(sympy_encrypt_times)
    sympy_avg_decrypt_time = np.mean(sympy_decrypt_times)

    # --- PyTorch-реализация ---
    logger.info("--- Тестирование PyTorch-реализации ---")
    # Прогрев
    for _ in range(n_warmup):
        _ = encrypt_torch("key", message, check_time=False)
    # Замер
    torch_encrypt_times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        enc_t = encrypt_torch("key", message, check_time=False)
        end_time = time.perf_counter()
        torch_encrypt_times.append(end_time - start_time)
    # Расшифровка
    torch_decrypt_times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        _ = decrypt_torch("key", enc_t, check_time=False)
        end_time = time.perf_counter()
        torch_decrypt_times.append(end_time - start_time)

    torch_avg_encrypt_time = np.mean(torch_encrypt_times)
    torch_avg_decrypt_time = np.mean(torch_decrypt_times)

    # --- Итоговая таблица (используем logger.info) ---
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("Таблица 2: Сравнение производительности (среднее время выполнения, сек)")
    logger.info("=" * 80)
    logger.info(f"{'Реализация':<20} | {'Время шифрования':<20} | {'Время расшифровки':<20}")
    logger.info("-" * 70)
    logger.info(f"{'Sympy-реализация':<20} | {sympy_avg_encrypt_time:<20.4f} | {sympy_avg_decrypt_time:<20.4f}")
    logger.info(f"{'PyTorch-реализация':<20} | {torch_avg_encrypt_time:<20.4f} | {torch_avg_decrypt_time:<20.4f}")
    logger.info("-" * 70)

    encrypt_speedup = sympy_avg_encrypt_time / torch_avg_encrypt_time
    decrypt_speedup = sympy_avg_decrypt_time / torch_avg_decrypt_time

    logger.info(f"{'Ускорение (раз)':<20} | {encrypt_speedup:<20.2f}x | {decrypt_speedup:<20.2f}x")
    logger.info("=" * 80)
    logger.info("\n")


def benchmark_sympy_mult(N: int, p: int = 3):
    """Измеряет время умножения полиномов с помощью Sympy."""
    # Подготовка данных
    x = symbols('x')
    # Используем полиномы с коэффициентами в Z_p, так как это типично для NTRU
    # Для p=3, это будут {-1, 0, 1}
    # genRand10 генерирует полиномы с такими коэффициентами
    poly1_np = genRand10(N, N // 6, N // 6)  # Генерируем случайные полиномы
    poly2_np = genRand10(N, N // 6, N // 6)

    P1 = Poly(poly1_np[::-1], x, domain=ZZ)
    P2 = Poly(poly2_np[::-1], x, domain=ZZ)

    I_list = [1] + [0] * (N - 1) + [-1]
    P_I = Poly(I_list, x, domain=ZZ)

    # Замер времени
    start_time = time.perf_counter()
    result = (P1 * P2) % P_I  # Основная операция
    # Чтобы быть честным, добавим и преобразование в массив, как в реальном коде
    coeffs = result.all_coeffs()
    end_time = time.perf_counter()

    return end_time - start_time


def benchmark_pytorch_mult(N: int, p: int = 3, device: str = 'cpu'):
    """Измеряет время умножения полиномов с помощью PyTorch."""
    # Подготовка данных
    # Генерируем те же самые полиномы для чистоты эксперимента
    np.random.seed(42)  # Фиксируем seed для np
    poly1_np = genRand10(N, N // 6, N // 6)
    np.random.seed(1337)  # Другой seed для второго полинома
    poly2_np = genRand10(N, N // 6, N // 6)

    # Переводим в тензоры
    p1 = torch.from_numpy(poly1_np.astype(np.int64)).to(device)
    p2 = torch.from_numpy(poly2_np.astype(np.int64)).to(device)

    # Замер времени
    start_time = time.perf_counter()
    # Ваша быстрая функция, которая делает (p1*p2) mod I mod p
    # Мы тестируем именно умножение в кольце R_q, поэтому модуль - это q
    # Но для сравнения скорости полиномиального умножения, модуль не так важен.
    # Давайте используем p, так как это тоже легитимная операция (fp * b).
    result = multiply_poly_pytorch_circulant(p1, p2, N, modulus=p)
    end_time = time.perf_counter()

    return end_time - start_time