import numpy as np
import sys
from sympy import Poly, symbols
from utils import *
from sympy.polys.polytools import rem
from sympy.polys.domains import ZZ, GF



class NTRUencrypt:
    """
    A class to encrypt some data based on a known public key.
    """
    def __init__(self, logger, N=503, p=3, q=256, d=18, debug=True, check_time=True):
    # def __init__(self, logger, N=503, p=3, q=256, df=61, dg=20, d=18, debug=True, check_time=False):
        """
        Initialise with some default N, p and q parameters.
        """
        self.N = N  # Public N
        self.p = p  # Public p
        self.q = q  # Public q
        self.debug = debug
        self.check_time = check_time
        self.logger = logger

        self.dr = d  # Number of 1's in r (for encryption)

        self.g = np.zeros((self.N,), dtype=int)  # Private polynomial g
        self.h = np.zeros((self.N,), dtype=int)  # Public key polynomial (mod q)
        self.r = np.zeros((self.N,), dtype=int)  # A random `blinding value'
        self.genr()
        self.m = np.zeros((self.N,), dtype=int)  # The message array
        self.e = np.zeros((self.N,), dtype=int)  # The encrypted message

        # Ideal as array representing polynomial
        self.I = np.zeros((self.N + 1,), dtype=int)
        self.I[self.N] = -1
        self.I[0] = 1

        self.readKey = False  # We have not yet read the public key file

        # Variables to save any possible encrypted messages (if req)
        self.Me = None  # The encrypted message as a string

    def readPub(self, filename="key.pub"):
        """
        Read a public key file, generate a new r value based on new N
        """
        with open(filename, "r") as f:
            self.p = int(f.readline().split(" ")[-1])
            self.q = int(f.readline().split(" ")[-1])
            self.N = int(f.readline().split(" ")[-1])
            self.dr = int(f.readline().split(" ")[-1])
            self.h = np.array(f.readline().split(" ")[3:-1], dtype=int)
        self.I = np.zeros((self.N + 1,), dtype=int)
        self.I[self.N] = -1
        self.I[0] = 1
        self.genr()
        self.readKey = True

    def genr(self):
        """
        Generate the random binding polynomial array r, with values mod q
        """
        self.r = genRand10(self.N, self.dr, self.dr)

    def setM(self, M):
        if not self.readKey:
            sys.exit("ERROR : Public key not read before setting message")
        if len(M) > self.N:
            sys.exit("ERROR : Message length longer than degree of polynomial ring ideal")
        for i in range(len(M)):
            if M[i] < -self.p / 2 or M[i] > self.p / 2:
                sys.exit("ERROR : Elements of message must be in [-p/2,p/2]")
        # Passed the error checks, so now save the class message function, inc leading zeros
        self.m = padArr(M, self.N)
        if self.debug:
            self.logger.debug(f"DEBUG setM: self.m установлен (первые 10): {self.m[:10]}")

    def encrypt(self, m=None, fixed_r=None):
        if self.debug:
            self.logger.debug(f"DEBUG encrypt (начало): self.m на входе (первые 10): {self.m[:10]}")
        if not self.readKey:
            sys.exit("Error : Not read the public key file, so cannot encrypt")

        current_m_coeffs = self.m
        if m is not None:
            # Убедимся, что m - это numpy массив нужной длины
            if len(m) != self.N:
                m_padded = np.zeros(self.N, dtype=np.int64)
                m_padded[:min(self.N, len(m))] = m[:min(self.N, len(m))]
                current_m_coeffs = m_padded
            else:
                current_m_coeffs = np.array(m, dtype=np.int64)

        current_r_coeffs = self.r
        if fixed_r is not None:
            current_r_coeffs = np.array(fixed_r, dtype=np.int64)

        if self.debug:
            self.logger.debug("\n--- ORIGINAL SYMPY ENCRYPT (ИСПРАВЛЕННАЯ ЛОГИКА) ---")
            self.logger.debug(f"N={self.N}, p={self.p}, q={self.q}")
            self.logger.debug(f"h (first 10): {self.h[:10]}")
            self.logger.debug(f"r (fixed, first 10): {current_r_coeffs[:10]}")
            self.logger.debug(f"m (fixed, first 10): {current_m_coeffs[:10]}")

        x = symbols('x')
        # Создаем полиномы в Z[x] (коэффициенты - обычные целые)
        # Помним про разворот для sympy: [a0,a1,...] -> [...,a1,a0]
        P_r_ZZ = Poly(current_r_coeffs[::-1], x, domain=ZZ)
        P_h_ZZ = Poly(self.h[::-1], x, domain=ZZ)
        P_m_ZZ = Poly(current_m_coeffs[::-1], x, domain=ZZ)

        I_list_coeffs_sympy_order = [1] + [0] * (self.N - 1) + [-1]
        P_I_ZZ = Poly(I_list_coeffs_sympy_order, x, domain=ZZ)

        # Шаг 1: r*h (линейная свёртка в Z[x])
        rh_poly_full_ZZ = P_r_ZZ * P_h_ZZ
        if self.debug: self.logger.debug(f"Sympy rh_poly_full (Z[x], sympy order): {rh_poly_full_ZZ.all_coeffs()[:10]}")

        # Шаг 2: (r*h)_coeffs % q
        term1_coeffs_mod_q_sympy_order = [c % self.q for c in rh_poly_full_ZZ.all_coeffs()]
        P_term1_mod_q = Poly(term1_coeffs_mod_q_sympy_order, x, domain=ZZ)  # Временно в ZZ
        if self.debug: self.logger.debug(
            f"Sympy term1_coeffs_mod_q ([0,q-1], sympy order): {P_term1_mod_q.all_coeffs()[:10]}")

        # Шаг 3: ... + m
        sum_intermediate = P_term1_mod_q + P_m_ZZ
        if self.debug: self.logger.debug(f"Sympy sum_intermediate (sympy order): {sum_intermediate.all_coeffs()[:10]}")

        # Шаг 4: ... % (x^N-1)
        sum_coeffs_reduced_mod_I = rem(sum_intermediate, P_I_ZZ, domain=ZZ)
        if self.debug: self.logger.debug(
            f"Sympy sum_coeffs_reduced_mod_I (after mod I, sympy order): {sum_coeffs_reduced_mod_I.all_coeffs()[:10]}")

        # Шаг 5: финальное ... % q
        e_mod_q_positive_coeffs_sympy_order = [c % self.q for c in sum_coeffs_reduced_mod_I.all_coeffs()]
        if self.debug: self.logger.debug(
            f"Sympy e_mod_q_positive ([0,q-1], sympy order): {e_mod_q_positive_coeffs_sympy_order[:10]}")

        # Шаг 6: Центрирование e
        e_centered_coeffs_sympy_order = []
        for c_val in e_mod_q_positive_coeffs_sympy_order:
            if 2 * c_val >= self.q:
                e_centered_coeffs_sympy_order.append(c_val - self.q)
            else:
                e_centered_coeffs_sympy_order.append(c_val)

        # Преобразуем в NumPy массив [a0, ..., aN-1]
        self.e = np.zeros(self.N, dtype=np.int64)  # Используем int64 для согласованности
        num_coeffs = len(e_centered_coeffs_sympy_order)
        # Дополняем старшие нули, если нужно
        temp_coeffs_padded = [0] * (self.N - num_coeffs) + e_centered_coeffs_sympy_order
        self.e = np.array(temp_coeffs_padded[::-1], dtype=np.int64)  # Разворачиваем

        if self.debug: self.logger.debug(f"Sympy self.e (ЦЕНТРИРОВАННЫЙ, [a0..aN-1]): {self.e[:10]}")

        return self.e  # Возвращаем массив для отладки

    def encryptString(self, M):
        """
        Encrypt the input string M by first converting to binary.

        NOTE : The public key must have been read before running this routine.
        """
        if not self.readKey:
            # Лучше использовать исключения, чем sys.exit, но оставим для консистентности
            sys.exit("Error : Not read the public key file, so cannot encrypt")

        # 1. Преобразуем входную строку M в массив битов
        bM = str2bit(M)

        # 2. Обрабатываем случай пустой строки
        current_len = len(bM)
        if current_len == 0:
            self.logger.warning("Attempting to encrypt an empty string. Result will be empty.")
            self.Me = ""
            return

        # 3. Вычисляем новую длину, кратную N, и дополняем нулями СПРАВА
        # Например, если N=107 и current_len=32 (для "test"), target_len будет 107.
        # Если current_len=200, target_len будет 214 (2 * 107).
        # Если current_len=107, target_len будет 107.
        if current_len % self.N == 0:
            target_len = current_len
        else:
            target_len = ((current_len // self.N) + 1) * self.N

        pad_len_right = target_len - current_len

        # Используем np.pad для дополнения нулями справа.
        # Кортеж (0, pad_len_right) означает "добавить 0 элементов слева, pad_len_right элементов справа".
        bM_padded = np.pad(bM, (0, pad_len_right), 'constant', constant_values=0)

        # 4. Инициализируем строку для результата
        self.Me = ""
        encrypted_parts = []  # Собирать части в список эффективнее, чем конкатенация строк

        # 5. Итерируем по блокам и шифруем
        num_blocks = len(bM_padded) // self.N
        for i in range(num_blocks):
            self.genr()  # Генерируем новый случайный r для каждого блока

            # Берем текущий блок сообщения для шифрования
            message_block = bM_padded[i * self.N: (i + 1) * self.N]

            # Устанавливаем и шифруем. setM и encrypt работают с self.m и self.r
            self.setM(message_block)
            self.encrypt()  # Результат будет в self.e

            # Добавляем результат в список
            encrypted_parts.append(arr2str(self.e))

        # 6. Собираем финальную строку шифротекста
        self.Me = " ".join(encrypted_parts)  # Соединяем все блоки через пробел