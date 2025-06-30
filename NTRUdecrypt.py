import time
import logging
import numpy as np
from math import log, gcd
import sys
from sympy import Poly, symbols, GF
from sympy.polys.domains import ZZ
from utils import checkPrime, poly_inv, genRand10, factor_int, padArr, bit2str_corrected # Убедитесь, что все импортированы

class NTRUdecrypt:
    """
    Класс для расшифровки и генерации ключей с использованием Sympy.
    """
    def __init__(self, logger, N=503, p=3, q=256, df=61, dg=20, d=18, debug=True, check_time=False):
        self.logger = logger # Инициализируем логгер в первую очередь
        self.N = N
        self.p = p
        self.q = q
        self.df = df
        self.dg = dg
        self.dr = d
        self.debug = debug
        self.check_time = check_time

        self.f = np.zeros((self.N,), dtype=np.int64)
        self.fp = np.zeros((self.N,), dtype=np.int64)
        self.fq = np.zeros((self.N,), dtype=np.int64)
        self.g = np.zeros((self.N,), dtype=np.int64)
        self.h = np.zeros((self.N,), dtype=np.int64)

        self.I = np.zeros((self.N + 1,), dtype=np.int64)
        self.I[self.N] = -1
        self.I[0] = 1

        self.M = None

        if self.debug:
            self.logger.debug(f"Initialized NTRUdecrypt with parameters: N={N}, p={p}, q={q}, df={df}, dg={dg}, d={d}")

    @staticmethod
    def time_function(func):
        def wrapper(self, *args, **kwargs):
            if self.check_time:
                start_time = time.perf_counter() # Используем perf_counter для большей точности
            result = func(self, *args, **kwargs)
            if self.check_time:
                end_time = time.perf_counter()
                # Предполагая, что ваш кастомный logger имеет метод time()
                # Если нет, замените на self.logger.debug или info
                try:
                    self.logger.time(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
                except AttributeError:
                    self.logger.debug(f"[TIME] Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
            return result
        return wrapper

    # @time_function
    def setNpq(self, N=None, p=None, q=None, df=None, dg=None, d=None):
        """
        Set the N, p and q values and perform checks on their validity.
        """
        if N is not None:
            if not checkPrime(N):
                sys.exit("\n\nERROR: Input value of N not prime\n\n")
            else:
                if df is None and 2 * self.df > N:
                    sys.exit("\n\nERROR: Input N too small compared to default df " + str(self.df) + "\n\n")
                if dg is None and 2 * self.dg > N:
                    sys.exit("\n\nERROR: Input N too small compared to default dg " + str(self.dg) + "\n\n")
                if d is None and 2 * self.dr > N:
                    sys.exit("\n\nERROR: Input N too small compared to default dr " + str(self.dr) + "\n\n")
                self.N = N
                self.reset_polynomials()

        if (p is None and q is not None) or (p is not None and q is None):
            sys.exit("\n\nError: Can only set p and q together, not individually")
        elif (p is not None) and (q is not None):
            if ((8 * p) > q):
                sys.exit("\n\nERROR: We require 8p <= q\n\n")
            elif (gcd(p, q) != 1):
                sys.exit("\n\nERROR: Input p and q are not coprime\n\n")
            else:
                self.p = p
                self.q = q

        if df is not None:
            if 2 * df > self.N:
                sys.exit("\n\nERROR: Input df such that 2*df>N\n\n")
            else:
                self.df = df

        if dg is not None:
            if 2 * dg > self.N:
                sys.exit("\n\nERROR: Input dg such that 2*dg>N\n\n")
            else:
                self.dg = dg

        if d is not None:
            if 2 * d > self.N:
                sys.exit("\n\nERROR: Input dr such that 2*dr>N\n\n")
            else:
                self.dr = d

        if self.debug:
            self.logger.debug("setNpq called with parameters: N={}, p={}, q={}, df={}, dg={}, d={}".format(
                self.N, self.p, self.q, self.df, self.dg, self.dr))

    def reset_polynomials(self):
        """ Reset polynomial arrays after changing N """
        self.f = np.zeros((self.N,), dtype=np.int64)
        self.fp = np.zeros((self.N,), dtype=np.int64)
        self.fq = np.zeros((self.N,), dtype=np.int64)
        self.g = np.zeros((self.N,), dtype=np.int64)
        self.h = np.zeros((self.N,), dtype=np.int64)
        self.I = np.zeros((self.N + 1,), dtype=np.int64)
        self.I[self.N] = -1
        self.I[0] = 1

    @time_function
    def invf(self):
        """
        Invert the f polynomial.
        """
        # --- ИСПРАВЛЕНИЕ: Убираем дублирующиеся вызовы и принты ---
        if self.debug:
            self.logger.debug(f"DEBUG: self.f для poly_inv (p={self.p}): {self.f[:10]}")

        fp_tmp = poly_inv(self.f, self.I, self.p)
        if self.debug:
            self.logger.debug(f"DEBUG: fp_tmp от poly_inv (p={self.p}): {fp_tmp[:10] if len(fp_tmp) > 0 else 'ПУСТОЙ'}")

        fq_tmp = poly_inv(self.f, self.I, self.q)
        if self.debug:
            self.logger.debug(f"DEBUG: fq_tmp от poly_inv (q={self.q}): {fq_tmp[:10] if len(fq_tmp) > 0 else 'ПУСТОЙ'}")

        if fp_tmp.size > 0 and fq_tmp.size > 0:  # Используем .size для numpy массивов
            # poly_inv теперь должен возвращать массив нужной длины N
            self.fp = fp_tmp
            self.fq = fq_tmp
            return True
        else:
            return False


    @time_function
    def genfg(self):
        """
        Randomly generate f and g for the private key and their inverses.
        """
        maxTries = 100
        self.g = genRand10(self.N, self.dg, self.dg)

        for i in range(maxTries):
            self.f = genRand10(self.N, self.df, self.df - 1)

            invStat = self.invf()
            if invStat:
                break
            elif i == maxTries - 1:
                sys.exit("Cannot generate required inverses of f")

    # @time_function
    def genh(self):
        """
        Generate the public key from the class values (that must have been generated previously).
        """
        x = symbols('x')
        while True:
            self.h = Poly((Poly(self.p * self.fq, x).trunc(self.q) * Poly(self.g, x)).trunc(self.q) \
                          % Poly(self.I, x)).all_coeffs()

            if len(factor_int(self.h[-1])) == 0:
                break
            self.genfg()

    # @time_function
    def writePub(self, filename="key"):
        """
        Write the public key file.
        """
        pubHead = "p ::: " + str(self.p) + "\nq ::: " + str(self.q) + "\nN ::: " + str(self.N) \
                  + "\nd ::: " + str(self.dr) + "\nh :::"
        np.savetxt(filename + ".pub", self.h, newline=" ", header=pubHead, fmt="%s")

    # @time_function
    def readPub(self, filename="key.pub"):
        """
        Read a public key file.
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

    # @time_function
    def writePriv(self, filename="key"):
        """
        Write the private key file.
        """
        privHead = "p ::: " + str(self.p) + "\nq ::: " + str(self.q) + "\nN ::: " \
                   + str(self.N) + "\ndf ::: " + str(self.df) + "\ndg ::: " + str(self.dg) \
                   + "\nd ::: " + str(self.dr) + "\nf/fp/fq/g :::"
        np.savetxt(filename + ".priv", (self.f, self.fp, self.fq, self.g), header=privHead, newline="\n", fmt="%s")

    # @time_function
    def readPriv(self, filename="key.priv"):
        """
        Read a public key file.
        """
        with open(filename, "r") as f:
            self.p = int(f.readline().split(" ")[-1])
            self.q = int(f.readline().split(" ")[-1])
            self.N = int(f.readline().split(" ")[-1])
            self.df = int(f.readline().split(" ")[-1])
            self.dg = int(f.readline().split(" ")[-1])
            self.dr = int(f.readline().split(" ")[-1])
            f.readline()
            f_str = f.readline().strip().split(" ")
            fp_str = f.readline().strip().split(" ")
            fq_str = f.readline().strip().split(" ")
            g_str = f.readline().strip().split(" ")

            self.f = np.array([int(s) for s in f_str if s], dtype=np.int64)
            self.fp = np.array([int(s) for s in fp_str if s], dtype=np.int64)
            self.fq = np.array([int(s) for s in fq_str if s], dtype=np.int64)
            self.g = np.array([int(s) for s in g_str if s], dtype=np.int64)

        if len(self.f) < self.N: self.f = np.pad(self.f, (0, self.N - len(self.f)), 'constant')
        if len(self.fp) < self.N: self.fp = np.pad(self.fp, (0, self.N - len(self.fp)), 'constant')

        self.I = np.zeros((self.N + 1,), dtype=np.int64)
        self.I[self.N] = -1
        self.I[0] = 1

        x_check = symbols('x')
        domain_zp_check = GF(self.p, symmetric=True) if self.p == 3 else GF(self.p, symmetric=False)
        P_f_check = Poly(self.f[::-1], x_check, domain=domain_zp_check)
        P_fp_check = Poly(self.fp[::-1], x_check, domain=domain_zp_check)
        I_check = Poly(self.I, x_check, domain=domain_zp_check)

        prod_check = (P_f_check * P_fp_check) % I_check

        expected_identity_coeffs = np.zeros(self.N, dtype=int)
        expected_identity_coeffs[0] = 1

        prod_coeffs_np = np.zeros(self.N, dtype=int)
        prod_coeffs_sympy = prod_check.all_coeffs()
        prod_coeffs_np_a0_order = prod_coeffs_sympy[::-1]
        prod_coeffs_np[:len(prod_coeffs_np_a0_order)] = prod_coeffs_np_a0_order

        if not np.array_equal(prod_coeffs_np, expected_identity_coeffs):
            self.logger.warning("Проверка в readPriv: fp НЕ ЯВЛЯЕТСЯ ОБРАТНЫМ к f по модулю p!")
        else:
            self.logger.debug("Проверка в readPriv: fp является обратным к f по модулю p - OK.")

    # @time_function
    def genPubPriv(self, keyfileName="key"):
        """
        Generate the public and private keys from class N, p and q values.
        Also write output files for the public and private keys.
        """
        self.genfg()
        self.genh()
        self.writePub(keyfileName)
        self.writePriv(keyfileName)

    # @time_function
    def decrypt(self, e_input_coeffs_centered):  # Ожидаем центрированный e на входе
        if len(e_input_coeffs_centered) != self.N:
            if len(e_input_coeffs_centered) > self.N:
                sys.exit(f"Encrypted message has degree > N (len={len(e_input_coeffs_centered)}, N={self.N})")

        if self.debug:
            self.logger.debug("--- ORIGINAL SYMPY DECRYPT ---")
            self.logger.debug(f"Input e (ЦЕНТРИРОВАННЫЙ, a0-aN-1): {e_input_coeffs_centered[:10]}")
            self.logger.debug(f"f (a0-aN-1): {self.f[:10]}")
            self.logger.debug(f"fp (a0-aN-1): {self.fp[:10]}")
            self.logger.debug(f"N={self.N}, p={self.p}, q={self.q}")

        x = symbols('x')
        P_f_ZZ = Poly(self.f[::-1], x, domain=ZZ)
        P_e_centered_ZZ = Poly(e_input_coeffs_centered[::-1], x, domain=ZZ)
        I_list_coeffs_sympy_order = [1] + [0] * (self.N - 1) + [-1]
        P_I_ZZ = Poly(I_list_coeffs_sympy_order, x, domain=ZZ)  # Идеал в ZZ

        # Шаг 1 D: a_poly = (f * e_centered) % I (в Z[x])
        a_poly_in_Z = (P_f_ZZ * P_e_centered_ZZ) % P_I_ZZ
        if self.debug:
            self.logger.debug(f"Sympy Decrypt: a_poly_in_Z (coeffs in Z, sympy order): {a_poly_in_Z.all_coeffs()[:10]}")

        # Шаг 2 D: Коэффициенты 'a' по модулю q, а затем ЦЕНТРИРОВАНИЕ
        coeffs_a_mod_q_list_sympy_order = [c % self.q for c in a_poly_in_Z.all_coeffs()]
        if self.debug:
            self.logger.debug(f"Sympy Decrypt: coeffs_a_mod_q_list ([0,q-1], sympy order): {coeffs_a_mod_q_list_sympy_order[:10]}")

        a_coeffs_truly_centered_sympy_order = []
        for c_val in coeffs_a_mod_q_list_sympy_order:
            if 2 * c_val >= self.q:
                a_coeffs_truly_centered_sympy_order.append(c_val - self.q)
            else:
                a_coeffs_truly_centered_sympy_order.append(c_val)
        if self.debug:
            self.logger.debug(f"Sympy Decrypt: a_coeffs_truly_centered (centered mod q, sympy order): {a_coeffs_truly_centered_sympy_order[:10]}")

        P_a_truly_centered_ZZ = Poly(a_coeffs_truly_centered_sympy_order, x, domain=ZZ)

        if self.p == 3:
            domain_for_p_ops = GF(self.p, symmetric=True)
        else:
            domain_for_p_ops = GF(self.p, symmetric=False)

        b_poly_mod_p = Poly(P_a_truly_centered_ZZ, domain=domain_for_p_ops)
        b_coeffs_sympy_order = b_poly_mod_p.all_coeffs()
        if self.debug:
            self.logger.debug(f"Sympy Decrypt: b_coeffs (domain_for_p_ops, sympy order): {b_coeffs_sympy_order[:10]}")

        P_fp_ZZ = Poly(self.fp[::-1], x, domain=ZZ)
        P_fp_mod_p = Poly(P_fp_ZZ, domain=domain_for_p_ops)  # fp тоже приводим к домену Z_p

        c_poly_intermediate = (P_fp_mod_p * b_poly_mod_p)  # Умножение в Z_p[x]

        P_I_mod_p = Poly(I_list_coeffs_sympy_order, x, domain=domain_for_p_ops)  # <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ

        c_poly = c_poly_intermediate % P_I_mod_p  # <--- ИСПОЛЬЗУЕМ P_I_mod_p

        m_prime_coeffs_sympy_order = c_poly.all_coeffs()
        m_prime_np = np.zeros(self.N, dtype=int)
        num_coeffs = len(m_prime_coeffs_sympy_order)  # Длина списка от sympy (может быть < N)
        for i in range(num_coeffs):
            idx_in_np_array = num_coeffs - 1 - i
            if idx_in_np_array < self.N:  # Защита, если num_coeffs > N (не должно быть)
                m_prime_np[idx_in_np_array] = m_prime_coeffs_sympy_order[i]

        if self.debug:
            self.logger.debug(f"Sympy Decrypt: m_prime_np (ИСПРАВЛЕННОЕ ПРЕОБРАЗОВАНИЕ, [a0..aN-1]): {m_prime_np[:10]}")
            self.logger.debug(f"Sympy Decrypt: m_prime_coeffs (domain_for_p_ops, sympy order): {m_prime_coeffs_sympy_order[:10]}")

        m_prime_np = np.zeros(self.N, dtype=int)
        num_coeffs = len(m_prime_coeffs_sympy_order)
        temp_coeffs_sympy_order_padded = [0] * (self.N - num_coeffs) + list(m_prime_coeffs_sympy_order)
        m_prime_np = np.array(temp_coeffs_sympy_order_padded[::-1], dtype=int)
        if self.debug:
            self.logger.debug(f"Sympy Decrypt: m_prime_np (из domain_for_p_ops, [a0..aN-1]): {m_prime_np[:10]}")

        if np.any((m_prime_np != 0) & (m_prime_np != 1)):
            self.logger.warning(
                f"m_prime_np содержит значения, отличные от 0 и 1: {m_prime_np[(m_prime_np != 0) & (m_prime_np != 1)]}")

        return m_prime_np

    # @time_function
    def decryptString(self, E):
        """
        Decrypt a message encoded using the requisite public key from an encoded to a decoded string.
        """
        Me = np.fromstring(E, dtype=np.int64, sep=' ')  # Используем int64
        if np.mod(len(Me), self.N) != 0:
            raise ValueError(f"Длина шифротекста {len(Me)} не кратна N={self.N}")

        Marr = np.array([], dtype=np.int64)  # Создаем Marr один раз
        num_blocks = len(Me) // self.N

        if self.debug:
            self.logger.debug(f"decryptString: Количество блоков в шифротексте: {num_blocks}")

        # --- ОДИН ЦИКЛ ПО БЛОКАМ ---
        for i in range(num_blocks):
            e_block = Me[i * self.N:(i + 1) * self.N]
            if self.debug:
                self.logger.debug(f"decryptString: Блок {i + 1}, входной e_block (первые 10): {e_block[:10]}")

            decrypted_block = self.decrypt(e_block)
            if self.debug:
                self.logger.debug(f"decryptString: Блок {i + 1}, decrypted_block (первые 35): {decrypted_block[:35]}")

            # Конкатенируем результат расшифровки блока к Marr
            # self.decrypt возвращает массив длины N, padArr не нужен
            Marr = np.concatenate((Marr, decrypted_block))

        if self.debug:
            self.logger.debug(f"DEBUG decryptString: Финальный Marr ПЕРЕД bit2str (первые 40): {Marr[:40]}")
            self.logger.debug(f"DEBUG decryptString: Длина финального Marr: {len(Marr)}")

        self.M = bit2str_corrected(Marr)  # Используем исправленную версию

        if self.debug:
            self.logger.debug(f"DEBUG decryptString: self.M ПОСЛЕ bit2str: '{self.M}'")
