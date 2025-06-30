import logging
import math
import time
import numpy as np
from math import log
import sys
from sympy import Poly, symbols, GF, invert, div # <--- Добавьте div
from sympy.polys.domains import ZZ
np.set_printoptions(threshold=sys.maxsize)


def factor_int(n):
    """
    Return a dictionary of the prime factorization of n.
    The keys of the dictionary are the prime factors of n, and the values are their
    multiplicities (i.e. the power to which each prime factor appears in the factorization).
    """
    factors_ = {}
    d = 2
    while n > 1:
        while n % d == 0:
            if d in factors_:
                factors_[d] += 1
            else:
                factors_[d] = 1
            n //= d
        d += 1
        if d*d > n:
            if n > 1:
                if n in factors_:
                    factors_[n] += 1
                else:
                    factors_[n] = 1
            break
    return factors_


def checkPrime(P):
    a = time.time()
    """
    Check if the input integer P is prime, if prime return True
    else return False.
    """
    if P <= 1:
        return False
    elif P == 2 or P == 3:
        return True
    else:
        # Otherwise, check if P is dividable by any value over 4 and under P/2
        for i in range(4, P // 2):
            if P % i == 0:
                return False

    return True


def poly_inv_3(poly_in, poly_I_coeffs, poly_mod):  # poly_I_coeffs - это список коэффициентов [1,0,...-1]
    """
    Find the inverse of the polynomial poly_in in the Galois filed GF(poly_mod)
    i.e. the inverse in
        Z/poly_mod[X]/poly_I

    Inputs and outputs are given as an array of coefficients where
        x^4 + 5x^2 + 3 == [1,0,5,0,3]

    Returns
    =======
    Either an empty array if the inverse cannot be found, or the inverse of the
    polynomial poly_in as an array of coefficients.

    References
    ==========
    https://arxiv.org/abs/1311.1779
    """
    x = symbols('x')
    final_domain_gf = GF(poly_mod, symmetric=False)
    domain_zz = ZZ
    Ppoly_I_expr_zz = Poly(poly_I_coeffs, x, domain=domain_zz).as_expr()
    Ppoly_I_poly_final_gf = Poly(poly_I_coeffs, x, domain=final_domain_gf)
    poly_in_expr_zz = Poly(poly_in, x, domain=domain_zz).as_expr()
    poly_in_poly_final_gf = Poly(poly_in, x, domain=final_domain_gf)

    inv_result_poly = None

    if checkPrime(poly_mod):
        try:
            inv_expr_gf = invert(poly_in_expr_zz, Ppoly_I_expr_zz, domain=final_domain_gf)
            inv_result_poly = Poly(inv_expr_gf, x, domain=final_domain_gf)
        except Exception:
            return np.array([])
    elif log(poly_mod, 2).is_integer():
        try:
            domain_gf2 = GF(2, symmetric=False)
            poly_in_poly_gf2 = Poly(poly_in, x, domain=domain_gf2)
            inv_expr_gf2 = invert(poly_in_poly_gf2.as_expr(), Ppoly_I_expr_zz, domain=domain_gf2)

            current_inv_coeffs_int = Poly(inv_expr_gf2, x, domain=domain_gf2).all_coeffs()

            ex = int(log(poly_mod, 2))
            for _ in range(1, ex):
                inv_iter_poly_zz = Poly(current_inv_coeffs_int, x, domain=domain_zz)
                poly_in_iter_poly_zz = Poly(poly_in, x, domain=domain_zz)
                Ppoly_I_iter_poly_zz = Poly(poly_I_coeffs, x, domain=domain_zz)

                next_inv_stage_zz = (2 * inv_iter_poly_zz - poly_in_iter_poly_zz * inv_iter_poly_zz ** 2) % Ppoly_I_iter_poly_zz

                truncated_poly_zz = next_inv_stage_zz.trunc(poly_mod)
                current_inv_coeffs_int = truncated_poly_zz.all_coeffs()

            inv_result_poly = Poly(current_inv_coeffs_int, x, domain=final_domain_gf)
        except Exception:
            return np.array([])
    else:
        return np.array([])

    if inv_result_poly is None:
        return np.array([])

    product_to_divide = inv_result_poly * poly_in_poly_final_gf
    try:
        _q, check_product = div(product_to_divide, Ppoly_I_poly_final_gf, domain=final_domain_gf)
    except Exception as e:
        sys.exit(f"ERROR : Error in polynomial division for check_product. Details: {e}")

    tmpCheck_coeffs = check_product.all_coeffs()

    tmpCheck = np.array([int(c) for c in tmpCheck_coeffs], dtype=int)

    if not (len(tmpCheck) == 1 and tmpCheck[0] == 1):
        sys.exit(
            f"ERROR : Error in calculation of polynomial inverse. Result of (inv*f)%I was not 1. Coeffs: {tmpCheck}")

    output_len = Ppoly_I_poly_final_gf.degree()  # Это N
    final_coeffs_int = [int(c) for c in inv_result_poly.all_coeffs()]
    return padArr(np.array(final_coeffs_int, dtype=int), output_len)


def poly_inv_2(poly_in, poly_I, poly_mod):
    x = symbols('x')
    Ppoly_I = Poly(poly_I, x)
    Npoly_I = len(Ppoly_I.all_coeffs())

    # Pre-compute domain
    if checkPrime(poly_mod):
        domain = GF(poly_mod, symmetric=False)
    elif log(poly_mod, 2).is_integer():
        domain = GF(2, symmetric=False)
    else:
        return np.array([])

    # Convert input polynomial once
    poly_in_expr = Poly(poly_in, x).as_expr()
    poly_I_expr = Ppoly_I.as_expr()

    try:
        if checkPrime(poly_mod):
            inv = invert(poly_in_expr, poly_I_expr, domain=domain)
        else:  # Power of 2 case
            inv = invert(poly_in_expr, poly_I_expr, domain=domain)


            # Optimize the iteration loop
            inv_poly = Poly(inv, x)
            poly_in_poly = Poly(poly_in, x)

            result = inv_poly
            exponent = int(log(poly_mod, 2))

            while exponent > 0:
                if exponent % 2 == 1:  # wenn exponent ungerade ist
                    result = ((2 * result - poly_in_poly * (result ** 2)) % Ppoly_I).trunc(poly_mod)
                inv_poly = (inv_poly ** 2) % Ppoly_I  # Quadrat des aktuellen inversen Polynoms
                exponent //= 2  # halbiere den Exponenten

            inv = result
    except:
        return np.array([])

    # Verification step
    result_poly = Poly((Poly(inv, x) * Poly(poly_in, x)) % Ppoly_I, domain=domain)
    tmpCheck = np.array(result_poly.all_coeffs(), dtype=int)

    if len(tmpCheck) > 1 or tmpCheck[0] != 1:
        raise ValueError("Error in calculation of polynomial inverse")

    return padArr(np.array(Poly(inv, x).all_coeffs(), dtype=int), Npoly_I - 1)


def poly_inv(poly_in, poly_I, poly_mod):  # poly_in это self.f, poly_I это self.I из NTRUdecrypt
    """
    Find the inverse of the polynomial poly_in in the Galois filed GF(poly_mod)
    i.e. the inverse in
        Z/poly_mod[X]/poly_I

    Inputs:
        poly_in: NumPy array [a0, ..., aN-1]
        poly_I: NumPy array [1, 0, ..., 0, -1] (представляет x^N - 1)
        poly_mod: integer (p или q)
    Returns:
        NumPy array [inv_a0, ..., inv_aN-1] длины N, или пустой массив если инверсия не удалась.
    """
    x = symbols('x')

    poly_in_sympy_order = poly_in[::-1].tolist()
    poly_I_sympy_coeffs = poly_I.tolist()

    inv_expr = None
    if checkPrime(poly_mod):
        try:
            inv_expr = invert(Poly(poly_in_sympy_order, x).as_expr(),
                              Poly(poly_I_sympy_coeffs, x).as_expr(),
                              domain=GF(poly_mod, symmetric=False))  # Используем symmetric=False для [0, p-1]
        except Exception as e:  # sympy.NotInvertible или другие ошибки
            print(f"DEBUG utils.poly_inv: Ошибка invert для простого poly_mod={poly_mod}: {e}")
            return np.array([])
    elif log(poly_mod, 2).is_integer():  # Случай q = 2^k
        try:
            # Процедура Hensel's Lemma / Newton iteration для поднятия из GF(2)
            # Сначала инверсия в GF(2)[x]/(I)
            inv_gf2_expr = invert(Poly(poly_in_sympy_order, x).as_expr(),
                                  Poly(poly_I_sympy_coeffs, x).as_expr(),
                                  domain=GF(2, symmetric=False))

            P_inv_current = Poly(inv_gf2_expr, x)  # Текущий инверс
            P_poly_in_for_lift = Poly(poly_in_sympy_order, x)  # Исходный полином f
            P_ideal_for_lift = Poly(poly_I_sympy_coeffs, x)  # Идеал

            exponent = int(log(poly_mod, 2))
            for _ in range(1, exponent):  # Поднимаем от GF(2) -> GF(4) -> GF(8) ... -> GF(2^exponent)

                # Создаем полиномы с коэффициентами в Z для итерации подъема
                P_inv_current_ZZ = Poly(P_inv_current.all_coeffs(), x, domain=ZZ)  # Коэфф. из {0,1} -> Z
                P_poly_in_ZZ = Poly(poly_in_sympy_order, x, domain=ZZ)
                P_ideal_ZZ = Poly(poly_I_sympy_coeffs, x, domain=ZZ)

                P_inv_current = ((2 * P_inv_current_ZZ - P_poly_in_ZZ * (P_inv_current_ZZ ** 2)) % P_ideal_ZZ).trunc(
                    poly_mod)

            inv_expr = P_inv_current  # Это уже объект Poly

        except Exception as e:
            print(f"DEBUG utils.poly_inv: Ошибка invert/lift для q=2^k, poly_mod={poly_mod}: {e}")
            return np.array([])
    else:
        print(f"DEBUG utils.poly_inv: poly_mod={poly_mod} не простое и не степень 2. Инверсия не поддерживается этим кодом.")
        return np.array([])

    if inv_expr is None:
        return np.array([])

    x = symbols('x')
    poly_in_sympy_order = poly_in[::-1].tolist()
    poly_I_sympy_coeffs = poly_I.tolist()
    P_poly_in_ZZ_check = Poly(poly_in_sympy_order, x, domain=ZZ)
    P_I_ZZ_check = Poly(poly_I_sympy_coeffs, x, domain=ZZ)

    if not isinstance(inv_expr, Poly):
        P_inv_to_check_ZZ = Poly(inv_expr, x, domain=ZZ)
    else:
        P_inv_to_check_ZZ = Poly(inv_expr.all_coeffs(), x, domain=ZZ)

    prod_in_ZZ = P_inv_to_check_ZZ * P_poly_in_ZZ_check
    rem_in_ZZ = prod_in_ZZ % P_I_ZZ_check

    coeffs_of_rem_mod_poly_mod_sympy_order = [c % poly_mod for c in rem_in_ZZ.all_coeffs()]

    expected_poly_1_coeffs_sympy_order_len_N = [0] * (len(poly_in) - 1) + [1 % poly_mod]

    num_actual_coeffs = len(coeffs_of_rem_mod_poly_mod_sympy_order)
    actual_coeffs_padded_sympy_order = [0] * (len(poly_in) - num_actual_coeffs) + coeffs_of_rem_mod_poly_mod_sympy_order

    is_correct_inv = False
    if actual_coeffs_padded_sympy_order == expected_poly_1_coeffs_sympy_order_len_N:
        is_correct_inv = True

    if not is_correct_inv:
        print(f"!!! ОТЛАДКА utils.poly_inv: Проверка f*fp != 1 для poly_mod={poly_mod} НЕ ПРОШЛА !!!")
        return np.array([])

    if not isinstance(inv_expr, Poly):
        P_inv_final = Poly(inv_expr, x)
    else:
        P_inv_final = inv_expr

    if poly_mod == 3:
        P_inv_to_return = Poly(P_inv_final, domain=GF(poly_mod, symmetric=True))
    else:
        P_inv_to_return = Poly(P_inv_final, domain=GF(poly_mod, symmetric=False))

    inv_coeffs_sympy_order = P_inv_to_return.all_coeffs()

    output_coeffs = np.zeros(len(poly_in), dtype=int)
    num_c = len(inv_coeffs_sympy_order)

    coeffs_a0_order_from_sympy = inv_coeffs_sympy_order[::-1]
    len_to_copy = min(len(output_coeffs), len(coeffs_a0_order_from_sympy))
    output_coeffs[:len_to_copy] = coeffs_a0_order_from_sympy[:len_to_copy]

    return output_coeffs

def padArr(A_in, A_out_size):
    """
    Take an input numpy integer array A_in and pad with leading zeros.
    Return the numpy array of size A_out_size with leading zeros
    """
    pad_len_left = A_out_size - len(A_in)
    if pad_len_left < 0:
        return A_in[:A_out_size]  # Обрезаем, если длиннее
    return np.pad(A_in, (pad_len_left, 0), 'constant', constant_values=0)


def genRand10(L, P, M):
    """
    Generate a numpy array of length L with P 1's, M -1's and the remaining elements 0.
    The elements will be in a random order, with randomisation done using np.random.shuffle.
    This is used to generate the f, p and r arrays for NTRU encryption based on [1].

    INPUTS:
    =======
    L : Integer, the length of the desired output array.
    P : Integer, the number of `positive' i.e. +1's in the array.
    M : Integer, the number of `negative' i.e. -1's in the array.

    RETURNS:
    ========
    An integer numpy array with P +1's, M -1's and L-P-M 0's.

    REFERENCES:
    ===========
    [1] Hoffstein J, Pipher J, Silverman JH. NTRU: A Ring-Based Public Key Cryptosystem.
        Algorithmic Number Theory. 1998; 267--288.
    """
    if P + M > L:
        sys.exit("ERROR: Asking for P+M>L.")

    R = np.zeros((L,), dtype=int)

    for i in range(L):
        if i < P:
            R[i] = 1
        elif i < P + M:
            R[i] = -1
        else:
            break

    np.random.shuffle(R)
    return R


def arr2str(ar):
    """
    Convert a numpy array to a string containing only the elements of the array.

    INPUTS:
    =======
    ar : Numpy array, elements will be concatenated and returned as string.

    RETURNS:
    ========
    A string containing all the elements of ar concatenated, each element separated by a space
    """
    st = np.array_str(ar)
    st = st.replace("[", "", 1)
    st = st.replace("]", "", 1)
    st = st.replace("\n", "")
    st = st.replace("     ", " ")
    st = st.replace("    ", " ")
    st = st.replace("   ", " ")
    st = st.replace("  ", " ")
    return st


def str2bit(st):
    try:
        encoded_str = str(st).encode('utf-8')
        int_val = int.from_bytes(encoded_str, byteorder="big")
        binary_representation = bin(int_val)[2:] # [2:] чтобы убрать "0b"
        expected_bit_length = len(encoded_str) * 8
        padded_binary_representation = binary_representation.zfill(expected_bit_length)
        return np.array(list(padded_binary_representation), dtype=int)
    except Exception as e_str2bit:
        print(f"ОШИБКА в str2bit для строки '{st}': {e_str2bit}")
        return np.array([], dtype=int) # Возвращаем пустой массив в случае ошибкиe=int)


def bit2str(bi):
    """
    Convert an array of bits to the string described by those bits.

    INPUTS:
    =======
    bi : Numpy integer array, containing only 1's and 0's. When flattened this represents a
         string (not including the "0b" prefix).

    RETURNS:
    ========
    A string, the binary values in the bi array converted to a string.
    """
    S = arr2str(bi)
    S = S.replace(" ", "")
    charOut = ""
    for i in range(len(S) // 8):
        if i == 0:
            charb = S[len(S) - 8:]
        else:
            charb = S[-(i + 1) * 8:-i * 8]

        # print(charb)
        charb = int(charb, 2)
        charOut = charb.to_bytes((charb.bit_length() + 7) // 8, "big").decode("utf-8", errors="ignore") + charOut

    return charOut


def bit2str_corrected(bi: np.ndarray) -> str:
    if not isinstance(bi, np.ndarray):  # Проверка типа
        return ""
    if bi.ndim > 1:
        bi = bi.flatten()

    try:
        s_list = [str(int(b_val)) for b_val in bi]
    except ValueError as e:
        return "ERROR_IN_CONVERSION"

    S_bits = "".join(s_list)
    charOut_list = []
    num_bits = len(S_bits)

    if num_bits < 8:
        return "".join(charOut_list)

    for i in range(0, num_bits - (num_bits % 8), 8):
        byte_str = S_bits[i: i + 8]

        try:
            byte_int = int(byte_str, 2)
        except ValueError:
            print(f"DEBUG bit2str_corrected: Не удалось преобразовать '{byte_str}' в int.")
            continue

        if byte_int == 0:
            break

        try:
            num_bytes_for_char = (byte_int.bit_length() + 7) // 8
            if num_bytes_for_char == 0 and byte_int == 0:
                num_bytes_for_char = 1
            elif num_bytes_for_char == 0 and byte_int != 0:

                num_bytes_for_char = 1
            decoded_char = byte_int.to_bytes(num_bytes_for_char, "big").decode("utf-8", errors="replace")

            charOut_list.append(decoded_char)
        except OverflowError:
            print(f"DEBUG bit2str_corrected: OverflowError для byte_int={byte_int}, num_bytes_for_char={num_bytes_for_char}")
        except UnicodeDecodeError:  # Это исключение теперь должно обрабатываться errors="replace"
            print(f"DEBUG bit2str_corrected: UnicodeDecodeError (не должно возникать с errors='replace') для байта {byte_str} (int {byte_int})")
            charOut_list.append('�')  # Символ замены по умолчанию
        except Exception as e:
            print(f"DEBUG bit2str_corrected: Общая ошибка декодирования байта {byte_str} (int {byte_int}): {e}")
            pass

    final_string = "".join(charOut_list)
    return final_string

def read_key_params_from_pub(filename_pub):
    try:
        with open(filename_pub, "r") as f:
            p_ = int(f.readline().split(" ")[-1])
            q_ = int(f.readline().split(" ")[-1])
            N_ = int(f.readline().split(" ")[-1])
            dr_ = int(f.readline().split(" ")[-1])
            h_line = f.readline().split(" ")
            h_start_index = 0
            for i, token in enumerate(h_line):
                if token == ":::":
                    h_start_index = i + 1
                    break

            h_np_ = np.array([s for s in h_line[h_start_index:] if s.strip() and s != '\n'], dtype=np.int64)

        if len(h_np_) < N_:
            h_np_ = np.pad(h_np_, (0, N_ - len(h_np_)), 'constant', constant_values=0)
        return h_np_[:N_], N_, p_, q_, dr_
    except FileNotFoundError:
        logger.error(f"Файл ключа {filename_pub} не найден.")
        exit()
    except Exception as e:
        logger.error(f"ОШИБКА при чтении файла {filename_pub}: {e}")
        import traceback
        traceback.print_exc()
        exit()
