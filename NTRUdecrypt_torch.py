# NTRUdecrypt_torch.py
import torch
import numpy as np
import sys
import time
from utils_torch import padArr_pytorch, multiply_poly_pytorch_circulant_mod_ideal_only
import utils


class NTRUdecryptTorch:
    """
    Класс для расшифровки данных с помощью NTRU, реализованный на PyTorch.
    Этот класс НЕ генерирует ключи, только загружает и использует их.
    """

    def __init__(self, logger, N=503, p=3, q=256, device='cpu', debug=True, check_time=False):
        self.logger = logger
        self.N = N
        self.p = p
        self.q = q
        self.device = torch.device(device)
        self.debug = debug
        self.check_time = check_time

        # Типы данных для вычислений и хранения
        self.compute_dtype = torch.int64
        self.storage_dtype = torch.int32

        # Инициализируем тензоры для приватного ключа
        self.f = torch.zeros(self.N, dtype=self.compute_dtype, device=self.device)
        self.fp = torch.zeros(self.N, dtype=self.compute_dtype, device=self.device)

        self.M_str = None  # Для хранения расшифрованной строки

        if self.debug:
            self.logger.debug(f"Initialized NTRUdecryptTorch with N={N}, p={p}, q={q}")

    def readPriv(self, filename="key.priv"):
        """
        Читает приватный ключ из файла.
        """
        try:
            with open(filename, "r") as f_priv:
                self.p = int(f_priv.readline().split(" ")[-1])
                self.q = int(f_priv.readline().split(" ")[-1])
                self.N = int(f_priv.readline().split(" ")[-1])
                # Пропускаем df, dg, dr, которые не нужны для расшифровки
                f_priv.readline()
                f_priv.readline()
                f_priv.readline()
                f_priv.readline()  # Пропускаем заголовок "f/fp/fq/g :::"

                # Читаем f и fp как NumPy массивы
                f_np = np.array([s for s in f_priv.readline().strip().split(" ") if s], dtype=np.int64)
                fp_np = np.array([s for s in f_priv.readline().strip().split(" ") if s], dtype=np.int64)

            # Преобразуем NumPy массивы в тензоры PyTorch
            self.f = torch.from_numpy(f_np).to(device=self.device, dtype=self.compute_dtype)
            self.fp = torch.from_numpy(fp_np).to(device=self.device, dtype=self.compute_dtype)

            # Дополняем нулями, если нужно
            if len(self.f) < self.N: self.f = padArr_pytorch(self.f, self.N)
            if len(self.fp) < self.N: self.fp = padArr_pytorch(self.fp, self.N)

        except FileNotFoundError:
            raise FileNotFoundError(f"Файл приватного ключа {filename} не найден.")
        except Exception as e:
            raise IOError(f"Ошибка при чтении файла {filename}: {e}")

    def decrypt_poly(self, e_input_coeffs_centered_tensor: torch.Tensor) -> torch.Tensor:
        """
        Расшифровывает один блок шифротекста (тензор) с использованием PyTorch.
        Воспроизводит "каноническую" логику NTRU.
        """
        if len(e_input_coeffs_centered_tensor) != self.N:
            raise ValueError(
                f"Длина шифротекста e ({len(e_input_coeffs_centered_tensor)}) должна быть равна N ({self.N})")

        if self.debug:
            self.logger.debug("--- PYTORCH DECRYPT ---")
            self.logger.debug(f"Input e (ЦЕНТРИРОВАННЫЙ, a0-aN-1): {e_input_coeffs_centered_tensor[:10].cpu().numpy()}")
            self.logger.debug(f"f (a0-aN-1): {self.f[:10].cpu().numpy()}")
            self.logger.debug(f"fp (a0-aN-1): {self.fp[:10].cpu().numpy()}")

        # Приводим все к compute_dtype
        f_comp = self.f.to(self.compute_dtype)
        e_comp = e_input_coeffs_centered_tensor.to(self.compute_dtype)
        fp_comp = self.fp.to(self.compute_dtype)

        # Шаг 1: a_raw = f * e_centered (mod x^N-1) (в Z[x])
        a_poly_in_Z = multiply_poly_pytorch_circulant_mod_ideal_only(f_comp, e_comp, self.N)
        if self.debug: self.logger.debug(
            f"PyTorch Decrypt: a_poly_in_Z (coeffs in Z): {a_poly_in_Z[:10].cpu().numpy()}")

        # Шаг 2: Коэффициенты 'a' по модулю q, а затем ЦЕНТРИРОВАНИЕ
        a_coeffs_mod_q = torch.remainder(a_poly_in_Z, self.q)
        if self.debug: self.logger.debug(
            f"PyTorch Decrypt: a_coeffs_mod_q ([0,q-1]): {a_coeffs_mod_q[:10].cpu().numpy()}")

        a_coeffs_truly_centered = a_coeffs_mod_q.clone()
        mask_q = 2 * a_coeffs_truly_centered >= self.q
        a_coeffs_truly_centered[mask_q] -= self.q
        if self.debug: self.logger.debug(
            f"PyTorch Decrypt: a_coeffs_truly_centered (centered mod q): {a_coeffs_truly_centered[:10].cpu().numpy()}")

        # Шаг 3: b = a_truly_centered (mod p).
        if self.p == 3:
            # Для p=3, центрируем результат к {-1,0,1}
            b_poly_mod_p = torch.remainder(a_coeffs_truly_centered, self.p)
            mask_p = 2 * b_poly_mod_p >= self.p
            b_poly_mod_p[mask_p] -= self.p
        else:  # Для p=2 или других, оставляем в [0,p-1]
            b_poly_mod_p = torch.remainder(a_coeffs_truly_centered, self.p)
        if self.debug: self.logger.debug(f"PyTorch Decrypt: b_poly_mod_p (mod p): {b_poly_mod_p[:10].cpu().numpy()}")

        # Шаг 4: m_prime = fp * b (mod x^N-1, mod p)
        # multiply_poly_pytorch_circulant_mod_ideal_only делает (P1*P2) mod (x^N-1)
        # Модуль p для коэффициентов нужно применить после.

        # Коэффициенты fp и b нужно привести к одному домену перед умножением.
        # Поскольку b уже mod p, приведем fp к mod p.
        fp_mod_p = torch.remainder(fp_comp, self.p)
        if self.p == 3:  # Если p=3, fp тоже нужно центрировать, чтобы соответствовать b
            mask_fp = 2 * fp_mod_p >= self.p
            fp_mod_p[mask_fp] -= self.p

        # Умножение в Z_p[x] / (x^N-1)
        c_poly_intermediate = multiply_poly_pytorch_circulant_mod_ideal_only(fp_mod_p, b_poly_mod_p, self.N)
        m_prime_coeffs = torch.remainder(c_poly_intermediate, self.p)

        # Финальное центрирование m', если необходимо
        if self.p == 3:
            mask_m_prime = 2 * m_prime_coeffs >= self.p
            m_prime_coeffs[mask_m_prime] -= self.p

        if self.debug: self.logger.debug(
            f"PyTorch Decrypt: m_prime_coeffs (final): {m_prime_coeffs[:10].cpu().numpy()}")

        # Если исходное сообщение было битами {0,1}, а m_prime_coeffs теперь {-1,0,1},
        # нужно преобразование. Например, -1 -> 1.
        if self.p == 3:
            m_prime_coeffs[m_prime_coeffs == -1] = 1  # Считаем, что -1 и 1 оба соответствуют биту 1

        return m_prime_coeffs.to(self.storage_dtype)  # Возвращаем тензор

    def decryptString(self, E_str: str):
        """
        Расшифровывает строку шифротекста.
        """
        # Парсим строку в NumPy массив
        try:
            Me_np = np.fromstring(E_str, dtype=np.int64, sep=' ')
        except ValueError:
            raise ValueError("Неверный формат строки шифротекста.")

        if np.mod(len(Me_np), self.N) != 0:
            raise ValueError("Длина шифротекста не кратна N")

        Me_tensor = torch.from_numpy(Me_np).to(device=self.device, dtype=self.compute_dtype)

        Marr_list_of_tensors = []
        num_blocks = len(Me_tensor) // self.N

        for i in range(num_blocks):
            e_block_tensor = Me_tensor[i * self.N:(i + 1) * self.N]
            decrypted_block_tensor = self.decrypt_poly(e_block_tensor)
            Marr_list_of_tensors.append(decrypted_block_tensor)

        # Собираем все блоки в один тензор, затем в NumPy массив
        if not Marr_list_of_tensors:
            self.M_str = ""
            return

        Marr_tensor = torch.cat(Marr_list_of_tensors)
        Marr_np = Marr_tensor.cpu().numpy()

        if self.debug: self.logger.debug(f"Финальный Marr ПЕРЕД bit2str (первые 40): {Marr_np[:40]}")

        self.M_str = utils.bit2str_corrected(Marr_np)

        if self.debug: self.logger.debug(f"Расшифрованное сообщение: '{self.M_str}'")