import torch
import numpy as np  # для преобразования из/в numpy при чтении/записи и для совместимости со старым utils
import sys
import utils
from utils import str2bit, arr2str
from utils_torch import (
    multiply_poly_pytorch_circulant, # Если все еще используется где-то
    multiply_poly_pytorch_circulant_mod_ideal_only, # Если была создана и используется
    linear_convolve_pytorch,                         # <--- ДОБАВЛЯЕМ ЭТУ
    linear_convolve_pytorch_alternative,
    linear_convolve_pytorch_direct,
    linear_convolve_pytorch_conv1d,
    reduce_poly_mod_ideal_pytorch,                   # <--- И ЭТУ
    genRand10_pytorch,
    padArr_pytorch
)


class NTRUencryptTorch:
    def __init__(self, logger, N=503, p=3, q=256, d=18, device='cpu', debug=True, check_time=True):
        self.logger = logger
        self.N = N
        self.p = p
        self.q = q
        self.dr = d
        self.device = torch.device(device)
        self.compute_dtype = torch.int64  # Тип для вычислений
        self.storage_dtype = torch.int32  # Тип для хранения, если нужно экономить
        self.debug = debug

        # Инициализируем тензоры
        # g не используется в шифровании NTRUEncrypt, только h
        self.h = torch.zeros(self.N, dtype=self.compute_dtype, device=self.device)
        self.r = torch.zeros(self.N, dtype=self.compute_dtype, device=self.device)
        self.m_poly = torch.zeros(self.N, dtype=self.compute_dtype,
                                  device=self.device)  # Переименовал self.m в self.m_poly
        self.e = torch.zeros(self.N, dtype=self.compute_dtype, device=self.device)

        self.genr()  # Генерируем начальный r

        # Идеал x^N - 1 не нужен явно, если умножение реализует циклическую свёртку

        self.readKey = False
        self.Me_str = None  # Переименовал self.Me в self.Me_str

        if self.debug:
            self.logger.debug(f"Initialized NTRUencryptTorch with N={N}, p={p}, q={q}, d={d}")

    def readPub(self, filename="key.pub"):
        with open(filename, "r") as f:
            self.p = int(f.readline().split(" ")[-1])
            self.q = int(f.readline().split(" ")[-1])
            self.N = int(f.readline().split(" ")[-1])
            self.dr = int(f.readline().split(" ")[-1])
            h_np = np.array(f.readline().split(" ")[3:-1], dtype=np.int64)  # Читаем как int64

        self.h = torch.from_numpy(h_np).to(device=self.device, dtype=self.compute_dtype)
        if len(self.h) != self.N:  # Проверка или дополнение, если нужно
            self.h = padArr_pytorch(self.h, self.N)  # Убедимся, что h имеет длину N

        self.genr()  # Перегенерируем r с новыми N, dr
        self.readKey = True

    def genr(self):
        self.r = genRand10_pytorch(self.N, self.dr, self.dr, device=self.device, dtype=self.compute_dtype)

    def setM_poly(self, m_coeffs_input):  # m_coeffs_input - это тензор или список/numpy массив
        if not self.readKey:
            sys.exit("ERROR : Public key not read before setting message")

        if isinstance(m_coeffs_input, np.ndarray):
            m_tensor = torch.from_numpy(m_coeffs_input.astype(np.int64)).to(device=self.device,
                                                                            dtype=self.compute_dtype)
        elif isinstance(m_coeffs_input, list):
            m_tensor = torch.tensor(m_coeffs_input, device=self.device, dtype=self.compute_dtype)
        elif isinstance(m_coeffs_input, torch.Tensor):
            m_tensor = m_coeffs_input.to(device=self.device, dtype=self.compute_dtype)
        else:
            sys.exit("ERROR: m_coeffs_input должен быть списком, NumPy массивом или тензором PyTorch")

        if len(m_tensor) > self.N:
            sys.exit("ERROR : Message length longer than N")

        self.m_poly = padArr_pytorch(m_tensor, self.N)

    def encrypt_poly(self, m_poly_input=None, fixed_r_tensor=None):
        if not self.readKey:
            raise RuntimeError("Error : Not read the public key file, so cannot encrypt")
            # sys.exit("Error : Not read the public key file, so cannot encrypt")

        current_m_poly = self.m_poly
        if m_poly_input is not None:
            if not isinstance(m_poly_input, torch.Tensor):
                m_poly_input = torch.tensor(m_poly_input, device=self.device, dtype=self.compute_dtype)
            if len(m_poly_input) != self.N:
                m_poly_input = padArr_pytorch(m_poly_input, self.N, pad_value=0)  # Убедимся, что дополняем нулями
            current_m_poly = m_poly_input

        current_r_tensor = self.r
        if fixed_r_tensor is not None:
            if not isinstance(fixed_r_tensor, torch.Tensor):
                fixed_r_tensor = torch.tensor(fixed_r_tensor, device=self.device, dtype=self.compute_dtype)
            if len(fixed_r_tensor) != self.N:
                fixed_r_tensor = padArr_pytorch(fixed_r_tensor, self.N, pad_value=0)
            current_r_tensor = fixed_r_tensor

        if self.debug:
            self.logger.debug("--- PYTORCH ENCRYPT ---")
            self.logger.debug(f"N={self.N}, p={self.p}, q={self.q}")
            self.logger.debug(f"h (first 10): {self.h[:10].cpu().numpy()}")
            self.logger.debug(f"r (fixed, first 10): {current_r_tensor[:10].cpu().numpy()}")
            self.logger.debug(f"m (fixed, first 10): {current_m_poly[:10].cpu().numpy()}")

        r_comp = current_r_tensor.to(device=self.device, dtype=self.compute_dtype)
        h_comp = self.h.to(device=self.device, dtype=self.compute_dtype)
        m_comp = current_m_poly.to(device=self.device, dtype=self.compute_dtype)

        # Шаг 1
        rh_poly_full = linear_convolve_pytorch_conv1d(r_comp, h_comp)
        if self.debug:
            self.logger.debug(f"PyTorch rh_poly_full (Z[x], first 10 coeffs): {rh_poly_full[:10].cpu().numpy()}")

        # Шаг 2
        term1_coeffs_mod_q = torch.remainder(rh_poly_full, self.q)
        if self.debug:
            self.logger.debug(f"PyTorch term1_coeffs_mod_q ([0,q-1], first 10 coeffs): {term1_coeffs_mod_q[:10].cpu().numpy()}")

        # Шаг 3
        m_comp_padded_for_sum = torch.nn.functional.pad(m_comp, (0, rh_poly_full.shape[0] - self.N), value=0)
        sum_intermediate_coeffs = term1_coeffs_mod_q + m_comp_padded_for_sum
        if self.debug:
            self.logger.debug(f"PyTorch sum_intermediate_coeffs (first 10 coeffs): {sum_intermediate_coeffs[:10].cpu().numpy()}")

        # Шаг 4
        sum_coeffs_reduced_mod_I = reduce_poly_mod_ideal_pytorch(sum_intermediate_coeffs, self.N)
        if self.debug:
            self.logger.debug(f"PyTorch sum_coeffs_reduced_mod_I (after mod I, first 10): {sum_coeffs_reduced_mod_I[:10].cpu().numpy()}")

        # Шаг 5
        e_mod_q_positive = torch.remainder(sum_coeffs_reduced_mod_I, self.q)
        if self.debug:
            self.logger.debug(f"PyTorch e_mod_q_positive ([0,q-1], first 10): {e_mod_q_positive[:10].cpu().numpy()}")

        # Центрирование e
        self.e = e_mod_q_positive.clone()
        mask_to_subtract_q = 2 * self.e >= self.q
        self.e[mask_to_subtract_q] -= self.q

        if self.debug:
            self.logger.debug(f"PyTorch self.e (ЦЕНТРИРОВАННЫЙ, [a0..aN-1], first 10): {self.e[:10].cpu().numpy()}")

        if self.e.dtype != self.storage_dtype:
            self.e = self.e.to(self.storage_dtype)

        return self.e  # Возвращаем тензор

    def encryptString(self, M_str_input: str):
        self.logger.debug(f"Torch Encrypting message: '{M_str_input}'")  # <--- Полезный INFO лог

        if not self.readKey:
            raise RuntimeError("Error : Not read the public key file, so cannot encrypt")

        bM_np = str2bit(M_str_input)  # Получаем NumPy массив битов {0,1}

        current_len = len(bM_np)
        if current_len % self.N != 0:
            target_len = ((current_len // self.N) + 1) * self.N
            bM_np_padded = np.pad(bM_np, (0, target_len - current_len), 'constant', constant_values=0)
        else:
            bM_np_padded = bM_np

        if bM_np_padded.size == 0:
            self.logger.warning("Attempting to encrypt an empty string or a string that resulted in zero bits.")
            self.Me_str = ""
            return

        # bM_np_padded = np.pad(bM_np, (0, target_len - current_len), constant_values=0)

        bM_tensor = torch.from_numpy(bM_np_padded.astype(np.int64)).to(device=self.device, dtype=self.compute_dtype)

        self.Me_str = ""
        num_blocks = len(bM_tensor) // self.N

        encrypted_blocks = []
        for i in range(num_blocks):
            self.genr()
            block_m = bM_tensor[i * self.N:(i + 1) * self.N]
            self.encrypt_poly(m_poly_input=block_m)  # Вызываем отлаженный метод
            e_block_np = self.e.cpu().numpy().astype(np.int32)
            self.Me_str += arr2str(e_block_np) + " "

        self.Me_str = self.Me_str.strip()