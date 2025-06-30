import torch
import utils
from utils_torch import genRand10_pytorch, padArr_pytorch, multiply_poly_pytorch_circulant
import time
import numpy as np
from logger import logger
from utils import factor_int
import logging
from NTRUencrypt_torch import NTRUencryptTorch
from NTRUdecrypt_torch import NTRUdecryptTorch

def encrypt_torch(name: str, message: str, check_time: bool = True, debug: bool = False) -> str:

    logger.debug("Torch Encrypting message with key: %s", name)
    logger.debug("Torch Encrypting message: %s", message)
    start_time = time.time()

    E = NTRUencryptTorch(logger=logger, debug=debug)
    E.readPub(f"{name}.pub")
    E.encryptString(message)

    if check_time:
        elapsed = time.time() - start_time
        logger.info(f"Torch Encryption took {elapsed:.4f} seconds")

    return E.Me_str

def decrypt_torch(name: str, cipher: str, check_time: bool = True, debug: bool = False) -> str:
    """
    Функция-обертка для расшифровки с использованием NTRUdecryptTorch.
    """
    logger.debug("Torch Decrypting message with key: %s", name)
    start_time = time.time()

    D = NTRUdecryptTorch(logger=logger, debug=debug, check_time=check_time)
    D.readPriv(f"{name}.priv")
    D.decryptString(cipher)

    if check_time:
        elapsed = time.time() - start_time
        logger.info(f"Torch Decryption took {elapsed:.4f} seconds")

    return D.M_str