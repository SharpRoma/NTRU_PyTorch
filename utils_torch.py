import torch
import sys  # для sys.exit, если вы хотите сохранить это поведение


def genRand10_pytorch(L: int, P: int, M: int, device: str = 'cpu', dtype: torch.dtype = torch.int64) -> torch.Tensor:
    """
    Генерирует 1D тензор PyTorch длины L с P единицами (+1), M минус единицами (-1),
    а остальные элементы - нули. Элементы расположены в случайном порядке.

    Args:
        L (int): Длина выходного тензора.
        P (int): Количество +1.
        M (int): Количество -1.
        device (str): Устройство для создания тензора ('cpu', 'cuda', и т.д.).
        dtype (torch.dtype): Тип данных для тензора (например, torch.int8, torch.int32, torch.int64).

    Returns:
        torch.Tensor: Сгенерированный 1D тензор.
    """
    if P + M > L:
        sys.exit(f"ERROR in genRand10_pytorch: P+M ({P + M}) > L ({L}).")

    ones = torch.ones(P, dtype=dtype, device=device)
    # Затем M минус единиц
    minus_ones = -torch.ones(M, dtype=dtype, device=device)  # или torch.full((M,), -1, ...)
    # Затем L - P - M нулей
    num_zeros = L - P - M
    if num_zeros < 0:  # Дополнительная проверка, хотя первая уже должна это покрыть
        sys.exit(f"ERROR in genRand10_pytorch: L - P - M ({num_zeros}) < 0. Проверьте входные L, P, M.")

    zeros = torch.zeros(num_zeros, dtype=dtype, device=device)

    # 2. Конкатенируем их в один тензор
    R_ordered = torch.cat((ones, minus_ones, zeros))

    # 3. Генерируем случайную перестановку индексов
    permutation_indices = torch.randperm(L, device=device)

    # 4. Перемешиваем тензор, используя эту перестановку
    R_shuffled = R_ordered[permutation_indices]

    return R_shuffled

def padArr_pytorch(tensor_in: torch.Tensor, out_size: int, pad_value: int = 0) -> torch.Tensor:
    """
    Дополняет входной 1D тензор PyTorch `tensor_in` ведущими значениями `pad_value`
    так, чтобы результирующий тензор имел длину `out_size`.
    Если `out_size` меньше длины `tensor_in`, тензор обрезается с конца.
    Если `out_size` равна длине `tensor_in`, тензор возвращается без изменений.

    Args:
        tensor_in (torch.Tensor): Входной 1D тензор.
        out_size (int): Желаемая длина выходного тензора.
        pad_value (int): Значение для дополнения.

    Returns:
        torch.Tensor: Дополненный или обрезанный 1D тензор.
    """
    current_len = tensor_in.shape[0]

    if out_size < 0:
        raise ValueError("out_size не может быть отрицательным.")

    if current_len == out_size:
        return tensor_in
    elif current_len > out_size:
        # Обрезаем тензор с конца, чтобы получить нужную длину
        return tensor_in[:out_size]
    else: # current_len < out_size
        # Вычисляем, сколько элементов нужно добавить слева
        pad_len_left = out_size - current_len
        # torch.nn.functional.pad для 1D тензора ожидает кортеж (pad_left, pad_right)
        # Мы дополняем только слева (ведущие значения)
        return torch.nn.functional.pad(tensor_in, (pad_len_left, 0), mode='constant', value=pad_value)


def multiply_poly_pytorch_circulant(poly1: torch.Tensor, poly2: torch.Tensor, N: int, modulus: int) -> torch.Tensor:
    """
    Умножает два полинома poly1 и poly2 с циклической редукцией по модулю (x^N - 1)
    и взятием коэффициентов по модулю `modulus`.
    Один из полиномов (poly1) преобразуется в циркулянтную матрицу.

    Args:
        poly1 (torch.Tensor): Первый полином (1D тензор коэффициентов, который станет циркулянтной матрицей).
        poly2 (torch.Tensor): Второй полином (1D тензор коэффициентов, который будет вектором).
        N (int): Степень для циклической редукции (длина полиномов).
        modulus (int): Модуль для коэффициентов.

    Returns:
        torch.Tensor: Результат умножения (1D тензор).
    """
    if not (isinstance(poly1, torch.Tensor) and isinstance(poly2, torch.Tensor)):
        raise TypeError("Входные полиномы должны быть тензорами PyTorch.")

    if poly1.ndim != 1 or poly2.ndim != 1:
        raise ValueError("Входные полиномы должны быть 1D тензорами.")

    if len(poly1) != N or len(poly2) != N:
        raise ValueError(f"Длина полиномов ({len(poly1)}, {len(poly2)}) должна быть равна N ({N}).")

    row_indices = torch.arange(N, device=poly1.device).unsqueeze(1)
    col_indices = torch.arange(N, device=poly1.device).unsqueeze(0)

    indices_for_circulant = (row_indices - col_indices) % N

    # Собираем циркулянтную матрицу, используя расширенное индексирование
    circulant_matrix = poly1[indices_for_circulant]  # Размеры (N, N)

    promoted_dtype = torch.promote_types(circulant_matrix.dtype, poly2.dtype)

    result_intermediate = torch.matmul(circulant_matrix.to(promoted_dtype), poly2.to(promoted_dtype))

    result_mod = torch.remainder(result_intermediate, modulus)

    # Возвращаем результат в исходном типе poly2 (или в compute_dtype, если он был повышен)
    return result_mod.to(poly2.dtype if poly2.dtype == promoted_dtype else promoted_dtype)


def multiply_poly_pytorch_circulant_mod_ideal_only(poly1: torch.Tensor, poly2: torch.Tensor, N: int) -> torch.Tensor:
    """
    Умножает два полинома poly1 и poly2 с циклической редукцией по модулю (x^N - 1).
    Коэффициенты НЕ приводятся по модулю q на этом этапе.
    Один из полиномов (poly1) преобразуется в циркулянтную матрицу.

    Args:
        poly1 (torch.Tensor): Первый полином (1D тензор коэффициентов, который станет циркулянтной матрицей).
        poly2 (torch.Tensor): Второй полином (1D тензор коэффициентов, который будет вектором).
        N (int): Степень для циклической редукции (длина полиномов).

    Returns:
        torch.Tensor: Результат умножения (1D тензор) с коэффициентами в Z (или в dtype входа).
    """
    if not (isinstance(poly1, torch.Tensor) and isinstance(poly2, torch.Tensor)):
        raise TypeError("Входные полиномы должны быть тензорами PyTorch.")

    if poly1.ndim != 1 or poly2.ndim != 1:
        raise ValueError("Входные полиномы должны быть 1D тензорами.")

    if len(poly1) != N or len(poly2) != N:
        raise ValueError(f"Длина полиномов ({len(poly1)}, {len(poly2)}) должна быть равна N ({N}).")

    row_indices = torch.arange(N, device=poly1.device).unsqueeze(1)
    col_indices = torch.arange(N, device=poly1.device).unsqueeze(0)
    indices_for_circulant = (row_indices - col_indices) % N
    circulant_matrix = poly1[indices_for_circulant]  # Размеры (N, N)

    promoted_dtype = torch.promote_types(circulant_matrix.dtype, poly2.dtype)

    result_intermediate = torch.matmul(circulant_matrix.to(promoted_dtype), poly2.to(promoted_dtype))

    return result_intermediate.to(poly2.dtype if poly2.dtype == promoted_dtype else promoted_dtype)


def linear_convolve_pytorch(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """
    Выполняет линейную свёртку (умножение полиномов в Z[x]).
    poly1, poly2 - 1D тензоры [a0, a1, ...].
    Результат будет иметь длину len(poly1) + len(poly2) - 1.
    """

    p1_c = poly1.to(device=poly2.device, dtype=torch.promote_types(poly1.dtype, poly2.dtype))
    p2_c = poly2.to(device=poly1.device, dtype=torch.promote_types(poly1.dtype, poly2.dtype))

    # Размеры для conv1d
    p1_reshaped = p1_c.view(1, 1, -1)  # (1, 1, len(p1))
    p2_reshaped = p2_c.view(1, 1, -1)  # (1, 1, len(p2))

    if len(p1_c) == 0 or len(p2_c) == 0:  # Обработка пустых полиномов
        return torch.tensor([], dtype=p1_c.dtype, device=p1_c.device)

    # Разворачиваем тот, который будет ядром (например, p1_c)
    kernel = torch.flip(p1_c, dims=[0]).unsqueeze(0).unsqueeze(0)  # (1, 1, len(p1))
    input_signal = p2_c.unsqueeze(0).unsqueeze(0)  # (1, 1, len(p2))

    padding_val = kernel.shape[2] - 1

    # Убедимся, что padding не отрицательный (если ядро длины 0 или 1)
    if padding_val < 0: padding_val = 0

    result = torch.nn.functional.conv1d(input_signal, kernel, padding=padding_val)
    return result.squeeze()  # (len(p1) + len(p2) - 1)


# utils_torch.py - новая функция
def reduce_poly_mod_ideal_pytorch(poly_coeffs: torch.Tensor, N: int) -> torch.Tensor:
    """
    Редуцирует полином poly_coeffs по модулю (x^N - 1).
    poly_coeffs: 1D тензор [a0, a1, ...], может быть длиннее N.
    Возвращает 1D тензор длины N.
    """
    current_len = poly_coeffs.shape[0]
    if current_len <= N:
        # Если уже короче или равен N, дополняем нулями до N, если нужно
        return padArr_pytorch(poly_coeffs, N)  # Используем вашу padArr_pytorch

    result_coeffs = poly_coeffs[:N].clone()  # Берем первые N коэффициентов

    i = N
    while i < current_len:
        len_to_add = min(N, current_len - i)
        result_coeffs[:len_to_add] += poly_coeffs[i: i + len_to_add]
        i += N

    return result_coeffs  # Длина будет N


def linear_convolve_pytorch_alternative(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """
    Выполняет линейную свёртку (умножение полиномов в Z[x]) через FFT, если длины большие,
    или через прямое вычисление для малых.
    Для простоты и прямого контроля, давайте реализуем через циклы, как в numpy.convolve,
    а затем можно оптимизировать.
    ИЛИ, более простой способ для PyTorch часто - это использовать conv1d, но правильно настроить.

    Проверенная конфигурация conv1d для полиномиального умножения A(x) * B(x)
    где A = [a0, a1, ..., an], B = [b0, b1, ..., bm]
    Результат C будет иметь коэффициенты [c0, c1, ..., c(n+m)]
    """
    n = poly1.shape[0] - 1
    m = poly2.shape[0] - 1

    if n < 0 or m < 0:  # если один из полиномов пустой
        return torch.tensor([], dtype=torch.promote_types(poly1.dtype, poly2.dtype), device=poly1.device)

    # Результат будет иметь степень n + m, т.е. n + m + 1 коэффициентов
    result_len = n + m + 1

    # Приводим к общему типу для вычислений
    p1c = poly1.to(dtype=torch.promote_types(poly1.dtype, poly2.dtype), device=poly2.device)
    p2c = poly2.to(dtype=torch.promote_types(poly1.dtype, poly2.dtype), device=poly1.device)

    kernel = torch.flip(p1c, dims=[0]).view(1, 1, n + 1)
    input_signal = p2c.view(1, 1, m + 1)

    padding_val = n

    out = torch.nn.functional.conv1d(input_signal, kernel, padding=padding_val)

    return out.squeeze()  # результат будет иметь длину (m+1) + (n+1) - 1 = m+n+1


def linear_convolve_pytorch_direct(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """
    Выполняет линейную свёртку (умножение полиномов в Z[x]) прямым методом.
    poly1, poly2 - 1D тензоры [a0, a1, ...].
    Результат будет иметь длину len(poly1) + len(poly2) - 1.
    """
    len1 = poly1.shape[0]
    len2 = poly2.shape[0]

    if len1 == 0 or len2 == 0:
        return torch.tensor([], dtype=torch.promote_types(poly1.dtype, poly2.dtype), device=poly1.device)

    out_len = len1 + len2 - 1
    result = torch.zeros(out_len, dtype=poly1.dtype, device=poly1.device)

    for i in range(len1):
        for j in range(len2):
            result[i + j] += poly1[i] * poly2[j]

    return result


def linear_convolve_pytorch_conv1d(poly1: torch.Tensor, poly2: torch.Tensor) -> torch.Tensor:
    """
    Выполняет линейную свёртку (умножение полиномов) с помощью conv1d.
    """
    # Определяем, какой полином короче, чтобы он стал ядром (для эффективности)
    if len(poly1) < len(poly2):
        p_short, p_long = poly1, poly2
    else:
        p_short, p_long = poly2, poly1

    # Убедимся, что они в compute_dtype (например, int64)
    p_short = p_short.to(dtype=torch.int64)
    p_long = p_long.to(dtype=torch.int64)

    # Разворачиваем короткий полином, чтобы conv1d выполнял свёртку, а не корреляцию
    kernel = torch.flip(p_short, dims=[0]).view(1, 1, -1)

    # Готовим длинный полином
    input_signal = p_long.view(1, 1, -1)

    # Padding должен быть равен длине ядра минус 1, чтобы получить полную свёртку
    padding_val = p_short.shape[0] - 1

    # Выполняем свёртку
    result = torch.nn.functional.conv1d(input_signal, kernel, padding=padding_val)

    return result.squeeze()