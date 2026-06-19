# KANONIKΟΠΟΙΗΣΗ



import numpy as np


def normalize_signal(symbols):
    p_average = np.mean(np.abs(symbols)**2)

    return symbols / np.sqrt(p_average)
