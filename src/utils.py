import os
import scipy.io
import numpy as np
import torch
from .models import ComplexAutoencoder

# ================= CONFIGURATION =================
# Based on your local code settings
DEFAULT_JSRS = [-10, -5, 0, 5, 10, 15]
DEFAULT_TRAIN_SNRS = [0, 5, 10]
DEFAULT_TEST_SNRS = [0, 5, 10]  # Used in Basic Test
DA_ADAPT_SNRS = [0, 5, 10]  # Used for Adaptation Set in DA
DA_TEST_SNRS = [10]  # Used for Test Set in DA


class FeatureExtractorWrapper:
    def __init__(self, model_path, device):
        self.device = device
        self.model = ComplexAutoencoder(input_dim=128, encoding_dim=128).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def extract(self, x_tensor):
        with torch.no_grad():
            if x_tensor.dim() == 3:
                x_tensor = x_tensor.reshape(x_tensor.size(0), -1)
            features, _ = self.model(x_tensor)
        return features


def split_indices(n, train_ratio=0.7, val_ratio=0.15, seed=417):
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]
    return idx_train, idx_val, idx_test


def load_data_from_mat(mat_path, interference_levels, snrs, mode='known'):
    """
    Loads data matching the logic in load_data_basic from best_da_test.py
    Returns X, y, and s (SNR) for splitting logic.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Data file not found: {mat_path}")

    data = scipy.io.loadmat(mat_path)
    X = data['x']
    y = data['y'].flatten()
    z = data['z'].flatten()
    s = data['s'].flatten()

    X_real = np.real(X)
    X_imag = np.imag(X)
    X_all = np.stack((X_real, X_imag), axis=1)
    X_all = np.transpose(X_all, (2, 1, 0))

    # Updated Class Definitions based on your local code
    known_mods = [0, 1, 3, 4]  # BPSK, MultiTone, SingleTone, Pulse
    unknown_mods = [2, 5, 6]  # Comb, NoiseFM, Chirp

    if mode == 'known':
        selected_mods = known_mods
        label_mapping = {mod: idx for idx, mod in enumerate(known_mods)}
    elif mode == 'unknown':
        selected_mods = unknown_mods
        label_mapping = {mod: idx for idx, mod in enumerate(unknown_mods)}
    else:
        selected_mods = known_mods + unknown_mods
        label_mapping = {i: i for i in range(7)}

    # Filter based on JSR, SNR (if provided), and Label
    # Note: If snrs is None or empty, we don't filter by SNR here (useful for DA split)
    mask = np.isin(y, selected_mods) & np.isin(z, interference_levels)
    if snrs is not None and len(snrs) > 0:
        mask = mask & np.isin(s, snrs)

    X_filtered = X_all[mask]
    y_filtered = y[mask]
    s_filtered = s[mask]

    # Normalization
    magnitudes = np.sqrt(np.sum(X_filtered ** 2, axis=1))
    max_magnitudes = np.max(magnitudes, axis=1, keepdims=True).reshape(-1, 1, 1)
    X_filtered = X_filtered / (max_magnitudes + 1e-8)

    y_filtered = np.array([label_mapping[yi] for yi in y_filtered], dtype=np.int64)

    return X_filtered, y_filtered, s_filtered