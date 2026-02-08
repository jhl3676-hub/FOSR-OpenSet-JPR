import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
import numpy as np
import os
from .models import ComplexAutoencoder, FOSR
from .losses import DistanceCrossEntropyLoss, LearnableCovCenterLoss
from .utils import load_data_from_mat, split_indices, FeatureExtractorWrapper, DEFAULT_JSRS, DEFAULT_TRAIN_SNRS


def run_ae_training(source_path, target_path, output_dir, device, use_da=False):
    print(f"\n[Step 1] Training Feature Enhancement Module (CV-AE)...")

    # Load with default train SNRs
    X_s, y_s, _ = load_data_from_mat(source_path, DEFAULT_JSRS, DEFAULT_TRAIN_SNRS)
    dset_s = TensorDataset(torch.tensor(X_s).float(), torch.tensor(y_s).long())
    full_dataset = dset_s

    if use_da and target_path:
        print("  - Mixing target domain data for AE training...")
        X_t, y_t, _ = load_data_from_mat(target_path, DEFAULT_JSRS, DEFAULT_TRAIN_SNRS)
        dset_t = TensorDataset(torch.tensor(X_t).float(), torch.tensor(y_t).long())

        np.random.seed(417)
        subset_indices = np.random.choice(len(dset_t), int(len(dset_t) * 0.1), replace=False)
        dset_t_sub = Subset(dset_t, subset_indices)
        full_dataset = ConcatDataset([dset_s, dset_t_sub])

    idx_train, idx_val, _ = split_indices(len(full_dataset), seed=417)
    train_loader = DataLoader(Subset(full_dataset, idx_train), batch_size=512, shuffle=True)

    model = ComplexAutoencoder().to(device)
    recon_loss_fn = nn.MSELoss()
    # Assuming standard AE training doesn't heavily rely on disc loss in your local script,
    # but keeping it as per previous setup. If your local script strictly uses MSE,
    # you can comment out disc_loss related lines.
    disc_loss_fn = DistanceCrossEntropyLoss(num_classes=4, feat_dim=128).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(disc_loss_fn.parameters()), lr=0.0008)

    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, out = model(x)
            loss = recon_loss_fn(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{epochs} complete")

    save_path = os.path.join(output_dir, "ae_best.pth")
    torch.save(model.state_dict(), save_path)
    return save_path


def run_fosr_training(source_path, ae_path, output_dir, device):
    print(f"\n[Step 2] Training FOSR Model...")

    X_all, y_all, _ = load_data_from_mat(source_path, DEFAULT_JSRS, DEFAULT_TRAIN_SNRS)
    feature_extractor = FeatureExtractorWrapper(ae_path, device)

    X_tensor = torch.tensor(X_all).float().to(device)
    y_tensor = torch.tensor(y_all).long().to(device)
    feats = feature_extractor.extract(X_tensor).unsqueeze(1)
    combined = torch.cat([X_tensor, feats], dim=1)

    idx_train, idx_val, _ = split_indices(len(combined), seed=417)
    train_loader = DataLoader(TensorDataset(combined[idx_train], y_tensor[idx_train]), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(combined[idx_val], y_tensor[idx_val]), batch_size=128, shuffle=False)

    model = FOSR(feat_dim=128).to(device)  # Ensure FOSR init matches models.py
    center_loss_fn = LearnableCovCenterLoss(num_classes=4, feat_dim=128).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(center_loss_fn.parameters()), lr=1e-3)
    rec_loss_fn = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(1, 31):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            feat, recon, _ = model(x)

            # Add noise as per local train.py (0.05)
            noisy_input = x + 0.05 * torch.randn_like(x)
            noisy_feat, _, _ = model(noisy_input)

            loss_rec = rec_loss_fn(recon, x)
            loss_center = center_loss_fn(feat, y, noisy_feat)

            # Local train.py: loss = 1 * loss_center + loss_rec
            loss = 1.0 * loss_center + loss_rec

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                feat, recon, _ = model(x)

                noisy = x + 0.05 * torch.randn_like(x)
                noisy_feat, _, _ = model(noisy)

                l_rec = rec_loss_fn(recon, x)
                l_cent = center_loss_fn(feat, y, noisy_feat)
                val_loss += (1.0 * l_cent + l_rec).item()

        avg_val = val_loss / len(val_loader)
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, "fosr_best.pth"))
            torch.save(center_loss_fn.state_dict(), os.path.join(output_dir, "center_loss.pth"))

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Val Loss {avg_val:.4f}")

    print(f"FOSR Training Done.")
    return os.path.join(output_dir, "fosr_best.pth")