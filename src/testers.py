import torch
import numpy as np
from .models import FOSR
from .utils import load_data_from_mat, FeatureExtractorWrapper, DEFAULT_JSRS, DEFAULT_TRAIN_SNRS, DEFAULT_TEST_SNRS, \
    DA_ADAPT_SNRS, DA_TEST_SNRS


# ================= MATHEMATICAL HELPERS =================

def compute_mahalanobis_params(features, labels, num_classes, device):
    """Computes basic centers and covariance matrices (Source only)."""
    centers = []
    inv_covs = []
    for c in range(num_classes):
        class_feats = features[labels == c]
        if len(class_feats) > 1:
            mu = class_feats.mean(dim=0)
            centers.append(mu)
            centered = class_feats - mu
            cov = torch.mm(centered.t(), centered) / (len(class_feats) - 1) + 1e-6 * torch.eye(features.size(1)).to(
                device)
            inv_covs.append(torch.linalg.inv((cov + cov.t()) / 2))
        else:
            centers.append(torch.zeros(features.size(1)).to(device))
            inv_covs.append(torch.eye(features.size(1)).to(device))
    return torch.stack(centers), torch.stack(inv_covs)


def class_wise_alignment(target_features, source_centers, device):
    """
    对应 best_da_test.py 中的 class_wise_alignment。
    执行 Global Mean Shift: Target - T_Mean + S_Mean
    """
    if target_features.size(0) == 0:
        return target_features

    target_features = target_features.to(device)
    source_centers = source_centers.to(device)

    # 初始对齐：Global Mean Shift
    t_global_mu = target_features.mean(dim=0)
    s_global_mu = source_centers.mean(dim=0)

    # 对齐逻辑：减去自身均值，加上源域均值
    current_features = target_features - t_global_mu + s_global_mu

    return current_features


def compute_mixed_covariance(source_features, source_labels,
                             target_features_aligned, target_pseudo_labels,
                             num_classes, device):
    """Computes Mixed Covariance (Source + Adapt Data)."""
    inv_covs = []
    for c in range(num_classes):
        s_feats = source_features[source_labels == c]
        t_feats = target_features_aligned[target_pseudo_labels == c]

        if len(t_feats) > 0:
            combined_feats = torch.cat([s_feats, t_feats], dim=0)
        else:
            combined_feats = s_feats

        if len(combined_feats) < 2:
            cov = torch.eye(combined_feats.size(1), device=device)
        else:
            center = combined_feats.mean(dim=0)
            centered = combined_feats - center.unsqueeze(0)
            cov = torch.mm(centered.t(), centered) / (combined_feats.size(0) - 1)
            cov = cov + 1e-6 * torch.eye(cov.size(0), device=device)
            cov = (cov + cov.t()) / 2

        try:
            inv_cov = torch.linalg.inv(cov)
        except:
            inv_cov = torch.eye(cov.size(0), device=device)

        inv_covs.append(inv_cov)

    return torch.stack(inv_covs)


def compute_thresholds_from_mix(source_features, source_labels,
                                target_features, target_pseudo_labels,
                                centers, inv_covs, device):
    """Computes 98% thresholds based on Mixed Distances."""

    def get_dists(feats):
        if feats.numel() == 0:
            return torch.zeros((0, centers.size(0)), device=device)
        dists_list = []
        for c in range(centers.size(0)):
            diff = feats - centers[c].unsqueeze(0)
            diff_trans = torch.mm(diff, inv_covs[c])
            d = (diff * diff_trans).sum(dim=1)
            dists_list.append(d)
        return torch.stack(dists_list, dim=1)

    s_dists_matrix = get_dists(source_features)
    t_dists_matrix = get_dists(target_features)

    thresholds = []
    for c in range(centers.size(0)):
        s_dists = s_dists_matrix[source_labels == c, c].cpu().numpy()

        if t_dists_matrix.size(0) > 0:
            t_dists = t_dists_matrix[target_pseudo_labels == c, c].cpu().numpy()
        else:
            t_dists = np.array([])


        thresh = np.percentile(t_dists, 98)

        thresholds.append(thresh)

    return thresholds


def compute_distances_and_evaluate(features, labels, thresholds, centers, inv_covs, device, known=True):
    if features.size(0) == 0:
        return 0.0

    num_classes = centers.size(0)
    batch_dists = []

    for c in range(num_classes):
        diff = features - centers[c].unsqueeze(0)
        diff_transformed = torch.mm(diff, inv_covs[c])
        dists = (diff * diff_transformed).sum(dim=1)
        batch_dists.append(dists)

    batch_dists = torch.stack(batch_dists, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    correct = 0
    total = len(labels)

    for i in range(total):
        sample_dists = batch_dists[i]
        true_label = labels[i]

        valid_classes = []
        for c in range(num_classes):
            if sample_dists[c] <= thresholds[c]:
                valid_classes.append(c)

        if valid_classes:
            pred = min(valid_classes, key=lambda x: sample_dists[x])
        else:
            pred = -1

        if known:
            if pred == true_label: correct += 1
        else:
            if pred == -1: correct += 1

    return correct / total if total > 0 else 0.0


def get_model_features(X, y, ext, model, device):
    if len(X) == 0:
        return torch.tensor([]).to(device), torch.tensor([]).to(device)

    xt = torch.tensor(X).float().to(device)
    yt = torch.tensor(y).long().to(device)

    ft = ext.extract(xt).unsqueeze(1)
    ct = torch.cat([xt, ft], dim=1)

    with torch.no_grad():
        features, _, _ = model(ct)

    return features, yt


# ================= TEST RUNNERS =================

def run_basic_test(source_path, target_path, ae_path, fosr_path, device):
    print(f"\n[Step 3] Running Basic Test (Source Only Mode)...")

    model = FOSR(feat_dim=128).to(device)
    model.load_state_dict(torch.load(fosr_path, map_location=device))
    model.eval()
    ext = FeatureExtractorWrapper(ae_path, device)

    known_accs, unknown_accs, na_accs = [], [], []

    for jsr in DEFAULT_JSRS:
        print(f"\nProcessing JSR={jsr}...")

        # 1. Source Stats
        X_s, y_s, _ = load_data_from_mat(source_path, [jsr], DEFAULT_TRAIN_SNRS, mode='known')
        feat_s, label_s = get_model_features(X_s, y_s, ext, model, device)

        centers, inv_covs = compute_mahalanobis_params(feat_s, label_s, 4, device)

        # Calculate thresholds (Source only)
        thresholds = compute_thresholds_from_mix(
            feat_s, label_s,
            torch.tensor([]).to(device), torch.tensor([]).to(device),
            centers, inv_covs, device
        )

        # 2. Target Test (Known)
        X_t_k, y_t_k, _ = load_data_from_mat(target_path, [jsr], DEFAULT_TEST_SNRS, mode='known')
        feat_k, label_k = get_model_features(X_t_k, y_t_k, ext, model, device)

        # 3. Target Test (Unknown)
        X_t_u, y_t_u, _ = load_data_from_mat(target_path, [jsr], DEFAULT_TEST_SNRS, mode='unknown')
        feat_u, label_u = get_model_features(X_t_u, y_t_u, ext, model, device)

        acc_k = compute_distances_and_evaluate(feat_k, label_k, thresholds, centers, inv_covs, device, known=True)
        acc_u = compute_distances_and_evaluate(feat_u, label_u, thresholds, centers, inv_covs, device, known=False)

        na = 0.5 * acc_k + 0.5 * acc_u
        known_accs.append(acc_k)
        unknown_accs.append(acc_u)
        na_accs.append(na)
        print(f"  JSR={jsr} | Known: {acc_k:.4f} | Unknown: {acc_u:.4f} | NA: {na:.4f}")

    print(f"\nAverage Known: {np.mean(known_accs):.4f}")
    print(f"Average Unknown: {np.mean(unknown_accs):.4f}")
    print(f"Average NA: {np.mean(na_accs):.4f}")


def run_da_test(source_path, target_path, ae_path, fosr_path, device):
    """
    DA Mode: Strictly matches best_da_test.py logic.
    Split: 20% Adapt, 15% Test.
    """
    print(f"\n[Step 3] Running Domain Adaptation Test (Full Logic)...")

    model = FOSR(feat_dim=128).to(device)
    model.load_state_dict(torch.load(fosr_path, map_location=device))
    model.eval()
    ext = FeatureExtractorWrapper(ae_path, device)

    known_accs, unknown_accs, na_accs = [], [], []

    for jsr in DEFAULT_JSRS:
        print(f"\n>>> Processing JSR = {jsr} ...")

        # --- 1. Source Processing ---
        X_s, y_s, _ = load_data_from_mat(source_path, [jsr], DEFAULT_TRAIN_SNRS)
        feat_s, label_s = get_model_features(X_s, y_s, ext, model, device)

        # Initial Source Params
        centers_s, _ = compute_mahalanobis_params(feat_s, label_s, 4, device)

        # --- 2. Target Split (20% Adapt, 15% Test) ---
        X_t, y_t, s_t = load_data_from_mat(target_path, [jsr], [], mode='known')

        np.random.seed(417)
        indices = np.random.permutation(len(y_t))

        # 修正：按照 best_da_test.py，cut1是20%，cut2是35% (即 20% + 15%)
        cut1 = int(len(y_t) * 0.20)
        cut2 = int(len(y_t) * 0.35)

        # Adapt Set
        idx_adapt_raw = indices[:cut1]
        mask_adapt = np.isin(s_t[idx_adapt_raw], DA_ADAPT_SNRS)
        X_adapt = X_t[idx_adapt_raw][mask_adapt]
        y_adapt = y_t[idx_adapt_raw][mask_adapt]

        # Test Set
        idx_test_raw = indices[cut1:cut2]
        mask_test = np.isin(s_t[idx_test_raw], DA_TEST_SNRS)
        X_test = X_t[idx_test_raw][mask_test]
        y_test = y_t[idx_test_raw][mask_test]

        # Unknown Set
        X_u, y_u, _ = load_data_from_mat(target_path, [jsr], DA_TEST_SNRS, mode='unknown')

        # --- 3. Adaptation Logic ---
        feat_adapt, label_adapt = get_model_features(X_adapt, y_adapt, ext, model, device)
        feat_test, label_test = get_model_features(X_test, y_test, ext, model, device)
        feat_u, label_u = get_model_features(X_u, y_u, ext, model, device)

        # [Step A] 对齐 (Adapt Set) - 仅用于计算 Mixed Stats
        feat_adapt_aligned = class_wise_alignment(feat_adapt, centers_s, device)

        # [Step B] 伪标签 (Pseudo Labeling)
        # 注意：这里使用对齐后的特征和源域中心计算距离
        dists = []
        for c in range(4):
            # 你的代码里用的是欧氏距离 (x-c)^2
            d = torch.sum((feat_adapt_aligned - centers_s[c]) ** 2, dim=1)
            dists.append(d)
        if len(dists) > 0:
            pseudo_labels_adapt = torch.stack(dists, dim=1).argmin(dim=1)
        else:
            pseudo_labels_adapt = torch.tensor([]).long().to(device)

        # [Step C] 混合协方差 (Mixed Covariance)
        mixed_inv_covs = compute_mixed_covariance(
            feat_s, label_s,
            feat_adapt_aligned, pseudo_labels_adapt,
            num_classes=4, device=device
        )

        # [Step D] 混合阈值 (Mixed Thresholds)
        thresholds = compute_thresholds_from_mix(
            feat_s, label_s,
            feat_adapt_aligned, pseudo_labels_adapt,
            centers_s, mixed_inv_covs, device
        )

        # --- 4. Evaluation (CRITICAL FIX) ---
        # 修正：测试集和未知集必须分别使用 class_wise_alignment 进行对齐
        # 而不是使用 Adapt Set 的 offset

        feat_test_aligned = class_wise_alignment(feat_test, centers_s, device)
        feat_u_aligned = class_wise_alignment(feat_u, centers_s, device)

        acc_k = compute_distances_and_evaluate(
            feat_test_aligned, label_test, thresholds, centers_s, mixed_inv_covs, device, known=True
        )
        acc_u = compute_distances_and_evaluate(
            feat_u_aligned, label_u, thresholds, centers_s, mixed_inv_covs, device, known=False
        )

        na = 0.5 * acc_k + 0.5 * acc_u
        known_accs.append(acc_k)
        unknown_accs.append(acc_u)
        na_accs.append(na)
        print(f"  JSR={jsr} | Known: {acc_k:.4f} | Unknown: {acc_u:.4f} | NA: {na:.4f}")

    print(f"\nAverage Known: {np.mean(known_accs):.4f}")
    print(f"Average Unknown: {np.mean(unknown_accs):.4f}")
    print(f"Average NA: {np.mean(na_accs):.4f}")