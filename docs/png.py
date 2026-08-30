import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# ==========================================
# 1. FASE 1: AGE REGRESSOR (10 Epoche)
# ==========================================
epochs_reg = list(range(1, 11))
train_mae = [23.34, 10.62, 5.97, 5.30, 4.98, 4.57, 4.40, 4.12, 3.80, 3.48]
val_mae   = [15.91, 6.86, 5.66, 5.41, 5.99, 5.60, 5.67, 5.85, 5.58, 5.21]
train_mse = [720.08, 182.33, 67.39, 52.02, 45.64, 37.90, 35.60, 31.11, 25.76, 21.68]
val_mse   = [335.69, 87.02, 65.10, 59.89, 69.12, 66.16, 67.72, 72.89, 60.72, 56.82]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ax1.plot(epochs_reg, train_mae, marker="o", linewidth=2.2, label="Train MAE", color="#1f77b4")
ax1.plot(epochs_reg, val_mae, marker="s", linewidth=2.2, linestyle="--", label="Val MAE", color="#ff7f0e")
ax1.set_title("Age Regressor — Mean Absolute Error (Years)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MAE (Years)")
ax1.set_xticks(epochs_reg)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_reg, train_mse, marker="o", linewidth=2.2, label="Train MSE", color="#2ca02c")
ax2.plot(epochs_reg, val_mse, marker="s", linewidth=2.2, linestyle="--", label="Val MSE", color="#d62728")
ax2.set_title("Age Regressor — Loss (MSE)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.set_xticks(epochs_reg)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fase1_age_regressor.png", dpi=300)
plt.close()

# ==========================================
# 2. FASE 2: PAIRWISE MLP (5 Epoche)
# ==========================================
epochs_mlp = list(range(1, 6))
mlp_train_loss = [0.1411, 0.0380, 0.0330, 0.0317, 0.0309]
mlp_val_loss   = [0.0407, 0.0321, 0.0291, 0.0282, 0.0272]
mlp_train_acc  = [0.9768, 0.9900, 0.9905, 0.9908, 0.9910]
mlp_val_acc    = [0.9902, 0.9911, 0.9902, 0.9902, 0.9934]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ax1.plot(epochs_mlp, mlp_train_loss, marker="o", linewidth=2.2, label="Train Loss", color="#1f77b4")
ax1.plot(epochs_mlp, mlp_val_loss, marker="s", linewidth=2.2, linestyle="--", label="Val Loss", color="#d62728")
ax1.set_title("Pairwise MLP — Loss (CrossEntropy)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_xticks(epochs_mlp)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_mlp, mlp_train_acc, marker="o", linewidth=2.2, label="Train Acc", color="#2ca02c")
ax2.plot(epochs_mlp, mlp_val_acc, marker="s", linewidth=2.2, linestyle="--", label="Val Acc", color="#ff7f0e")
ax2.set_title("Pairwise MLP — Classification Accuracy", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_xticks(epochs_mlp)
ax2.set_ylim(0.96, 1.0)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fase2_pairwise_mlp.png", dpi=300)
plt.close()

# ==========================================
# 3. EARLY FUSION: RESNET 6 CANALI (30 Epoche)
# ==========================================
epochs_ef = list(range(1, 31))
ef_train_loss = [0.5987, 0.4617, 0.3581, 0.2752, 0.1972, 0.1467, 0.1059, 0.0862, 0.0678, 0.0618, 0.0531, 0.0455, 0.0421, 0.0387, 0.0417, 0.0350, 0.0310, 0.0334, 0.0308, 0.0266, 0.0289, 0.0257, 0.0242, 0.0221, 0.0240, 0.0244, 0.0251, 0.0219, 0.0179, 0.0212]
ef_val_loss   = [0.5476, 0.4841, 0.5686, 0.6750, 0.7844, 0.8578, 0.8293, 0.7703, 0.7913, 0.8919, 0.8386, 0.9212, 0.8961, 0.7601, 0.8489, 0.8904, 1.0474, 0.8696, 0.8476, 0.9270, 0.8929, 0.9300, 0.8141, 1.0179, 0.8719, 0.8577, 0.8975, 0.8327, 0.9193, 0.8000]
ef_train_acc  = [0.6731, 0.7746, 0.8369, 0.8831, 0.9192, 0.9424, 0.9593, 0.9673, 0.9748, 0.9773, 0.9804, 0.9831, 0.9852, 0.9860, 0.9852, 0.9871, 0.9890, 0.9878, 0.9888, 0.9902, 0.9897, 0.9904, 0.9918, 0.9922, 0.9911, 0.9914, 0.9913, 0.9925, 0.9937, 0.9926]
ef_val_acc    = [0.7112, 0.7672, 0.7568, 0.7494, 0.7622, 0.7680, 0.7479, 0.7944, 0.7898, 0.7869, 0.7871, 0.8054, 0.7925, 0.8210, 0.8025, 0.8029, 0.7911, 0.8037, 0.8000, 0.7917, 0.7992, 0.7992, 0.8205, 0.7990, 0.8166, 0.8112, 0.8091, 0.8044, 0.8023, 0.8201]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ax1.plot(epochs_ef, ef_train_loss, marker="o", markersize=3.5, linewidth=2, label="Train Loss", color="#1f77b4")
ax1.plot(epochs_ef, ef_val_loss, marker="s", markersize=3.5, linewidth=2, linestyle="--", label="Val Loss", color="#d62728")
ax1.set_title("Early Fusion — CrossEntropy Loss", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_xticks(range(1, 31, 2))
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_ef, ef_train_acc, marker="o", markersize=3.5, linewidth=2, label="Train Acc", color="#2ca02c")
ax2.plot(epochs_ef, ef_val_acc, marker="s", markersize=3.5, linewidth=2, linestyle="--", label="Val Acc", color="#ff7f0e")
ax2.set_title("Early Fusion — Classification Accuracy", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_xticks(range(1, 31, 2))
ax2.set_ylim(0.65, 1.02)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fase3_early_fusion.png", dpi=300)
plt.close()

print("File generati: fase1_age_regressor.png, fase2_pairwise_mlp.png, fase3_early_fusion.png")