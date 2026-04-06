import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import (
    classification_report, confusion_matrix,
)

# --------------------------
# Config
# --------------------------
# Set working directory to the correct path
os.chdir(r"C:\Users\Taula\Downloads\Ukraine cities\patches_siamese")
# Path for split dataset used for training 
NPZ_PATH = r"C:\Users\Taula\Downloads\Ukraine cities\patches_siamese\FinalBalanced_splits.npz"  

# Training hyperparameters
BATCH_SIZE = 64 # Samples per batch
EPOCHS = 40 # Number of epochs to train set high as some models require more 
PATIENCE = 5 # Early stopping patience, so it stops if no improvement on val after 5 epochs

# Model and output config
SAVE_BEST_TO = "best_siamese.pt" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
OUT_DIR = "outputs" 
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Utilities
# --------------------------

# Seed is set so that when comparing models and choosing different architectures they would stay consistent
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This ensures that there is reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set and call global seed for reproducibility
SEED = 42
seed_everything(SEED)

# Ensure that the patch arrays are fit into Pytorch format (N,C,H,W) 
def ensure_nchw(arr):
    """
    Ensure array is (N, C, H, W). 
     N (how many images)
     C (channels R/G/B/NIR)
     H (height)
     W (width) 
    Returns float32 sentinel-2 images scaled back down to [0,1] 
    """
    if arr.ndim == 3:
        # (N, H, W) -> add channel
        # Example: 100 grayscale images, 64x64 -> (100,1,64,64)
        arr = arr[:, None, :, :]
        
    elif arr.ndim == 4:
        # detect NHWC which is correct vs NCHW which needs fixing
        n, a, b, c = arr.shape
        
        # heuristic: if last dim is small (under 8) assume its RGB=3 AND middle dims look spatial (>8), assume NHWC
        if c <= 8 and a > 8 and b > 8:
            # convert from NHWC -> NCHW
            arr = np.transpose(arr, (0, 3, 1, 2))
        # else assume already NCHW
    else:
        raise ValueError(f"Unexpected array shape {arr.shape}, expected 3D or 4D.")

    # convert to float32 for PyTorch compatibility
    arr = arr.astype(np.float32)
    
    # sentinel-2 images divided by 10000 to get true surface reflectance
    arr = arr / 10000.0
        
    return arr

# --------------------------
# Dataset
# --------------------------
# Sets the dataset into the right format using the ensure function and then normalises the data
class PairsDataset(Dataset): 
    def __init__(self, X_before, X_after, y, mean, std): 
        # Store the "before" and "after" image patches
        self.before = ensure_nchw(X_before)   # keep as numpy, pytorch format (scales the Sentinel-2 images)
        self.after  = ensure_nchw(X_after)
        self.labels = y.astype(np.float32)

        # store per-channel normalisation stats
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(std,  dtype=torch.float32)[:, None, None]
        
        # Basic sanity checks after data is normalised
        assert self.before.ndim == 4 and self.after.ndim == 4, "Expected (N,C,H,W)"
        assert self.before.shape == self.after.shape, "Before/after shapes must match"

    def __len__(self): 
        # Return number of samples in the dataset
        return len(self.labels) 
    
    def __getitem__(self, index): 
        # Retrieve one sample (before patch, after patch, label)
        before_patch = torch.from_numpy(self.before[index])
        after_patch  = torch.from_numpy(self.after[index])

        # apply per-channel standarisation 
        before_patch = (before_patch - self.mean) / self.std
        after_patch  = (after_patch  - self.mean) / self.std

        label_patch  = torch.tensor(self.labels[index], dtype=torch.float32)
        return before_patch, after_patch, label_patch

# --------------------------
# Model
# --------------------------
class SiameseCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 1) Load pretrained ResNet-152 for the backbone
        # Use shared feature extractor for each input image (Before first then after)
        resnet = models.resnet152(weights="IMAGENET1K_V1")

        # 2) Adjust first convolution layer for when the input has more than 3 channels
        if in_channels != 3:
            with torch.no_grad():
                w = resnet.conv1.weight  # (64, 3, 7, 7) Pretrained RGB weights
                # New conv that accepts the specified number of channels
                new_conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # Copy existing RGB weights
                new_conv1.weight[:, :3] = w
                # If more than 3 channels, replicate mean of RGB weights
                if in_channels > 3:
                    extra = w.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
                    new_conv1.weight[:, 3:] = extra
                resnet.conv1 = new_conv1

        # 3) Keep features up to the global average pooling layer which makes the output 2048 dimensional vector
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # -> (N, 2048, 1, 1)

        # 4) MLP head
        # Each image goes through ResNet -> 2048-dim feature vector
        # For a pair of embeddings we combine them in 4 ways: [before, after, difference and product]
        # that makes 2048 * 4 = 8192 features total
        # The head is a small feed-forward network that takes this 8192-dim vector which goes through layers to final logit
        self.head = nn.Sequential(
            nn.Linear(8192, 4096), nn.BatchNorm1d(4096), nn.GELU(), nn.Dropout(0.6),
            nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 256),   nn.BatchNorm1d(256),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward_once(self, x):
        # Pass a single image patch through the ResNet feature extractor
        feature_map = self.features(x)         
        # Flatten the feature map into a vector 
        return feature_map.view(feature_map.size(0), -1) 

    def forward(self, before, after):
        # Extract 2048-dim embeddings for both inputs of shared weight backbone
        feature_before, feature_after = self.forward_once(before), self.forward_once(after)
        combined_features = torch.cat([feature_before, feature_after, torch.abs(feature_before - feature_after), feature_before * feature_after], dim=1) #(N, 8192)
        # pass through head and 1 logit per pair and squeeze removes extra dimension so output is one dimension for BCEWithLogitsLoss
        return self.head(combined_features).squeeze(1)
    
# --------------------------
# Data loading
# --------------------------
def load_data(npz_path):
    # Load train/val/test splits
    data = np.load(npz_path)
    # Xb = "before" image patches
    # Xa = "after" image patches
    # y = labels
    # X = xb and xa stacked into X for normalisation
    
    return (
        data["Xb_train"], data["Xa_train"], data["y_train"],   # before, after, labels (train)
        data["Xb_val"],   data["Xa_val"],   data["y_val"],     # before, after, labels (val)
        data["Xb_test"],  data["Xa_test"],  data["y_test"]     # before, after, labels (test)
    )

# normalisation over training set only
# axis 0 = N (how many images)
# axis 1 = C (channels R/G/B/NIR)
# axis 2 = H (height)
# axis 3 = W (width)

def compute_mean_std(Xb_train, Xa_train):
    # Use the same preprocessing as the Dataset 
    Xb = ensure_nchw(Xb_train)
    Xa = ensure_nchw(Xa_train)
    # all training patches together for normalisation
    X  = np.concatenate([Xb, Xa], axis=0)      
    # mean = per channel RGBN mean over batch and spatial dims 
    mean = X.mean(axis=(0, 2, 3))              
    # std = per channel RGBN std over batch and spatial dims 
    std  = X.std(axis=(0, 2, 3)) + 1e-8        
    return mean, std

def make_loaders(npz_path, batch_size=BATCH_SIZE):
    # Upack splits
    (before_train, after_train, labels_train,
     before_val,   after_val,   labels_val,
     before_test,  after_test,  labels_test) = load_data(npz_path)

    # Detect channels after your preprocessing
    in_ch = ensure_nchw(before_train[:1]).shape[1]

    # Compute mean/std from training only (after ensure_nchw scaling)
    mean, std = compute_mean_std(before_train, after_train)

    # Wrap into Dataset objects (use the same stats for all splits)
    ds_train = PairsDataset(before_train, after_train, labels_train, mean, std)
    ds_val   = PairsDataset(before_val,   after_val,   labels_val,   mean, std)
    ds_test  = PairsDataset(before_test,  after_test,  labels_test,  mean, std)

    # Create PyTorch dataloaders for batching and shuffeling
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, pin_memory=True)

    return dl_train, dl_val, dl_test, in_ch

# --------------------------
# Training & Evaluation
# --------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE, device=DEVICE, save_path=SAVE_BEST_TO):
    criterion = nn.BCEWithLogitsLoss() # Binary classification loss (logits + sigmoid inside the loss function)
    # Resnet learning rate lower as its pretrained so its for fine tuning
    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": 1e-4},  # backbone ResNet
        {"params": model.head.parameters(), "lr": 1e-3}       # Siamese head
    ], weight_decay=1e-4)  # regularization

    best_val_loss = float("inf") # track lowest validation loss 
    patience_counter = 0
    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for before_patches, after_patches, labels_batch in train_loader:
            before_patches, after_patches, labels_batch = (
                before_patches.to(device), 
                after_patches.to(device), 
                labels_batch.to(device)
            )

            optimizer.zero_grad()
            
            # Forward + loss
            logits = model(before_patches, after_patches)
            loss = criterion(logits, labels_batch)  
            
            # Backpropagation and weight update
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5) # prevent exploding gradients by not allowing the update step to be too big
            optimizer.step()
            
            # Track training metrics
            run_loss += loss.item() * labels_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels_batch.long()).sum().item()
            total += labels_batch.size(0)

        train_loss = run_loss / max(1, total)
        train_acc = correct / max(1, total)

        # ---- Validate ----
        model.eval()
        val_loss_sum, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for before_patches, after_patches, labels_batch in val_loader:
                before_patches, after_patches, labels_batch = (
                    before_patches.to(device), 
                    after_patches.to(device), 
                    labels_batch.to(device)
                )
                logits = model(before_patches, after_patches)
                loss = criterion(logits, labels_batch)
                val_loss_sum += loss.item() * labels_batch.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                v_correct += (preds == labels_batch.long()).sum().item()
                v_total += labels_batch.size(0)

        val_loss = val_loss_sum / max(1, v_total)
        val_acc = v_correct / max(1, v_total)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        # ---- Early stopping ----
        # if validation does not improve then save best model so far else add another to the patience counter which starts at 5
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
   

# -- Evaluation --
def evaluate_siamese_from_arrays(probs_array, labels_array, out_dir=OUT_DIR):
    # Turn probabilities into binary predictions (threshold 0.5)
    pred_labels = (probs_array > 0.5).astype(int)

    # Accuracy for each
    acc = (pred_labels == labels_array).mean()
    print(f"\n Test Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels_array, pred_labels, labels=[0, 1])
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Save both raw counts and normalised confusion matrix
    save_confusion(cm, ["No-Damage", "Damage"],
                   os.path.join(out_dir, "confusion_matrix_counts.png"),
                   normalize=False)
    save_confusion(cm, ["No-Damage", "Damage"],
                   os.path.join(out_dir, "confusion_matrix_normalized.png"),
                   normalize=True)

    # Classification report
    report = classification_report(
        labels_array,
        pred_labels,
        target_names=["No-Damage", "Damage"],
        output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(labels_array, pred_labels, target_names=["No-Damage", "Damage"]))

    # Save metrics to JSON
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    

    return acc, cm, report, labels_array, pred_labels, probs_array

@torch.no_grad()
def _probs_and_labels(model, loader, device):
    # Run the model on a loader and return all probabilities + true labels
    model.eval()
    all_probs, all_labels = [], []
    
    for before_patches, after_patches, labels_batch in loader:
        # move inputs to the device and labels stay on CPU
        before_patches, after_patches = before_patches.to(device), after_patches.to(device)
        logits = model(before_patches, after_patches)
        # Store the probabilities and store true labels
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels_batch.numpy())
        
    # Flatten into arrays
    probs_array = np.concatenate(all_probs)             # (N,) predicted probalities
    labels_array = np.concatenate(all_labels).astype(int)  # 0/1 labels
    return probs_array, labels_array

# Save confusion matrix
def save_confusion(cm, classes, save_path, normalize=False, title="Confusion Matrix"):
    # ensure folder exists
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if normalize:
        cm_display = cm.astype(np.float64)
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_display = cm_display / row_sums
        fmt = ".2f"
    else:
        cm_display = cm.astype(np.int64)
        fmt = "d"

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
        title=title + (" (normalized)" if normalize else " (counts)")
    )
    vmax = cm_display.max() if cm_display.size else 1
    vmin = cm_display.min() if cm_display.size else 0
    thresh = (vmax + vmin) / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            ax.text(
                j, i, format(cm_display[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black"
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")

# --------------------------
# Run
# --------------------------
def main():
    # Data
    train_loader, val_loader, test_loader, in_ch = make_loaders(NPZ_PATH, BATCH_SIZE)
    print(f"Train {len(train_loader.dataset)} | Val {len(val_loader.dataset)} | Test {len(test_loader.dataset)} | Channels={in_ch}")

    # Model
    model = SiameseCNN(in_channels=in_ch)

    # Train
    train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE, device=DEVICE, save_path=SAVE_BEST_TO)

    # Load best model weights once
    state = torch.load(SAVE_BEST_TO, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)

    # Get probabilities & labels once per split
    test_probs, test_labels = _probs_and_labels(model, test_loader, DEVICE)

    # Evaluate
    evaluate_siamese_from_arrays(test_probs, test_labels, OUT_DIR)

    # Save predictions + labels for later comparisons
    np.savez_compressed(
        os.path.join(OUT_DIR, "model_outputs.npz"),
        labels=test_labels,
        probs=test_probs
    )
       
# Run script
if __name__ == "__main__":
        main()
    