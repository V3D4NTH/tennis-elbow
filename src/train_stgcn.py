#!/usr/bin/env python3
"""
Enhanced Chebyshev ST-GCN trainer - trains BOTH wrist extension and wrist flexion models.
Reads your pickles (33 joints) → uses 16 → variable-length → saves both models.
"""
# ----------  imports  ---------------------------------------------------------
import torch, torch.nn as nn, torch.optim as optim, numpy as np, pickle, argparse, tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import time

# ----------  defaults (no args needed)  ---------------------------------------
DEFAULT_DATA_ROOT  = Path(r"C:\\Users\\itsth\\Downloads\\wrist_extension_processed-20251105T093718Z-1-001")
DEFAULT_EPOCHS     = 30
DEFAULT_BATCH      = 32
DEFAULT_LR         = 5e-4
DEFAULT_DROPOUT    = 0.5
DEFAULT_K          = 3
DEFAULT_OUTPUT_DIR = Path("models")  # Directory to save models

# Exercise configurations
EXERCISES = ["wrist_extension", "wrist_flexion"]

# ----------  MediaPipe skeleton (16 joints)  ---------------------------------
V = 16
EDGES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

# ----------  Chebyshev ST-GCN (K=3)  -----------------------------------------
class ChebConv(nn.Module):
    def __init__(self, c_in, c_out, K=3, bias=True):
        super().__init__()
        self.K, self.c_in, self.c_out = K, c_in, c_out
        self.weight = nn.Parameter(torch.Tensor(K, c_in, c_out))
        if bias: self.bias = nn.Parameter(torch.Tensor(c_out))
        else: self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None: nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (B, c_in, V, T)   adj: (K, V, V)
        support = torch.einsum("kvw,bcvt->bckvt", adj, x)          # (B, c_in, K, V, T)
        out = torch.einsum("bckvt,kco->bovt", support, self.weight)   # (B, c_out, V, T)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)                   # broadcast to last 3 dims
        return out


class STConvBlock(nn.Module):
    def __init__(self, c_in, c_out, K=3, t_kernel=5, dropout=0.3):
        super().__init__()
        self.spatial = ChebConv(c_in, c_out, K)
        self.temporal = nn.Sequential(
            nn.Conv2d(c_out, c_out, (1, t_kernel), padding=(0, t_kernel // 2)),
            nn.BatchNorm2d(c_out), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x, adj):
        x = self.spatial(x, adj)      # (B, c_out, V, T)
        x = self.temporal(x)          # (B, c_out, V, T)
        return x


class RealSTGCN(nn.Module):
    def __init__(self, c_in=3, num_classes=3, V=16, K=3, dropout=0.5):
        super().__init__()
        self.register_buffer("A", self._cheb_polynomials(self._build_adj(V), K))
        self.st1 = STConvBlock(c_in, 64,  K, dropout=dropout)
        self.st2 = STConvBlock(64,  128, K, dropout=dropout)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    @staticmethod
    def _build_adj(V):
        A = np.eye(V)
        for u, v in EDGES:
            u, v = min(u, V - 1), min(v, V - 1)
            A[u, v] = A[v, u] = 1
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(np.where(D > 0, D ** -0.5, 0))
        return torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float32)

    @staticmethod
    def _cheb_polynomials(adj, K):
        adj = adj.numpy()
        poly = [np.eye(adj.shape[0]), adj]
        for i in range(2, K):
            poly.append(2 * adj @ poly[-1] - poly[-2])
        return torch.stack([torch.tensor(p, dtype=torch.float32) for p in poly])

    def forward(self, x, lengths):
        # x: (B, T, V, C)  →  (B, C, V, T)
        B, T, V, C = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()
        # mask padded frames
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(1)
        x = x * mask
        # ST-blocks
        x = self.st1(x, self.A)
        x = self.st2(x, self.A)
        # global pool
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)


# ---------------------------  data  ------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, exercise: str, data_root: Path):
        self.root = data_root / exercise
        self.pkl_files = list(self.root.rglob("*.pkl"))
        if not self.pkl_files:
            raise FileNotFoundError(f"No pickles in {self.root}")
        print(f"  Found {len(self.pkl_files)} pickle files for {exercise}")

    def __len__(self):
        return len(self.pkl_files)

    def _label_from_name(self, name: str) -> int:
        nm = name.lower()
        return 1 if "bad" in nm else 2 if "mediocre" in nm else 0

    def __getitem__(self, idx):
        pkl = self.pkl_files[idx]
        with open(pkl, 'rb') as fh:
            seq = pickle.load(fh)
        if not seq:                      # EMPTY → skip flag
            return None, None
        label = self._label_from_name(pkl.stem)
        # keep only first 16 joints (x,y,z)
        tensor = torch.tensor([[frm[:16][i][:3] for i in range(16)] for frm in seq], dtype=torch.float32)
        return tensor, label


def collate_fn(batch):
    """Drop Nones, then pad variable-length sequences."""
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return (torch.empty(0, 0, 16, 3), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    tensors, labels = zip(*batch)
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
    padded  = pad_sequence(tensors, batch_first=True)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


# ---------------------------  training  --------------------------------------
def train_one_epoch(model, loader, adj, criterion, optimizer, device):
    model.train()
    running_loss, n_correct, n_total = 0.0, 0, 0
    for x, lengths, y in loader:
        if x.size(0) == 0: continue               # skip empty mini-batch
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, lengths)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_correct += (out.argmax(1) == y).sum().item()
        n_total += y.size(0)
    return running_loss / len(loader) if len(loader) > 0 else 0, n_correct / n_total if n_total > 0 else 0


def evaluate(model, loader, adj, criterion, device):
    model.eval()
    running_loss, n_correct, n_total = 0.0, 0, 0
    with torch.no_grad():
        for x, lengths, y in loader:
            if x.size(0) == 0: continue
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            out = model(x, lengths)
            loss = criterion(out, y)
            running_loss += loss.item()
            n_correct += (out.argmax(1) == y).sum().item()
            n_total += y.size(0)
    return running_loss / len(loader) if len(loader) > 0 else 0, n_correct / n_total if n_total > 0 else 0


def train_model_for_exercise(exercise, args, device):
    """Train a single model for a specific exercise."""
    print(f"\n{'='*70}")
    print(f"Training ST-GCN for {exercise.upper()}")
    print(f"{'='*70}")
    
    # Create output path for this model
    model_save_path = args.output_dir / f"stgcn_{exercise}_model.pth"
    
    # Load data
    try:
        ds = SequenceDataset(exercise, args.data_root)
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print(f"  Skipping {exercise} training")
        return None
    
    all_tensors, all_labels = [], []
    for t, l in ds:
        if t is not None:                      # skip empties
            all_tensors.append(t)
            all_labels.append(l)
    
    if len(all_tensors) == 0:
        print(f"⚠ Warning: No valid sequences found for {exercise}")
        return None
    
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    # Count samples per class
    unique_labels, counts = torch.unique(all_labels, return_counts=True)
    print(f"  Class distribution:")
    class_names = {0: "good", 1: "bad", 2: "mediocre"}
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"    {class_names.get(label, label)}: {count} samples")

    # Split data
    train_idx, val_idx = train_test_split(
        range(len(all_tensors)), test_size=0.2, stratify=all_labels, random_state=42
    )
    train_set = [(all_tensors[i], all_labels[i]) for i in train_idx]
    val_set   = [(all_tensors[i], all_labels[i]) for i in val_idx]

    print(f"  Training samples: {len(train_set)}")
    print(f"  Validation samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = RealSTGCN(c_in=3, num_classes=3, V=16, K=args.K, dropout=args.dropout).to(device)
    adj   = model.A
    criterion = nn.CrossEntropyLoss(weight=None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)

    # Training loop
    best_acc = 0
    print(f"\nStarting training for {exercise}...")
    
    # Use tqdm with description for this specific exercise
    pbar = tqdm.trange(1, args.epochs + 1, desc=f"{exercise.replace('_', ' ').title()}")
    
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, adj, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, adj, criterion, device)
        scheduler.step(val_acc)
        
        # Update progress bar description with current accuracy
        pbar.set_postfix({
            'train_acc': f'{train_acc*100:.1f}%',
            'val_acc': f'{val_acc*100:.1f}%',
            'best': f'{best_acc*100:.1f}%'
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            tqdm.tqdm.write(f"  → New best model for {exercise} saved ({val_acc*100:.2f}%)")

    print(f"\n✓ {exercise.upper()} Training Complete")
    print(f"  Best validation accuracy: {best_acc*100:.2f}%")
    print(f"  Model saved to: {model_save_path}")
    
    return {
        'exercise': exercise,
        'best_acc': best_acc,
        'model_path': model_save_path
    }


# ---------------------------  main  ------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN models for BOTH wrist exercises")
    parser.add_argument("--data_root",  type=Path,  default=DEFAULT_DATA_ROOT,  help="processed_data folder")
    parser.add_argument("--output_dir", type=Path,  default=DEFAULT_OUTPUT_DIR, help="directory to save models")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS,     help="training epochs per model")
    parser.add_argument("--batch",      type=int,   default=DEFAULT_BATCH,      help="batch size")
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR,         help="learning rate")
    parser.add_argument("--dropout",    type=float, default=DEFAULT_DROPOUT,    help="dropout")
    parser.add_argument("--K",          type=int,   default=DEFAULT_K,          help="Chebyshev order")
    parser.add_argument("--exercises",  nargs='+',  default=EXERCISES,         help="exercises to train (default: both)")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Exercises to train: {args.exercises}")
    
    # Track results
    results = []
    start_time = time.time()
    
    # Train model for each exercise
    for exercise in args.exercises:
        result = train_model_for_exercise(exercise, args, device)
        if result:
            results.append(result)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"\nModels trained:")
    for result in results:
        print(f"  • {result['exercise']}: {result['best_acc']*100:.2f}% accuracy")
        print(f"    → {result['model_path']}")
    
    if len(results) < len(args.exercises):
        print(f"\n⚠ Warning: {len(args.exercises) - len(results)} model(s) could not be trained due to missing data")


if __name__ == "__main__":
    main()