import numpy as np
import torch
import torch.nn as nn
import os, pickle

# np.load patched by Cell 1 of the notebook before this is ever imported.

class _BiGRU(nn.Module):
    """
    3-layer bidirectional GRU with dropout.
    Input : (batch, T, input_dim)
    Output: (batch, T, num_classes)
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)          # (B, T, 2*H)
        out = self.norm(out)
        return self.head(out)          # (B, T, C)


class model:
    _KEYS = ['pose', 'velocity', 'joint_positions', 'joint_velocities',
             'gripper_positions', 'gripper_velocities', 'measured_wrench']
    _NC  = 13
    _RAW = 37

    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.3):
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout
        self.input_dim  = self._RAW * 3        # raw + delta + delta-delta = 111
        self.mean = self.std = None
        self.is_trained = False
        self._build()

    def _build(self):
        self.net = _BiGRU(
            self.input_dim, self.hidden_dim, self._NC,
            num_layers=self.num_layers, dropout=self.dropout
        ).to(self.device)

    # ── features ──────────────────────────────────────────────────────────────
    def _raw(self, sample):
        return np.concatenate([sample[k] for k in self._KEYS], axis=-1)  # (T, 37)

    def _feat(self, sample):
        X = self._raw(sample)
        if self.mean is not None:
            X = (X - self.mean) / (self.std + 1e-8)
        dX  = np.diff(X,  axis=0, prepend=X[:1])
        ddX = np.diff(dX, axis=0, prepend=dX[:1])
        return np.concatenate([X, dX, ddX], axis=-1).astype(np.float32)  # (T, 111)

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, dataset, n_epochs=20, lr=5e-4, use_class_weights=True):
        n = len(dataset)

        print('Step 1/3  Normalisation stats ...')
        raw = np.concatenate([self._raw(dataset[i]) for i in range(n)], axis=0)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

        print(f'Step 2/3  Extracting features [{n} sessions] ...')
        Xs, ys = [], []
        for i in range(n):
            Xs.append(self._feat(dataset[i]))
            ys.append(dataset[i]['labels'])
            if (i+1) % 10 == 0 or (i+1) == n:
                print(f'          {i+1}/{n}')

        all_labels = np.concatenate(ys)
        if use_class_weights:
            counts = np.bincount(all_labels, minlength=self._NC).astype(float)
            w = 1.0 / (counts + 1); w = w / w.sum() * self._NC
            print(f'          Class weights (min={w.min():.2f}, max={w.max():.2f})')
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(w, dtype=torch.float32).to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        opt   = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/50)

        print(f'Step 3/3  Training on {self.device} for {n_epochs} epochs ...')
        self.net.train()
        idx = list(range(n))

        for ep in range(n_epochs):
            np.random.shuffle(idx)
            ep_loss = 0.0
            correct = total = 0

            for i in idx:
                noise = (np.random.randn(*Xs[i].shape) * 0.02).astype(np.float32)
                X = torch.tensor(Xs[i] + noise).unsqueeze(0).to(self.device)  # (1, T, 111)
                y = torch.tensor(ys[i], dtype=torch.long).to(self.device)     # (T,)

                opt.zero_grad()
                logits = self.net(X).squeeze(0)   # (T, 13)
                loss   = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

                ep_loss  += loss.item()
                correct  += (logits.argmax(-1) == y).sum().item()
                total    += len(y)

            sched.step()
            acc = correct / total * 100
            print(f'Ep {ep+1:2d}/{n_epochs} | loss={ep_loss/n:.4f} | '
                  f'train_acc={acc:.1f}% | lr={sched.get_last_lr()[0]:.2e}')

        self.is_trained = True

    # ── predict ───────────────────────────────────────────────────────────────
    def predict_sequence(self, sample):
        self.net.eval()
        X = torch.tensor(self._feat(sample)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.net(X).squeeze(0).argmax(-1).cpu().numpy()
        return self._clean(preds, min_len=5).astype(np.int64)

    def _clean(self, preds, min_len=5):
        arr = preds.copy(); changed = True
        while changed:
            changed = False; segs = []; i = 0
            while i < len(arr):
                j = i
                while j < len(arr) and arr[j] == arr[i]: j += 1
                segs.append((i, j, arr[i])); i = j
            for k, (s, e, c) in enumerate(segs):
                if e - s < min_len:
                    if k == 0:
                        nb = segs[1][2] if len(segs) > 1 else c
                    elif k == len(segs) - 1:
                        nb = segs[k-1][2]
                    else:
                        L, R = segs[k-1], segs[k+1]
                        nb = L[2] if (L[1]-L[0]) >= (R[1]-R[0]) else R[2]
                    if nb != c:
                        arr[s:e] = nb; changed = True; break
        return arr

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path):
        d = {
            'weights':    {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
            'mean':       self.mean,
            'std':        self.std,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout':    self.dropout,
            'input_dim':  self.input_dim,
        }
        with open(path, 'wb') as f: pickle.dump(d, f)
        print(f'Saved -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)')

    def load_weights(self, path):
        with open(path, 'rb') as f: d = pickle.load(f)
        self.mean       = d['mean']
        self.std        = d['std']
        self.hidden_dim = d.get('hidden_dim', self.hidden_dim)
        self.num_layers = d.get('num_layers', self.num_layers)
        self.dropout    = d.get('dropout',    self.dropout)
        self.input_dim  = d.get('input_dim',  self.input_dim)
        self._build()
        self.net.load_state_dict(
            {k: torch.tensor(v).to(self.device) for k, v in d['weights'].items()})
        self.is_trained = True
        print(f'Loaded <- {path}')
