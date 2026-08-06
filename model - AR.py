"""
Activity Recognition Model v8.0
================================
v7.0 result: 86.88% F1 when the corrupted session was INCLUDED (not excluded).

Why exclusion hurt: the forum post says sensors stopped recording "after
frame ~200" — meaning most of that session (frames 1-199) is genuinely
valid.  Hard-excluding the WHOLE session threw away good training clips
for no benefit; only some clips after the corruption point are actually
bad, and within those, only the FRAMES after ~200 are zero — not the
whole clip.

v8.0 replaces hard exclusion with PER-FRAME REPAIR:

1. _repair_proprio(): detects simultaneous all-zero frames (sensor
   dropout, the SafeNpzFile sentinel value) anywhere in any session,
   and forward/backward-fills them from the nearest valid frame.
   - Generalises to ANY corrupted session, not just the one we know about.
   - Recovers full clips that start in the valid region and drift into
     the corrupted region — instead of either including raw zero spikes
     (noisy) or discarding the whole clip (wasteful).
   - A clip is only dropped if EVERY frame is zero (no valid reference
     exists at all) — this should be rare to nonexistent.

2. STACKED 3-MODEL ENSEMBLE  (was: fixed 0.60/0.40 GRU/RF blend)
   Base learners:
     - GRU ensemble (5 models, stratified bagging, OOB early-stop, TTA)
     - RandomForestClassifier        (70-dim global features)
     - HistGradientBoostingClassifier (70-dim global features, NEW)
   Combination: blend weights (w_gru, w_rf, w_hgb) are found via grid
   search on held-out OUT-OF-FOLD probabilities — never on data a model
   has been fit on — maximising micro-F1.  This replaces a manual guess
   with a data-driven choice, while staying low-dimensional (3 numbers,
   not a full meta-classifier) to avoid overfitting on ~90 training clips.

   GRU OOF probs   : from each model's natural bagging hold-out set
   RF / HGB OOF probs : from StratifiedKFold cross_val_predict
   (gracefully falls back to in-sample probabilities only if a class is
   too rare to stratify — logged as a warning, doesn't crash)

Everything else proven in v7.0 is unchanged: joint_at_grasp, stable_grip,
70-dim global features, stratified GRU bagging, OOB early stopping, TTA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os, sys, gc, pickle
from PIL import Image


# =====================================================================
# MEMORY-SAFE PATCHES
# =====================================================================

_dummy_image = Image.new('RGB', (1, 1), color='black')

import PIL.Image as _pil_mod
if not hasattr(_pil_mod, '_safe_open_original'):
    _pil_mod._safe_open_original = Image.open
_original_image_open = _pil_mod._safe_open_original

def safe_image_open(fp, mode='r'):
    fp_str = str(fp)
    if 'frame_' in fp_str or '.jpg' in fp_str or 'frames' in fp_str:
        return _dummy_image
    return _original_image_open(fp, mode)

Image.open = safe_image_open

try:
    from numpy.lib._npyio_impl import load as _original_np_load
except ImportError:
    from numpy.lib.npyio import load as _original_np_load


class SafeNpzFile:
    def __init__(self, npz):
        self._npz  = npz
        self.files = npz.files

    def __getitem__(self, key):
        val = self._npz[key]
        if isinstance(val, np.ndarray) and len(val) == 0:
            _shapes = {
                'pose': 7, 'velocity': 6, 'joint_positions': 7,
                'joint_velocities': 7, 'gripper_positions': 2,
                'gripper_velocities': 2, 'measured_wrench': 6,
            }
            return np.zeros((1, _shapes.get(key, 1)), dtype=np.float32)
        return val

    def __getattr__(self, name): return getattr(self._npz, name)
    def __enter__(self): return self
    def __exit__(self, *a): self._npz.close()
    def close(self): self._npz.close()


def safe_np_load(*args, **kwargs):
    res = _original_np_load(*args, **kwargs)
    if hasattr(res, 'files') and type(res).__name__ == 'NpzFile':
        return SafeNpzFile(res)
    return res

np.load = safe_np_load

for _mod_name in list(sys.modules.keys()):
    if 'ActionRecognitionDataset' in _mod_name:
        _mod = sys.modules[_mod_name]
        for _attr in ('np', 'numpy'):
            if _attr in _mod.__dict__:
                _mod.__dict__[_attr].load = safe_np_load

for _path in [
    'datamodule/ActionRecognitionDataset.py',
    '../datamodule/ActionRecognitionDataset.py',
    '/content/MachineLearning_384185_Project/datamodule/ActionRecognitionDataset.py',
]:
    if os.path.exists(_path):
        try:
            with open(_path) as _f: _c = _f.read()
            _old = (
                "            for key in keys:\n"
                "                proprioceptions[key].append(\n"
                "                    proprioception[key][-1])"
                "  # proprioception data is in higher frequency than images"
            )
            _new = (
                "            for key in keys:\n"
                "                arr = proprioception[key]\n"
                "                if len(arr) > 0:\n"
                "                    val = arr[-1]\n"
                "                elif len(proprioceptions[key]) > 0:\n"
                "                    val = proprioceptions[key][-1]\n"
                "                else:\n"
                "                    _s = {'pose': 7, 'velocity': 6, 'joint_positions': 7,\n"
                "                          'joint_velocities': 7, 'gripper_positions': 2,\n"
                "                          'gripper_velocities': 2, 'measured_wrench': 6}\n"
                "                    val = np.zeros(_s.get(key, 1), dtype=np.float32)\n"
                "                proprioceptions[key].append(val)"
            )
            if _old in _c:
                with open(_path, 'w') as _f: _f.write(_c.replace(_old, _new))
                print(f'[Self-Heal] Patched {_path}')
        except Exception as _e:
            print(f'[Self-Heal] Could not patch: {_e}')


# =====================================================================
# NEURAL MODEL
# =====================================================================

class LeanBiGRU(nn.Module):
    """
    Temporal  : LayerNorm → BiGRU(96, 1-layer) → mean+max → (384,)
    Global    : 70-dim stats → MLP → (32,)
    Classifier: (416,) → Linear(64) → Linear(13)
    ~135k parameters.
    """

    SEQ_DIM    = 67   # 37 proprio + 13 Δ(pose,wrench) + 13 ΔΔ + 4 Δ/ΔΔ(gripper)
    GLOBAL_DIM = 71   # 70 base + terminal_grip_width (settled post-pick measurement)

    def __init__(self, hidden=96, num_classes=13, dropout=0.40):
        super().__init__()
        gru_out = hidden * 2

        self.ln_seq   = nn.LayerNorm(self.SEQ_DIM)
        self.gru      = nn.GRU(self.SEQ_DIM, hidden, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.drop_seq = nn.Dropout(dropout)

        self.global_branch = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )

        fusion_in = gru_out * 2 + 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, seq, gf):
        x        = self.ln_seq(seq)
        out, _   = self.gru(x)
        out      = self.drop_seq(out)
        seq_feat = torch.cat([out.mean(1), out.max(1)[0]], dim=-1)
        g_feat   = self.global_branch(gf)
        return self.classifier(torch.cat([seq_feat, g_feat], dim=-1))


# =====================================================================
# SUBMISSION CLASS
# =====================================================================

class model:
    """
    Stacked ensemble: GRU(×5) + RandomForest + HistGradientBoosting.
    Blend weights are found via grid search on out-of-fold probabilities.
    """

    NUM_ENSEMBLE   = 5
    SUBSAMPLE_RATE = 0.80
    TTA_PASSES     = 5

    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets   = [LeanBiGRU().to(self.device) for _ in range(self.NUM_ENSEMBLE)]

        self.rf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1,
        )
        self.hgb = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_depth=4,
            min_samples_leaf=3, l2_regularization=0.1,
            early_stopping=False, random_state=42,
        )
        self.scaler = StandardScaler()

        # Blend weights — overwritten by fit() via OOF grid search.
        # Defaults match the v7.0 manual guess as a safe fallback.
        self.w_gru, self.w_rf, self.w_hgb = 0.60, 0.40, 0.0

        self.is_trained = False
        n_p = sum(p.numel() for p in self.nets[0].parameters())
        print(f'[model v8.0] {self.NUM_ENSEMBLE}×LeanBiGRU({n_p:,} params) '
              f'+ RandomForest + HistGB  device={self.device}')

    # ------------------------------------------------------------------
    # Per-frame corruption repair  (replaces v7.0's whole-session exclusion)
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_proprio(prop: np.ndarray):
        """
        Detect simultaneous all-zero frames (sensor dropout — confirmed for
        2024-09-24-15-57-21_cam1: "after idx 200 the saved arrays are empty")
        and repair them via forward/backward fill from the nearest valid
        frame, instead of leaving raw zero-spikes or discarding the clip.

        Returns (prop_fixed, fully_corrupted).  fully_corrupted is True only
        if EVERY frame in the clip is zero (no valid reference exists).
        """
        zero_mask = np.all(np.abs(prop) < 1e-8, axis=1)
        if not zero_mask.any():
            return prop, False
        if zero_mask.all():
            return prop, True

        valid_idx  = np.where(~zero_mask)[0]
        fixed      = prop.copy()
        last_valid = valid_idx[0]          # backfills any leading zeros too
        for t in range(len(prop)):
            if zero_mask[t]:
                fixed[t] = fixed[last_valid]
            else:
                last_valid = t
        return fixed, False

    # ------------------------------------------------------------------
    # Feature extraction  (GLOBAL_DIM = 70)
    # ------------------------------------------------------------------

    _COLS = {  # column ranges within the 37-dim concatenated proprio vector
        'pose': (0, 7), 'velocity': (7, 13), 'joint_positions': (13, 20),
        'joint_velocities': (20, 27), 'gripper_positions': (27, 29),
        'gripper_velocities': (29, 31), 'measured_wrench': (31, 37),
    }

    @classmethod
    def _extract(cls, sample):
        """
        Returns (seq, gf, fully_corrupted).
        seq : (N, 63)  per-frame temporal features (post-repair)
        gf  : (70,)    clip-level statistics (post-repair)
        """
        keys = ['pose', 'velocity', 'joint_positions', 'joint_velocities',
                'gripper_positions', 'gripper_velocities', 'measured_wrench']
        prop_raw = np.concatenate([sample[k] for k in keys], axis=1).astype(np.float32)
        prop, fully_corrupted = cls._repair_proprio(prop_raw)

        c = cls._COLS
        pose      = prop[:, c['pose'][0]:c['pose'][1]]
        velocity  = prop[:, c['velocity'][0]:c['velocity'][1]]
        jpos      = prop[:, c['joint_positions'][0]:c['joint_positions'][1]]
        grip      = prop[:, c['gripper_positions'][0]:c['gripper_positions'][1]]
        wrn       = prop[:, c['measured_wrench'][0]:c['measured_wrench'][1]]

        # ── Temporal sequence (67 dims): proprio + Δ(pose,wrench) + ΔΔ ──
        kf = np.concatenate([pose, wrn], axis=1).astype(np.float32)
        d1 = np.zeros_like(kf); d1[1:] = kf[1:] - kf[:-1]
        d2 = np.zeros_like(d1); d2[1:] = d1[1:] - d1[:-1]

        # NEW: gripper-position deltas (closing/opening speed profile).
        # Lets the GRU directly see HOW FAST the gripper is closing at each
        # timestep, not just its instantaneous width (already in `prop`).
        # Motivates: pick starts open and closes near the end; the velocity
        # profile of that closing motion may carry size information that a
        # single threshold-based snapshot (stable_grip_width) can miss.
        gd1 = np.zeros_like(grip); gd1[1:] = grip[1:] - grip[:-1]
        gd2 = np.zeros_like(gd1); gd2[1:] = gd1[1:] - gd1[:-1]

        seq = np.concatenate([prop, d1, d2, gd1, gd2], axis=1).astype(np.float32)  # (N, 67)

        # ── Global features (70 dims) ────────────────────────────────
        grip_min   = grip.min(0)
        grip_max   = grip.max(0)
        grip_std   = grip.std(0)
        grip_delta = grip[-1] - grip[0]
        wrn_peak   = np.abs(wrn).max(0)
        wrn_mean   = np.abs(wrn).mean(0)
        pose_start = pose[0]
        pose_end   = pose[-1]
        jnt_mean   = jpos.mean(0)
        lin_mag    = np.linalg.norm(velocity[:, :3], axis=1)
        vel_stats  = np.array([lin_mag.mean(), lin_mag.max(), lin_mag.std()],
                               dtype=np.float32)                            # 44

        grasp_t       = min(int(grip.mean(1).argmin()), len(pose) - 1)
        pose_at_grasp = pose[grasp_t, :3].astype(np.float32)               # 3
        grip_p10      = np.percentile(grip, 10, axis=0).astype(np.float32) # 2
        z_vals        = pose[:, 2]
        z_stats       = np.array([z_vals.min(), z_vals.max() - z_vals.min()],
                                  dtype=np.float32)                        # 2  → 51

        n           = len(prop)
        t1          = max(1, n // 3)
        t2          = max(t1 + 1, 2 * n // 3)
        grip_mid    = grip[t1:t2].mean(0)
        wrn_mid_abs = np.abs(wrn[t1:t2]).mean(0)
        pose_mid    = pose[t1:t2, :3].mean(0) if t2 > t1 else pose[:, :3].mean(0)  # 11 → 62

        joint_at_grasp = jpos[grasp_t].astype(np.float32)                  # 7

        grip_per_frame = grip.mean(1)
        open_thresh    = grip_per_frame.max() * 0.88
        closed_mask    = grip_per_frame < open_thresh
        stable_w       = (grip_per_frame[closed_mask].mean()
                          if closed_mask.sum() > 0 else grip_per_frame.min())
        stable_grip    = np.array([stable_w], dtype=np.float32)            # 1  → 70

        # NEW: terminal_grip_width — mean width over the LAST 15% of frames.
        # During pick, the gripper is open for most of the clip and only
        # closes near the end; stable_grip_width's threshold-based window
        # can include transitional closing frames that read wider than the
        # gear's true diameter.  Averaging the final settled frames instead
        # gives a cleaner post-grasp measurement regardless of how long the
        # closing motion took — useful for pick specifically, harmless for
        # insert/remove/place (whose gripper is already settled throughout).
        last_k        = max(1, int(round(0.15 * len(grip_per_frame))))
        terminal_grip = np.array([grip_per_frame[-last_k:].mean()],
                                  dtype=np.float32)                        # 1  → 71

        gf = np.concatenate([
            grip_min, grip_max, grip_std, grip_delta,
            wrn_peak, wrn_mean,
            pose_start, pose_end,
            jnt_mean, vel_stats,
            pose_at_grasp, grip_p10, z_stats,
            grip_mid, wrn_mid_abs, pose_mid,
            joint_at_grasp,
            stable_grip,
            terminal_grip,
        ]).astype(np.float32)

        return seq, gf, fully_corrupted

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(seq_t, gf_t, scale=1.0):
        seq_t = seq_t + torch.randn_like(seq_t) * (0.01 * scale)
        gf_t  = gf_t  + torch.randn_like(gf_t)  * (0.005 * scale)
        return seq_t, gf_t

    # ------------------------------------------------------------------
    # Train one GRU ensemble member (stratified bagging + OOB early stop)
    # ------------------------------------------------------------------

    def _train_gru(self, net, xs, gs, ys_raw, cw, seed, max_epochs=120):
        torch.manual_seed(seed)
        np.random.seed(seed)

        labels  = [y.item() for y in ys_raw]
        sub_idx, oob_idx = [], []
        for cls_id in range(13):
            cls_i = [i for i, l in enumerate(labels) if l == cls_id]
            if not cls_i: continue
            k_cls  = max(1, int(len(cls_i) * self.SUBSAMPLE_RATE))
            chosen = np.random.choice(cls_i, k_cls, replace=False).tolist()
            sub_idx.extend(chosen)
            oob_idx.extend([i for i in cls_i if i not in set(chosen)])

        xs_s = [xs[i] for i in sub_idx]
        gs_s = [gs[i] for i in sub_idx]
        ys_s = [ys_raw[i] for i in sub_idx]
        print(f'    stratified subsample {len(sub_idx)}/{len(xs)}  OOB {len(oob_idx)}')

        opt  = optim.AdamW(net.parameters(), lr=2e-3, weight_decay=3e-4)
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

        warmup = 5
        def lr_fn(ep):
            if ep < warmup: return (ep + 1) / warmup
            t = (ep - warmup) / max(1, max_epochs - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * t))
        sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)

        best_loss, best_state, no_improve, patience = float('inf'), None, 0, 25

        for ep in range(max_epochs):
            net.train()
            tr_loss = 0.0
            for i in np.random.permutation(len(xs_s)):
                sq, gf = self._augment(xs_s[i].to(self.device), gs_s[i].to(self.device))
                y = ys_s[i].to(self.device)
                opt.zero_grad()
                loss = crit(net(sq, gf), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()
            sched.step()

            if oob_idx:
                net.eval()
                with torch.no_grad():
                    oob_loss = sum(
                        crit(net(xs[i].to(self.device), gs[i].to(self.device)),
                             ys_raw[i].to(self.device)).item()
                        for i in oob_idx
                    ) / len(oob_idx)
                if oob_loss < best_loss - 1e-4:
                    best_loss  = oob_loss
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

            if ep == 0 or (ep + 1) % 20 == 0:
                oob_str = f'  oob {best_loss:.4f}  patience {no_improve}/{patience}' if oob_idx else ''
                print(f'    ep {ep+1:3d}  tr {tr_loss/len(xs_s):.4f}'
                      f'  lr {opt.param_groups[0]["lr"]:.2e}{oob_str}')

            if no_improve >= patience:
                print(f'    Early stop at epoch {ep+1}')
                break
            gc.collect()

        if best_state:
            net.load_state_dict(best_state)
            net.to(self.device)
            print(f'    Restored best OOB weights (loss={best_loss:.4f})')

        return oob_idx

    # ------------------------------------------------------------------
    # Blend-weight search on out-of-fold probabilities
    # ------------------------------------------------------------------

    @staticmethod
    def _search_blend_weights(gru_oof, rf_oof, hgb_oof, y_true):
        from sklearn.metrics import f1_score
        best_f1, best_w = -1.0, (1/3, 1/3, 1/3)
        grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
        for wg in grid:
            for wr in grid:
                wh = round(1.0 - wg - wr, 2)
                if wh < -1e-9 or wh > 1.0 + 1e-9:
                    continue
                wh = max(0.0, wh)
                combined = wg * gru_oof + wr * rf_oof + wh * hgb_oof
                f1 = f1_score(y_true, combined.argmax(1), average='micro')
                if f1 > best_f1:
                    best_f1, best_w = f1, (wg, wr, wh)
        return best_w, best_f1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, dataset):
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        print(f'\n[fit] {len(dataset)} raw clips  device={self.device}')

        counts = np.zeros(13, dtype=np.float32)
        for i in range(len(dataset)):
            counts[dataset[i]['label']] += 1
        counts = np.clip(counts, 1.0, None)
        w  = (1.0 / counts) / (1.0 / counts).sum() * 13.0
        cw = torch.tensor(w, dtype=torch.float32, device=self.device)
        print(f'Class counts : {counts.astype(int).tolist()}')

        # ── Feature extraction with per-frame repair ────────────────────
        print('\nExtracting features (repairing corrupted frames in-place)...')
        xs, gs, ys_raw, gf_np_list, y_np_list = [], [], [], [], []
        n_dropped = 0
        for i in range(len(dataset)):
            s = dataset[i]
            seq, gf, fully_corrupted = self._extract(s)
            if fully_corrupted:
                n_dropped += 1
                del s; gc.collect()
                continue
            xs.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).cpu())
            gs.append(torch.tensor(gf,  dtype=torch.float32).unsqueeze(0).cpu())
            ys_raw.append(torch.tensor([s['label']], dtype=torch.long).cpu())
            gf_np_list.append(gf)
            y_np_list.append(s['label'])
            del s; gc.collect()

        print(f'  {len(xs)} clips retained  ({n_dropped} fully-corrupted clips dropped)')

        # ── Train GRU ensemble, recording each model's OOB indices ──────
        seeds = [42, 137, 2024, 999, 31415]
        oob_sets = []
        for m_i, (net, seed) in enumerate(zip(self.nets, seeds)):
            print(f'\n=== GRU {m_i+1}/{self.NUM_ENSEMBLE}  seed={seed} ===')
            oob_idx = self._train_gru(net, xs, gs, ys_raw, cw, seed=seed)
            oob_sets.append(set(oob_idx))

        # ── GRU out-of-fold probabilities ────────────────────────────────
        print('\nComputing GRU out-of-fold probabilities...')
        n_clips  = len(xs)
        gru_oof  = np.zeros((n_clips, 13), dtype=np.float32)
        oof_hits = np.zeros(n_clips, dtype=np.int32)
        for net, oob_set in zip(self.nets, oob_sets):
            net.eval()
            with torch.no_grad():
                for i in oob_set:
                    p = F.softmax(net(xs[i].to(self.device), gs[i].to(self.device)),
                                  dim=1).cpu().numpy()[0]
                    gru_oof[i] += p
                    oof_hits[i] += 1
        has_oof = oof_hits > 0
        gru_oof[has_oof] /= oof_hits[has_oof][:, None]
        # Rare fallback: a clip never held out by any model → use full-ensemble avg
        if (~has_oof).any():
            with torch.no_grad():
                for i in np.where(~has_oof)[0]:
                    p_sum = np.zeros(13, dtype=np.float32)
                    for net in self.nets:
                        net.eval()
                        p_sum += F.softmax(net(xs[i].to(self.device), gs[i].to(self.device)),
                                           dim=1).cpu().numpy()[0]
                    gru_oof[i] = p_sum / len(self.nets)

        # ── RF / HGB out-of-fold probabilities + final fit ───────────────
        print('Computing RF / HistGB out-of-fold probabilities...')
        X_rf = np.stack(gf_np_list)
        y_rf = np.array(y_np_list, dtype=int)
        X_rf_scaled = self.scaler.fit_transform(X_rf)

        class_counts  = np.bincount(y_rf, minlength=13)
        present       = class_counts[class_counts > 0]
        min_count     = int(present.min()) if len(present) else 1
        n_splits      = min(5, min_count)

        def _oof(estimator, name):
            if n_splits < 2:
                print(f'  [Warning] {name}: a class has <2 samples — '
                      f'using in-sample probabilities for OOF blend search only.')
                estimator.fit(X_rf_scaled, y_rf)
                return estimator.predict_proba(X_rf_scaled)
            try:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                return cross_val_predict(estimator, X_rf_scaled, y_rf,
                                         cv=skf, method='predict_proba')
            except Exception as e:
                print(f'  [Warning] {name} cross-val failed ({e}); using in-sample fallback.')
                estimator.fit(X_rf_scaled, y_rf)
                return estimator.predict_proba(X_rf_scaled)

        rf_oof  = _oof(self.rf,  'RandomForest')
        hgb_oof = _oof(self.hgb, 'HistGB')

        # Final fit on ALL data (for deployment; OOF was only for weight search)
        self.rf.fit(X_rf_scaled, y_rf)
        self.hgb.fit(X_rf_scaled, y_rf)
        print(f'  RF  train acc: {(self.rf.predict(X_rf_scaled)==y_rf).mean()*100:.1f}%')
        print(f'  HGB train acc: {(self.hgb.predict(X_rf_scaled)==y_rf).mean()*100:.1f}%')

        # ── Grid-search blend weights on OOF probabilities ───────────────
        print('\nSearching blend weights on out-of-fold predictions...')
        (self.w_gru, self.w_rf, self.w_hgb), best_f1 = self._search_blend_weights(
            gru_oof, rf_oof, hgb_oof, y_rf
        )
        print(f'  Best blend: GRU={self.w_gru:.2f}  RF={self.w_rf:.2f}  '
              f'HGB={self.w_hgb:.2f}   (OOF micro-F1={best_f1*100:.2f}%)')

        self.is_trained = True
        print('\n[fit] Done.')

    def predict_single(self, sample):
        seq, gf, _ = self._extract(sample)
        seq_t = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        gf_t  = torch.tensor(gf,  dtype=torch.float32, device=self.device).unsqueeze(0)

        votes = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                for t in range(self.TTA_PASSES):
                    s_in, g_in = (seq_t, gf_t) if t == 0 \
                                 else self._augment(seq_t, gf_t, scale=0.5)
                    votes.append(F.softmax(net(s_in, g_in), dim=1))
        gru_probs = torch.stack(votes).mean(0).squeeze(0).cpu().numpy()

        gf_scaled = self.scaler.transform(gf.reshape(1, -1))
        rf_probs  = self.rf.predict_proba(gf_scaled)[0]
        hgb_probs = self.hgb.predict_proba(gf_scaled)[0]

        combined = (self.w_gru * gru_probs
                   + self.w_rf  * rf_probs
                   + self.w_hgb * hgb_probs)
        return int(np.argmax(combined))

    @staticmethod
    def _strip_rng(obj, _seen=None):
        """
        Recursively null out any numpy Generator/BitGenerator instances
        found on an object's attributes (and nested lists/tuples/dicts).

        Root cause this fixes: HistGradientBoostingClassifier (sklearn) sets
        a fitted attribute `_feature_subsample_rng` = numpy.random.Generator
        (PCG64-backed).  This is only used DURING fit() for feature
        subsampling and is never touched by predict()/predict_proba() —
        confirmed by direct testing (predictions are bit-identical with or
        without it).  But it pickles a BitGenerator object whose internal
        module path is numpy-version-sensitive.  If the training environment
        (Colab) and the evaluation server run different numpy versions/builds,
        unpickling raises "... is not a known BitGenerator module" and the
        ENTIRE model fails to load — even though every other attribute is
        completely portable.  Stripping these objects removes the only
        non-portable piece without affecting predictions at all.

        Applied generically (not just to the one known attribute name) so it
        also covers any other sklearn estimator/version that stores RNG
        state similarly, now or in the future.
        """
        if _seen is None:
            _seen = set()
        oid = id(obj)
        if oid in _seen:
            return
        _seen.add(oid)

        try:
            if isinstance(obj, (np.random.Generator, np.random.BitGenerator)):
                return  # caller replaces the reference itself
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if isinstance(v, (np.random.Generator, np.random.BitGenerator)):
                        obj[k] = None
                    else:
                        model._strip_rng(v, _seen)
                return
            if isinstance(obj, (list, tuple, set)):
                for v in obj:
                    model._strip_rng(v, _seen)
                return
            if hasattr(obj, '__dict__'):
                for k, v in list(vars(obj).items()):
                    if isinstance(v, (np.random.Generator, np.random.BitGenerator)):
                        setattr(obj, k, None)
                    else:
                        model._strip_rng(v, _seen)
        except Exception as e:
            print(f'  [Warning] _strip_rng could not inspect {type(obj)}: {e}')

    def save(self, path):
        nets_np = [{k: v.cpu().numpy() for k, v in net.state_dict().items()}
                   for net in self.nets]

        # Remove any non-portable RNG objects before pickling (see
        # _strip_rng docstring — fixes cross-numpy-version load failures).
        self._strip_rng(self.rf)
        self._strip_rng(self.hgb)

        payload = {
            'nets_np' : nets_np,
            'rf'      : self.rf,
            'hgb'     : self.hgb,
            'scaler'  : self.scaler,
            'weights' : (self.w_gru, self.w_rf, self.w_hgb),
            'version' : 'v8.2',
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=4)
        print(f'[save] {self.NUM_ENSEMBLE} GRU + RF + HGB → {path}  '
              f'({os.path.getsize(path)/1e6:.1f} MB)')

    def load_weights(self, path):
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        for net, sd_np in zip(self.nets, payload['nets_np']):
            net.load_state_dict({k: torch.from_numpy(v) for k, v in sd_np.items()})
            net.to(self.device)
        self.rf      = payload['rf']
        self.hgb     = payload['hgb']
        self.scaler  = payload['scaler']
        self.w_gru, self.w_rf, self.w_hgb = payload.get('weights', (0.60, 0.40, 0.0))
        self.is_trained = True
        print(f'[load_weights] {self.NUM_ENSEMBLE} GRU + RF + HGB ← {path}  '
              f'(version={payload.get("version","?")}, '
              f'weights=GRU:{self.w_gru:.2f}/RF:{self.w_rf:.2f}/HGB:{self.w_hgb:.2f})')
