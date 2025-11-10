# Libras — Landmark pipeline (MediaPipe)
import os, uuid, json, math, random, hashlib
from pathlib import Path
from typing import List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from PIL import Image, ImageDraw
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ----------------- Repro -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
tf.config.set_soft_device_placement(True)
print("TF:", tf.__version__)

# ----------------- Paths -----------------
ROOT = Path(os.environ.get("LIBRAS_ROOT", "/workspace")).resolve()
print("[ROOT]", ROOT)

DS1_TRAIN_DIR = ROOT / "dataset-letters1" / "dataset" / "training"
DS1_TEST_DIR  = ROOT / "dataset-letters1" / "dataset" / "test"
DS2_VAL_DIR   = ROOT / "dataset-letters2" / "dataset"

CACHE_DIR = ROOT / "landmarks_cache"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
LANDMARKS_CSV = CACHE_DIR / "landmarks_v2.csv"
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

HAND_TASK_PATH = str((ROOT / "hand_landmarker.task").resolve())
assert Path(HAND_TASK_PATH).exists(), f"hand_landmarker.task not found: {HAND_TASK_PATH}"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ----------------- Dataset helpers -----------------
def get_classes(train_dir: Path) -> List[str]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing dataset dir: {train_dir}")
    return sorted(d.name for d in train_dir.iterdir() if d.is_dir())

def list_images(root: Path, class_names: List[str]) -> List[Tuple[str, str]]:
    rows = []
    for c in class_names:
        p = root / c
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.suffix.lower() in EXTS:
                rows.append((str(f.resolve()), c))
    return rows

class_names = get_classes(DS1_TRAIN_DIR)
n_classes   = len(class_names)
lab2id      = {c: i for i, c in enumerate(class_names)}
id2lab      = np.array(class_names)
print(f"[Classes] {n_classes}: {class_names}")

# ----------------- MediaPipe Hand Landmarker -----------------
HAND = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
    )
)

def pil_to_mp_image(path: str) -> mp.Image:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

def extract_landmarks(img_path: str):
    """Return (21x3 landmarks array, handedness 'Left'/'Right') or None."""
    res = HAND.detect(pil_to_mp_image(img_path))
    if not res.hand_landmarks:
        return None
    pts = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)
    handed = "Right"
    try:
        if res.handedness:
            h0 = res.handedness[0]
            if hasattr(h0, "categories") and h0.categories:
                handed = h0.categories[0].category_name
            elif isinstance(h0, (list, tuple)) and h0 and hasattr(h0[0], "category_name"):
                handed = h0[0].category_name
    except Exception:
        pass
    return pts, handed

# ----------------- Canonicalization & features -----------------
PALM_IDX = [0, 1, 2, 5, 9]
FEAT_DIM = 43

def _center_scale_xy(xy: np.ndarray) -> np.ndarray:
    center = xy[PALM_IDX].mean(axis=0, keepdims=True)
    xy = xy - center
    dmax = 1e-6
    for i in range(21):
        dif = xy[i+1:] - xy[i]
        if not dif.size:
            break
        d = np.sqrt((dif**2).sum(axis=1)).max(initial=0.0)
        if d > dmax:
            dmax = d
    return xy / dmax

def _rotate_to_up(xy: np.ndarray) -> np.ndarray:
    v = xy[9] - xy[0]
    ang = np.arctan2(v[0], v[1])  # keep original convention
    ca, sa = np.cos(-ang), np.sin(-ang)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    return xy @ R.T

def canonicalize_pts_rot(pts_xyz: np.ndarray, handed: str) -> np.ndarray:
    """Mirror Left->Right, center, scale, rotate; return 43-D feat (xy + handedness bit)."""
    pts = pts_xyz.copy()
    is_left = handed.lower().startswith("left")
    if is_left:
        pts[:, 0] = 1.0 - pts[:, 0]
    xy = _center_scale_xy(pts[:, :2])
    xy = _rotate_to_up(xy)
    xy = np.clip(xy, -2.0, 2.0)
    feat = xy.reshape(-1).astype(np.float32)
    hbit = 0.0 if is_left else 1.0
    return np.concatenate([feat, [hbit]], axis=0)

# ----------------- Cache (v2) -----------------
def hash_path(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:16]

def build_or_load_cache() -> pd.DataFrame:
    if LANDMARKS_CSV.exists():
        df = pd.read_csv(LANDMARKS_CSV)
        print(f"[Cache] Loaded {len(df)} rows from {LANDMARKS_CSV}")
        return df

    rows = []
    for split_name, root in [
        ("ds1_train", DS1_TRAIN_DIR),
        ("ds1_test",  DS1_TEST_DIR),
        ("ds2_pool",  DS2_VAL_DIR),
    ]:
        pairs = list_images(root, class_names)
        print(f"[Scan] {split_name}: {len(pairs)} images")
        for i, (img_path, lab) in enumerate(pairs):
            out = extract_landmarks(img_path)
            if out is None:
                continue
            pts, handed = out
            feat = canonicalize_pts_rot(pts, handed)
            rows.append({
                "key": hash_path(img_path),
                "path": img_path,
                "label": lab,
                "y": lab2id[lab],
                "split_hint": split_name,
                "feat_json": json.dumps(feat.tolist()),
            })
            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] {i+1}/{len(pairs)}...")
    df = pd.DataFrame(rows)
    df.to_csv(LANDMARKS_CSV, index=False)
    print(f"[Cache] Saved {len(df)} rows -> {LANDMARKS_CSV}")
    return df

df_cache = build_or_load_cache()

# ----------------- Splits -----------------
def subset(df, mask):
    return df.loc[mask].reset_index(drop=True)

df_ds1 = subset(df_cache, df_cache["split_hint"].isin(["ds1_train", "ds1_test"]))
df_ds2 = subset(df_cache, df_cache["split_hint"] == "ds2_pool")

def stratified_split_50_25_25(df: pd.DataFrame, seed=SEED):
    parts = []
    for _, grp in df.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=seed)
        n = len(grp)
        n_tr = max(1, int(round(n * 0.50)))
        n_va = max(1, int(round(n * 0.25)))
        n_tr = min(n_tr, n - 2)
        n_va = min(n_va, n - n_tr - 1)
        parts.append((grp.iloc[:n_tr], grp.iloc[n_tr:n_tr+n_va], grp.iloc[n_tr+n_va:]))
    tr = pd.concat([a for a, _, _ in parts], ignore_index=True)
    va = pd.concat([b for _, b, _ in parts], ignore_index=True)
    te = pd.concat([c for *_, c in parts], ignore_index=True)
    return tr, va, te

df1_tr, df1_va, df1_te = stratified_split_50_25_25(df_ds1)

def take_k_per_class(df, k, seed=SEED):
    idxs = []
    for _, grp in df.groupby("label"):
        idxs += grp.sample(n=min(k, len(grp)), random_state=seed).index.tolist()
    return df.loc[idxs]

K_PER_CLASS = 30
support = take_k_per_class(df_ds2, K_PER_CLASS)
df2_rem = df_ds2.drop(index=support.index)

def split_50_50(df, seed=SEED):
    a, b = [], []
    for _, grp in df.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=seed)
        nA = len(grp) // 2
        a.append(grp.iloc[:nA])
        b.append(grp.iloc[nA:])
    return pd.concat(a, ignore_index=True), pd.concat(b, ignore_index=True)

df2_va, df2_te = split_50_50(df2_rem)

SUPP_MULT = 3
support_os = pd.concat([support] * SUPP_MULT, ignore_index=True)

df_train = pd.concat([df1_tr, support_os], ignore_index=True)
df_val   = pd.concat([df1_va, df2_va],   ignore_index=True)
df_test  = pd.concat([df1_te, df2_te],   ignore_index=True)

for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    print(f"[Split] {name}: {len(d)}")

# ----------------- Numpy arrays -----------------
def df_to_np(df: pd.DataFrame):
    X = np.vstack([np.asarray(json.loads(s), dtype=np.float32) for s in df["feat_json"]])
    y = df["y"].values.astype("int32")
    assert X.shape[1] == FEAT_DIM
    return X, y

X_train, y_train = df_to_np(df_train)
X_val,   y_val   = df_to_np(df_val)
X_test,  y_test  = df_to_np(df_test)
print("X shapes:", X_train.shape, X_val.shape, X_test.shape)

# ----------------- Landmark-space augmentation -----------------
ROT_DEG      = 20.0
JITTER_SIGMA = 0.02
SCALE_JIT    = 0.05
KP_DROP_P    = 0.05
AUG_P        = 0.7

def augment43(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for i in range(len(X)):
        feat = X[i].copy()
        xy = feat[:42].reshape(21, 2)

        th = np.deg2rad(np.random.uniform(-ROT_DEG, ROT_DEG))
        ca, sa = np.cos(th), np.sin(th)
        xy = xy @ np.array([[ca, -sa], [sa, ca]], dtype=np.float32).T

        xy *= 1.0 + np.random.uniform(-SCALE_JIT, SCALE_JIT)
        xy += np.random.normal(0, JITTER_SIGMA, size=xy.shape).astype(np.float32)

        if KP_DROP_P:
            mask = np.random.rand(21) < KP_DROP_P
            if mask.any():
                palm = xy[PALM_IDX].mean(axis=0, keepdims=True)
                xy[mask] = palm

        feat[:42] = np.clip(xy, -2.0, 2.0).reshape(-1)
        X[i] = feat
    return X

if AUG_P:
    mask = np.random.rand(len(X_train)) < AUG_P
    X_train[mask] = augment43(X_train[mask])

# ----------------- tf.data -----------------
BATCH = 1024

def make_balanced_train_ds(X, y):
    with tf.device("/CPU:0"):
        class_ds = []
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            ds = tf.data.Dataset.from_tensor_slices((X[idx], y[idx]))
            ds = ds.shuffle(min(len(idx), 4096), seed=SEED,
                            reshuffle_each_iteration=True).repeat()
            class_ds.append(ds)
        ds = tf.data.Dataset.sample_from_datasets(class_ds, seed=SEED)
        opts = tf.data.Options()
        opts.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE).with_options(opts)

def make_eval_ds(X, y):
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        opts = tf.data.Options()
        opts.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE).with_options(opts)

train_ds = make_balanced_train_ds(X_train, y_train)
val_ds   = make_eval_ds(X_val,  y_val)
test_ds  = make_eval_ds(X_test, y_test)

def to_one_hot(x, y):
    return x, tf.one_hot(y, n_classes)

train_ds = train_ds.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = val_ds.map(to_one_hot,   num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = test_ds.map(to_one_hot,  num_parallel_calls=tf.data.AUTOTUNE)

steps_per_epoch = max(1, math.ceil(len(X_train) / BATCH))

# ----------------- Model -----------------
tf.keras.backend.clear_session()

norm = layers.Normalization()
norm.adapt(X_train)

wd, drop, width, lr, lsmooth = 2e-4, 0.35, 384, 3e-4, 0.07

with tf.device("/CPU:0"):
    inp = layers.Input(shape=(FEAT_DIM,), name="feat")
    x = norm(inp)
    for _ in range(2):
        x = layers.Dense(width, activation="relu",
                         kernel_regularizer=regularizers.l2(wd))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(n_classes, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inp, out, name="libras_landmarks_mlp_v2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=lsmooth),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )

# ----------------- Macro-F1 callback + training -----------------
class ValMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, name="val_macro_f1"):
        super().__init__()
        self.val_ds = val_ds
        self.name = name
        self.best = -1.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, y_pred = [], []
        for xb, yb_onehot in self.val_ds:
            pb = self.model.predict(xb, verbose=0)
            y_true.append(np.argmax(yb_onehot.numpy(), axis=1))
            y_pred.append(np.argmax(pb, axis=1))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logs[self.name] = macro
        print(f"\n[{self.name}] {macro:.4f}")
        if macro > self.best:
            self.best = macro

val_f1_cb = ValMacroF1(val_ds)
CKPT = str((MODEL_DIR / f"best_landmarks_v2_{uuid.uuid4().hex}.weights.h5").resolve())

cbs = [
    val_f1_cb,
    tf.keras.callbacks.ModelCheckpoint(
        CKPT,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_macro_f1",
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=4,
        min_delta=1e-4,
        restore_best_weights=True,
        monitor="val_macro_f1",
        mode="max",
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=2,
        factor=0.5,
        min_lr=1e-6,
        monitor="val_macro_f1",
        mode="max",
        verbose=1,
    ),
]

print("\n[Train] up to 30 epochs (CPU)")
model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    callbacks=cbs,
    verbose=1,
)

# ----------------- VIS: 10 random val images w/ landmarks + top-3 -----------------
TFLITE_FP32 = MODEL_DIR / "libras_landmarks_mlp_v2_fp32.tflite"
LABELS_TXT  = MODEL_DIR / "labels.txt"

if LABELS_TXT.exists():
    with open(LABELS_TXT, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
else:
    labels = list(class_names)
assert len(labels) == n_classes, "labels.txt mismatch with training classes."

def load_tflite_interpreter():
    if TFLITE_FP32.exists():
        itp = tf.lite.Interpreter(model_path=str(TFLITE_FP32))
        itp.allocate_tensors()
        in_det  = itp.get_input_details()[0]
        out_det = itp.get_output_details()[0]
        return itp, in_det, out_det
    return None, None, None

_itp, _in_det, _out_det = load_tflite_interpreter()

def predict_probs_43(feat43: np.ndarray) -> np.ndarray:
    x = feat43.astype(np.float32)[None, :]
    if _itp is not None:
        _itp.set_tensor(_in_det["index"], x)
        _itp.invoke()
        return _itp.get_tensor(_out_det["index"])[0]
    return model.predict(x, verbose=0)[0]

def draw_landmarks_on_pil(pil_img: Image.Image, pts01):
    img = pil_img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    r = max(2, int(min(w, h) * 0.006))
    for p in pts01:
        x, y = p.x * w, p.y * h
        d.ellipse((x-r, y-r, x+r, y+r), outline=(0, 255, 255), width=2)
    return img

def plot_example(pil_img, top3_labels, top3_scores, title):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8, 3),
        gridspec_kw={"width_ratios": [1, 2]}
    )
    axL.barh(list(reversed(top3_labels)), list(reversed(top3_scores)))
    axL.set_xlim(0, 1)
    axL.set_xlabel("prob")
    axL.set_title("Top-3")
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)

    axR.imshow(pil_img)
    axR.axis("off")

    fig.suptitle(title, y=1.02, fontsize=9)
    plt.tight_layout()

rows = df_val.sample(n=min(10, len(df_val)), random_state=SEED).to_dict(orient="records")

for r in rows:
    path, true_lab = r["path"], r["label"]
    pil = Image.open(path).convert("RGB")

    out = extract_landmarks(path)
    if out is None:
        plt.figure(figsize=(6, 3))
        plt.imshow(pil); plt.axis("off")
        plt.title(f"(NO HAND) {os.path.basename(path)} | true={true_lab}")
        continue

    pts_xyz, handed = out
    feat43 = canonicalize_pts_rot(pts_xyz, handed)
    probs  = predict_probs_43(feat43)
    idxs = np.argsort(-probs)[:3]
    top_labs = [labels[i] for i in idxs]
    top_vals = probs[idxs]

    res = HAND.detect(pil_to_mp_image(path))
    if res.hand_landmarks:
        pil = draw_landmarks_on_pil(pil, res.hand_landmarks[0])

    title = f"{os.path.basename(path)} | true={true_lab} | pred={top_labs[0]} ({top_vals[0]*100:.1f}%)"
    plot_example(pil, top_labs, top_vals, title)

plt.show()

# ----------------- Pretty eval on TEST (plots only for CM) -----------------
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="cividis")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color="white" if val > thresh else "black",
            )
    plt.tight_layout()
    return fig, ax

def eval_ds_pretty(model, ds, title, class_names, n_classes):
    print(f"\n[Eval] {title}")
    y_true, y_pred = [], []

    for Xb, yb in ds:
        pb = model.predict(Xb, verbose=0)
        y_pred.extend(np.argmax(pb, axis=1))
        yb_np = yb.numpy()
        if yb_np.ndim == 2 and yb_np.shape[1] == n_classes:
            y_true.extend(np.argmax(yb_np, axis=1))
        else:
            y_true.extend(yb_np.astype(np.int32))

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("Macro-F1:", f"{macro:.4f}")

    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    plot_confusion_matrix(cm, class_names, title=f"{title} - Confusion Matrix")
    plt.show()

    fig = plt.figure(figsize=(8, 4))
    fig.text(0.01, 0.01, report_str, fontsize=8, family="monospace")
    fig.suptitle(f"{title} - Classification Report", y=0.98)
    plt.tight_layout()
    plt.show()

eval_ds_pretty(model, make_eval_ds(X_test, y_test), "TEST", class_names, n_classes)
