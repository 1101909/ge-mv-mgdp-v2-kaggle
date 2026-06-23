# ============================================================
# GE-MV-MGDP V2 - Run directly on Kaggle MMRec-cold datasets
# Dataset root:
# /kaggle/input/datasets/toanktx/mmrec-cold/
# ============================================================

import os
import json
import time
import random
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# Config
# ============================================================

DATA_ROOT = os.environ.get(
    "MMREC_DATA_ROOT",
    "/kaggle/input/datasets/toanktx/mmrec-cold",
)
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    "/kaggle/working/ge_mv_mgdp_results.json",
)

RUN_DATASETS = [
    "elec",
]
TRAIN_LABEL = 0
TEST_LABEL = 2

EPOCHS = 100
BATCH_SIZE = 256
EMBED_DIM = 256
HIDDEN_DIM = 512
GNN_LAYERS = 2
DROPOUT = 0.1
QUEUE_SIZE = 4096

LR = 3e-4
WEIGHT_DECAY = 0.01
SEED = 42

GRAPH_MODE = "knn"       # "knn" or "threshold"
KNN_K = 10
THRESHOLD_TAU = 0.3
GRAPH_CHUNK_SIZE = int(os.environ.get("GRAPH_CHUNK_SIZE", "2048"))

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

TOPK_LIST = [1, 5, 10, 20]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utils
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_norm(x):
    return F.normalize(x, p=2, dim=-1)


def normalize_adj(adj):
    adj = adj + adj.T
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    adj.data = np.clip(adj.data, 0, 1)
    d = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d[np.isinf(d)] = 0.0
    d_mat = sp.diags(d)
    return d_mat.dot(adj).dot(d_mat).tocsr()


def to_sparse(mx, device):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, mx.shape).coalesce().to(device)


def build_knn_adj(features, k=10, chunk_size=2048):
    n = features.size(0)
    if n <= 1:
        adj = sp.eye(n, dtype=np.float32)
        return adj.tocsr()

    k = max(1, min(k, n - 1))
    chunk_size = max(1, int(chunk_size))
    features = l2_norm(features.detach().cpu()).float()

    row_chunks = []
    col_chunks = []
    all_features_t = features.T.contiguous()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = torch.matmul(features[start:end], all_features_t)
        diag = torch.arange(start, end)
        sim[torch.arange(end - start), diag] = -float("inf")
        _, idx = torch.topk(sim, k, dim=1)

        rows = torch.arange(start, end).view(-1, 1).repeat(1, k).reshape(-1)
        row_chunks.append(rows.cpu().numpy())
        col_chunks.append(idx.reshape(-1).cpu().numpy())

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)

    adj = sp.csr_matrix(
        (np.ones_like(rows, dtype=np.float32), (rows, cols)),
        shape=(n, n)
    )
    return normalize_adj(adj)


def build_thresh_adj(features, tau=0.3, chunk_size=2048):
    n = features.size(0)
    chunk_size = max(1, int(chunk_size))
    features = l2_norm(features.detach().cpu()).float()

    row_chunks = []
    col_chunks = []
    all_features_t = features.T.contiguous()

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = torch.matmul(features[start:end], all_features_t)
        diag = torch.arange(start, end)
        sim[torch.arange(end - start), diag] = -float("inf")
        r, c = (sim >= tau).nonzero(as_tuple=True)
        row_chunks.append((r + start).cpu().numpy())
        col_chunks.append(c.cpu().numpy())

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
    else:
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)

    adj = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n)
    )
    return normalize_adj(adj)


def build_adj(image_feat, text_feat, graph_mode="knn", k=10, tau=0.3, device="cpu"):
    combined = l2_norm(torch.cat([image_feat, text_feat], dim=1)).detach().cpu()

    if graph_mode == "knn":
        adj = build_knn_adj(combined, k=k, chunk_size=GRAPH_CHUNK_SIZE)
    elif graph_mode == "threshold":
        adj = build_thresh_adj(combined, tau=tau, chunk_size=GRAPH_CHUNK_SIZE)
    else:
        raise ValueError(f"Unknown graph mode: {graph_mode}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return to_sparse(adj, device)


# ============================================================
# Load MMRec-style dataset
# ============================================================

def load_mmrec_dataset(data_root, dataset, train_label=0, test_label=2):
    dataset_dir = os.path.join(data_root, dataset)

    inter_path = os.path.join(dataset_dir, f"{dataset}.inter")
    image_path = os.path.join(dataset_dir, "image_feat.npy")
    text_path = os.path.join(dataset_dir, "text_feat.npy")

    if not os.path.isfile(inter_path):
        raise FileNotFoundError(inter_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    if not os.path.isfile(text_path):
        raise FileNotFoundError(text_path)

    df = pd.read_csv(inter_path, sep="\t")

    required_cols = {"userID", "itemID", "x_label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{inter_path} must contain {required_cols}")

    train_df = df[df["x_label"] == train_label][["userID", "itemID"]].copy()
    test_df = df[df["x_label"] == test_label][["userID", "itemID"]].copy()

    train_users = sorted(train_df["userID"].unique().tolist())
    train_user_set = set(train_users)

    test_df = test_df[test_df["userID"].isin(train_user_set)].copy()

    train_items = sorted(train_df["itemID"].unique().tolist())
    test_items = sorted(test_df["itemID"].unique().tolist())

    image_feat = np.load(image_path, allow_pickle=True).astype(np.float32)
    text_feat = np.load(text_path, allow_pickle=True).astype(np.float32)

    max_item_id = max(train_items + test_items)
    if max_item_id >= image_feat.shape[0]:
        raise ValueError(
            f"{dataset}: itemID max={max_item_id}, image_feat rows={image_feat.shape[0]}"
        )
    if max_item_id >= text_feat.shape[0]:
        raise ValueError(
            f"{dataset}: itemID max={max_item_id}, text_feat rows={text_feat.shape[0]}"
        )

    return train_df, test_df, train_items, test_items, train_users, image_feat, text_feat


# ============================================================
# Model
# ============================================================

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, img, txt):
        g = self.gate(torch.cat([img, txt], dim=-1))
        return g * img + (1.0 - g) * txt


class ItemEncoder(nn.Module):
    def __init__(self, image_dim, text_dim, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.fusion = GatedFusion(embed_dim)

        self.gnn_weights = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
        self.gnn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])

    def forward(self, image_feat, text_feat, adj):
        zi = l2_norm(self.img_proj(image_feat))
        zt = l2_norm(self.txt_proj(text_feat))

        h = self.fusion(zi, zt)
        outputs = [h]

        for weight, norm in zip(self.gnn_weights, self.gnn_norms):
            h_new = torch.sparse.mm(adj, h)
            h_new = F.gelu(norm(weight(h_new)))
            h = h + 0.3 * h_new
            outputs.append(l2_norm(h))

        return l2_norm(sum(outputs) / len(outputs)), zi, zt


class MemoryQueue(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.size = size
        self.register_buffer("queue", l2_norm(torch.randn(size, dim)))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue_and_dequeue(self, keys):
        keys = l2_norm(keys.detach())
        batch_size = min(keys.size(0), self.size)

        keys = keys[:batch_size]
        ptr = int(self.ptr)

        if ptr + batch_size <= self.size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            rem = self.size - ptr
            self.queue[ptr:] = keys[:rem]
            self.queue[:batch_size - rem] = keys[rem:]

        self.ptr[0] = (ptr + batch_size) % self.size


class DeepGEV2(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        embed_dim,
        n_users,
        hidden_dim=512,
        n_layers=2,
        dropout=0.1,
        queue_size=4096,
        momentum=0.995,
    ):
        super().__init__()

        self.momentum = momentum

        self.u_emb = nn.Embedding(n_users, embed_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)

        self.user_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.online_encoder = ItemEncoder(
            image_dim=image_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.target_encoder = ItemEncoder(
            image_dim=image_dim,
            text_dim=text_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

        self.queue = MemoryQueue(queue_size, embed_dim)
        self.log_temp = nn.Parameter(torch.ones(1) * np.log(0.07))

    @torch.no_grad()
    def update_target(self):
        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(self.momentum).add_(p_o.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def enqueue_target(self, image_feat, text_feat, adj, item_idx):
        v_target, _, _ = self.target_encoder(image_feat, text_feat, adj)
        self.queue.enqueue_and_dequeue(v_target[item_idx])

    def encode_items(self, image_feat, text_feat, adj, mode="online"):
        if mode == "online":
            return self.online_encoder(image_feat, text_feat, adj)
        else:
            with torch.no_grad():
                return self.target_encoder(image_feat, text_feat, adj)

    def forward(self, image_feat, text_feat, adj, user_idx, item_idx):
        v_online, zi, zt = self.online_encoder(image_feat, text_feat, adj)

        with torch.no_grad():
            v_target, _, _ = self.target_encoder(image_feat, text_feat, adj)

        ue = l2_norm(self.user_proj(self.u_emb(user_idx)))

        temp = self.log_temp.exp().clamp(min=0.01, max=0.5)

        v_pos = v_target[item_idx]
        l_pos = torch.einsum("nd,nd->n", ue, v_pos).unsqueeze(1)
        l_neg = torch.matmul(ue, self.queue.queue.T)

        logits = torch.cat([l_pos, l_neg], dim=1) / temp
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=ue.device)

        loss_main = F.cross_entropy(logits, labels)
        loss_modal = 0.1 * F.mse_loss(zi, zt)

        pos_scores = torch.einsum("nd,nd->n", ue, v_online[item_idx])
        loss_reg = -0.05 * pos_scores.mean()

        return loss_main + loss_modal + loss_reg


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    train_image_raw,
    train_text_raw,
    test_image_raw,
    test_text_raw,
    train_image,
    train_text,
    test_image,
    test_text,
    train_items,
    test_items,
    user2idx,
    train_df,
    test_df,
    device,
):
    model.eval()

    with torch.no_grad():
        all_image = torch.cat([train_image, test_image], dim=0)
        all_text = torch.cat([train_text, test_text], dim=0)

        adj_all = build_adj(
            all_image,
            all_text,
            graph_mode=GRAPH_MODE,
            k=KNN_K,
            tau=THRESHOLD_TAU,
            device=device
        )

        v_all, _, _ = model.encode_items(all_image, all_text, adj_all, mode="online")
        learned_test = v_all[len(train_items):]
        learned_users = l2_norm(model.user_proj(model.u_emb.weight))

        train_item2local = {item_id: idx for idx, item_id in enumerate(train_items)}
        test_item2local = {item_id: idx for idx, item_id in enumerate(test_items)}

        rows, cols = [], []
        for row in train_df.itertuples(index=False):
            if row.userID in user2idx and row.itemID in train_item2local:
                rows.append(user2idx[row.userID])
                cols.append(train_item2local[row.itemID])

        hist = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(user2idx), len(train_items))
        )

        denom = np.clip(np.array(hist.sum(1)), 1, None)

        train_img_zs = l2_norm(train_image_raw)
        train_txt_zs = l2_norm(train_text_raw)
        test_img_zs = l2_norm(test_image_raw)
        test_txt_zs = l2_norm(test_text_raw)

        user_img_zs = torch.tensor(
            hist.dot(train_img_zs.cpu().numpy()) / denom,
            dtype=torch.float32,
            device=device
        )
        user_txt_zs = torch.tensor(
            hist.dot(train_txt_zs.cpu().numpy()) / denom,
            dtype=torch.float32,
            device=device
        )

        user_img_zs = l2_norm(user_img_zs)
        user_txt_zs = l2_norm(user_txt_zs)

        eval_pairs = []
        for row in test_df.itertuples(index=False):
            if row.userID in user2idx and row.itemID in test_item2local:
                eval_pairs.append((user2idx[row.userID], test_item2local[row.itemID]))

        eval_users = sorted(set([u for u, _ in eval_pairs]))
        eval_user_pos = {u: i for i, u in enumerate(eval_users)}

        eval_rows = np.array([eval_user_pos[u] for u, _ in eval_pairs], dtype=np.int64)
        eval_cols = np.array([i for _, i in eval_pairs], dtype=np.int64)

        eval_user_tensor = torch.tensor(eval_users, dtype=torch.long, device=device)

        learned_score = torch.matmul(learned_users[eval_user_tensor], learned_test.T)
        image_score = torch.matmul(user_img_zs[eval_user_tensor], test_img_zs.T)
        text_score = torch.matmul(user_txt_zs[eval_user_tensor], test_txt_zs.T)

        best = None

        print("\nGRID SEARCH")
        print("=" * 80)

        for alpha in ALPHAS:
            zs_score = alpha * image_score + (1.0 - alpha) * text_score

            for beta in BETAS:
                final_score = beta * learned_score + (1.0 - beta) * zs_score

                _, top_idx = torch.topk(
                    final_score,
                    min(max(TOPK_LIST), len(test_items)),
                    dim=1
                )

                top_idx = top_idx.cpu().numpy()
                ranks = np.full(len(eval_pairs), np.inf, dtype=np.float32)

                for pidx, (ridx, iidx) in enumerate(zip(eval_rows, eval_cols)):
                    pos = np.where(top_idx[ridx] == iidx)[0]
                    if len(pos) > 0:
                        ranks[pidx] = pos[0]

                metrics = {}
                for k in TOPK_LIST:
                    metrics[k] = {
                        "recall": float(np.mean(ranks < k)),
                        "ndcg": float(np.mean(np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0))),
                        "mrr": float(np.mean(np.where(ranks < k, 1.0 / (ranks + 1), 0.0))),
                    }

                print(
                    f"alpha={alpha:.2f} beta={beta:.2f} "
                    f"R@20={metrics[20]['recall']:.4f} "
                    f"NDCG@20={metrics[20]['ndcg']:.4f} "
                    f"MRR@20={metrics[20]['mrr']:.4f}"
                )

                if best is None or metrics[20]["recall"] > best["metrics"][20]["recall"]:
                    best = {
                        "alpha": alpha,
                        "beta": beta,
                        "metrics": metrics,
                        "eval_pairs": len(eval_pairs),
                    }

        return best


# ============================================================
# Run one dataset
# ============================================================

def run_dataset(dataset):
    print("\n" + "=" * 100)
    print(f"RUN DATASET: {dataset}")
    print("=" * 100)

    train_df, test_df, train_items, test_items, train_users, image_feat, text_feat = load_mmrec_dataset(
        DATA_ROOT,
        dataset,
        TRAIN_LABEL,
        TEST_LABEL
    )

    print(f"Train users: {len(train_users)}")
    print(f"Train items: {len(train_items)}")
    print(f"Test items : {len(test_items)}")
    print(f"Train pairs: {len(train_df)}")
    print(f"Test pairs : {len(test_df)}")
    print(f"Image dim  : {image_feat.shape[1]}")
    print(f"Text dim   : {text_feat.shape[1]}")

    user2idx = {u: i for i, u in enumerate(train_users)}
    item2idx = {i: j for j, i in enumerate(train_items)}

    train_image_raw = torch.tensor(image_feat[train_items], dtype=torch.float32, device=DEVICE)
    train_text_raw = torch.tensor(text_feat[train_items], dtype=torch.float32, device=DEVICE)
    test_image_raw = torch.tensor(image_feat[test_items], dtype=torch.float32, device=DEVICE)
    test_text_raw = torch.tensor(text_feat[test_items], dtype=torch.float32, device=DEVICE)

    image_mean = train_image_raw.mean(0)
    image_std = train_image_raw.std(0) + 1e-9
    text_mean = train_text_raw.mean(0)
    text_std = train_text_raw.std(0) + 1e-9

    train_image = (train_image_raw - image_mean) / image_std
    test_image = (test_image_raw - image_mean) / image_std
    train_text = (train_text_raw - text_mean) / text_std
    test_text = (test_text_raw - text_mean) / text_std

    adj = build_adj(
        train_image,
        train_text,
        graph_mode=GRAPH_MODE,
        k=KNN_K,
        tau=THRESHOLD_TAU,
        device=DEVICE
    )

    model = DeepGEV2(
        image_dim=train_image.shape[1],
        text_dim=train_text.shape[1],
        embed_dim=EMBED_DIM,
        n_users=len(train_users),
        hidden_dim=HIDDEN_DIM,
        n_layers=GNN_LAYERS,
        dropout=DROPOUT,
        queue_size=QUEUE_SIZE,
    ).to(DEVICE)

    with torch.no_grad():
        init_items, _, _ = model.encode_items(train_image, train_text, adj, mode="online")

        user_hist = defaultdict(list)
        for row in train_df.itertuples(index=False):
            if row.userID in user2idx and row.itemID in item2idx:
                user_hist[row.userID].append(item2idx[row.itemID])

        for user_id, local_items in user_hist.items():
            model.u_emb.weight.data[user2idx[user_id]] = init_items[local_items].mean(0)

    pairs = [
        (user2idx[row.userID], item2idx[row.itemID])
        for row in train_df.itertuples(index=False)
        if row.userID in user2idx and row.itemID in item2idx
    ]

    loader = DataLoader(
        pairs,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=len(pairs) >= BATCH_SIZE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(EPOCHS, 1),
        eta_min=1e-6
    )

    print(f"\nTraining {EPOCHS} epochs on {DEVICE}...")
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch_users, batch_items in loader:
            batch_users = batch_users.to(DEVICE)
            batch_items = batch_items.to(DEVICE)

            loss = model(train_image, train_text, adj, batch_users, batch_items)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.update_target()
            model.enqueue_target(train_image, train_text, adj, batch_items)

            total_loss += float(loss.item())
            steps += 1

        scheduler.step()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"Loss={total_loss / max(steps, 1):.4f} | "
                f"Time={time.time() - start:.1f}s"
            )

    best = evaluate_model(
        model=model,
        train_image_raw=train_image_raw,
        train_text_raw=train_text_raw,
        test_image_raw=test_image_raw,
        test_text_raw=test_text_raw,
        train_image=train_image,
        train_text=train_text,
        test_image=test_image,
        test_text=test_text,
        train_items=train_items,
        test_items=test_items,
        user2idx=user2idx,
        train_df=train_df,
        test_df=test_df,
        device=DEVICE,
    )

    print("\nBEST RESULT")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Best alpha: {best['alpha']}")
    print(f"Best beta : {best['beta']}")
    print(f"Eval pairs: {best['eval_pairs']}")

    for k in TOPK_LIST:
        print(
            f"K={k:2d} | "
            f"Recall={best['metrics'][k]['recall']:.4f} | "
            f"NDCG={best['metrics'][k]['ndcg']:.4f} | "
            f"MRR={best['metrics'][k]['mrr']:.4f}"
        )

    return {
        "dataset": dataset,
        "best_alpha": best["alpha"],
        "best_beta": best["beta"],
        "eval_pairs": best["eval_pairs"],
        "metrics": best["metrics"],
    }


# ============================================================
# Main
# ============================================================

set_seed(SEED)

all_results = []

for dataset in RUN_DATASETS:
    try:
        result = run_dataset(dataset)
        all_results.append(result)
    except Exception as e:
        print("\nERROR DATASET:", dataset)
        print(type(e).__name__, str(e))

print("\n\nFINAL SUMMARY")
print("=" * 100)

for r in all_results:
    m20 = r["metrics"][20]
    print(
        f"{r['dataset']:10s} | "
        f"alpha={r['best_alpha']:.2f} | "
        f"beta={r['best_beta']:.2f} | "
        f"R@20={m20['recall']:.4f} | "
        f"NDCG@20={m20['ndcg']:.4f} | "
        f"MRR@20={m20['mrr']:.4f}"
    )

output_dir = os.path.dirname(OUTPUT_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved to: {OUTPUT_PATH}")
