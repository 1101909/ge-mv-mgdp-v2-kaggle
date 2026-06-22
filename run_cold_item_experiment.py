import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cold-item recommendation experiment on train/test JSONL files."
    )
    parser.add_argument("--train-path", default="train_cold_item.jsonl")
    parser.add_argument("--test-path", default="test_cold_item.jsonl")
    parser.add_argument("--output-path", default="results_cold_item.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tau-graph", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--alphas",
        default="0.0,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated alpha grid for validation tuning.",
    )
    parser.add_argument(
        "--betas",
        default="0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated beta grid for validation tuning.",
    )
    return parser.parse_args()


ARGS = parse_args()
TOPK_LIST = [1, 5, 10, 20]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHAS = [float(x) for x in ARGS.alphas.split(",") if x.strip()]
BETAS = [float(x) for x in ARGS.betas.split(",") if x.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_norm(x):
    return F.normalize(x, p=2, dim=-1)


def mean_feature(value, item_id, key):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr.mean(axis=0)
    raise ValueError(f"Invalid {key} for item={item_id}: shape={arr.shape}")


def load_jsonl_items(path, max_items=None):
    items = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_items is not None and len(items) >= max_items:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = obj.get("item")
            users = obj.get("users") or []
            img_feature = obj.get("img_feature")
            text_feature = obj.get("text_feature")
            if not item_id or not users or img_feature is None or text_feature is None:
                skipped += 1
                continue
            try:
                img = mean_feature(img_feature, item_id, "img_feature")
                txt = mean_feature(text_feature, item_id, "text_feature")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            items[item_id] = {"users": list(users), "img": img, "txt": txt}
    print(f"Loaded {len(items)} items from {path} (skipped={skipped})")
    return items


def build_thresh_adj(features, tau=0.3):
    n = features.size(0)
    sim = torch.matmul(features, features.T)
    mask = (sim >= tau) & (~torch.eye(n, device=sim.device, dtype=torch.bool))
    rows, cols = mask.nonzero(as_tuple=True)
    adj = sp.csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (rows.cpu().numpy(), cols.cpu().numpy()),
        ),
        shape=(n, n),
    )
    adj = adj + adj.T
    adj = adj + sp.eye(n, dtype=np.float32)
    adj.data = np.clip(adj.data, 0, 1)

    degree = np.power(np.asarray(adj.sum(axis=1)).flatten(), -0.5)
    degree[np.isinf(degree)] = 0.0
    d_mat = sp.diags(degree)
    return d_mat.dot(adj).dot(d_mat).tocsr()


def to_sparse(mx, device):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, mx.shape).coalesce().to(device)


def get_v(data, ids):
    img = np.stack([data[item_id]["img"] for item_id in ids]).astype(np.float32)
    txt = np.stack([data[item_id]["txt"] for item_id in ids]).astype(np.float32)
    return (
        torch.tensor(img, dtype=torch.float32, device=DEVICE),
        torch.tensor(txt, dtype=torch.float32, device=DEVICE),
    )


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x1, x2):
        gate = self.gate(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1.0 - gate) * x2


class ItemEncoder(nn.Module):
    def __init__(self, img_dim, txt_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.pt = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.fusion = GatedFusion(emb_dim)
        self.gnn_weights = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(n_layers)])
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(n_layers)])

    def forward(self, img, txt, adj):
        zi = l2_norm(self.pi(img))
        zt = l2_norm(self.pt(txt))
        h = self.fusion(zi, zt)
        outs = [h]
        for weight, norm in zip(self.gnn_weights, self.gnn_norms):
            h_new = torch.sparse.mm(adj, h)
            h_new = F.gelu(norm(weight(h_new)))
            h = h + 0.3 * h_new
            outs.append(l2_norm(h))
        return l2_norm(sum(outs) / len(outs)), zi, zt


class MemoryQueue(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.register_buffer("queue", l2_norm(torch.randn(size, dim)))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.size = size

    @torch.no_grad()
    def enqueue_and_dequeue(self, keys):
        keys = l2_norm(keys.detach())
        batch_size = min(keys.shape[0], self.size)
        keys = keys[:batch_size]
        ptr = int(self.ptr.item())
        if ptr + batch_size <= self.size:
            self.queue[ptr : ptr + batch_size] = keys
        else:
            rem = self.size - ptr
            self.queue[ptr:] = keys[:rem]
            self.queue[: batch_size - rem] = keys[rem:]
        self.ptr[0] = (ptr + batch_size) % self.size


class DeepGEV2(nn.Module):
    def __init__(self, img_dim, txt_dim, emb_dim, n_users, hidden_dim, n_layers, dropout, queue_size):
        super().__init__()
        self.user_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.online_encoder = ItemEncoder(img_dim, txt_dim, emb_dim, hidden_dim, n_layers, dropout)
        self.target_encoder = ItemEncoder(img_dim, txt_dim, emb_dim, hidden_dim, n_layers, dropout)
        for p_online, p_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.copy_(p_online.data)
            p_target.requires_grad = False

        self.queue = MemoryQueue(queue_size, emb_dim)
        self.u_emb = nn.Embedding(n_users, emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        self.log_temp = nn.Parameter(torch.ones(1) * np.log(0.07))
        self.momentum = 0.99

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        for p_online, p_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data, alpha=1.0 - self.momentum)

    def encode_items(self, img, txt, adj, mode="online"):
        if mode == "online":
            return self.online_encoder(img, txt, adj)
        with torch.no_grad():
            return self.target_encoder(img, txt, adj)

    @torch.no_grad()
    def enqueue_target(self, img, txt, adj, item_idx):
        v_target, _, _ = self.target_encoder(img, txt, adj)
        self.queue.enqueue_and_dequeue(v_target[item_idx])

    def forward(self, img, txt, adj, user_idx, item_idx):
        v_online, zi, zt = self.online_encoder(img, txt, adj)
        user_emb = l2_norm(self.user_proj(self.u_emb(user_idx)))
        temp = self.log_temp.exp().clamp(min=0.01, max=0.5)

        with torch.no_grad():
            v_target, _, _ = self.target_encoder(img, txt, adj)

        pos = v_target[item_idx]
        l_pos = torch.einsum("nc,nc->n", user_emb, pos).unsqueeze(-1)
        l_neg = torch.matmul(user_emb, self.queue.queue.T)
        logits = torch.cat([l_pos, l_neg], dim=1) / temp
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=user_emb.device)

        loss_main = F.cross_entropy(logits, labels)
        loss_modal = F.mse_loss(zi, zt) * 0.1
        pos_scores = (user_emb * v_online[item_idx]).sum(dim=-1)
        loss_reg = -pos_scores.mean() * 0.05
        return loss_main + loss_modal + loss_reg


@torch.no_grad()
def eval_split(
    split_name,
    split_raw,
    model,
    train_img,
    train_txt,
    train_img_raw,
    train_txt_raw,
    img_mean,
    img_std,
    txt_mean,
    txt_std,
    train_item_ids,
    u2idx,
    history_matrix,
    build_tau,
    alpha_beta=None,
    do_grid=False,
):
    model.eval()
    split_ids = list(split_raw.keys())
    if not split_ids:
        print(f"[{split_name}] empty split.")
        return None

    split_img_raw, split_txt_raw = get_v(split_raw, split_ids)
    split_img = (split_img_raw - img_mean) / img_std
    split_txt = (split_txt_raw - txt_mean) / txt_std

    all_img = torch.cat([train_img, split_img], dim=0)
    all_txt = torch.cat([train_txt, split_txt], dim=0)
    adj_all = to_sparse(
        build_thresh_adj(l2_norm(torch.cat([all_img, all_txt], dim=1)).detach().cpu(), tau=build_tau),
        DEVICE,
    )
    v_all, _, _ = model.encode_items(all_img, all_txt, adj_all, mode="online")
    learned_split = v_all[len(train_item_ids) :]
    learned_users = l2_norm(model.user_proj(model.u_emb.weight))

    if split_img_raw.shape[1] != split_txt_raw.shape[1]:
        raise ValueError(
            "Zero-shot alpha fusion requires img_feature and text_feature to have the same dimension. "
            f"Got img={split_img_raw.shape[1]}, text={split_txt_raw.shape[1]}."
        )

    split_img_zs = l2_norm(split_img_raw)
    split_txt_zs = l2_norm(split_txt_raw)
    train_img_zs = l2_norm(train_img_raw)
    train_txt_zs = l2_norm(train_txt_raw)

    def run_once(alpha, beta):
        item_zs = l2_norm(alpha * split_img_zs + (1.0 - alpha) * split_txt_zs)
        train_hist_zs = l2_norm(alpha * train_img_zs + (1.0 - alpha) * train_txt_zs)
        user_zs_np = history_matrix.dot(train_hist_zs.detach().cpu().numpy())
        user_counts = np.asarray(history_matrix.sum(axis=1), dtype=np.float32)
        user_counts = np.clip(user_counts, 1.0, None)
        user_zs = l2_norm(torch.tensor(user_zs_np / user_counts, dtype=torch.float32, device=DEVICE))

        hits = {k: 0 for k in TOPK_LIST}
        ndcgs = {k: 0.0 for k in TOPK_LIST}
        mrrs = {k: 0.0 for k in TOPK_LIST}
        count = 0
        max_k = max(TOPK_LIST)

        for item_pos, item_id in enumerate(split_ids):
            for user_id in split_raw[item_id]["users"]:
                uid = u2idx.get(user_id)
                if uid is None:
                    continue
                score_l = torch.matmul(learned_split, learned_users[uid])
                score_zs = torch.matmul(item_zs, user_zs[uid])
                final_score = beta * score_l + (1.0 - beta) * score_zs
                top_idx = torch.topk(final_score, max_k).indices.detach().cpu().numpy()
                count += 1

                match = np.where(top_idx == item_pos)[0]
                rank = match[0] if len(match) else np.inf
                for k in TOPK_LIST:
                    if rank < k:
                        hits[k] += 1
                        ndcgs[k] += 1.0 / np.log2(rank + 2)
                        mrrs[k] += 1.0 / (rank + 1)

        return {
            str(k): {
                "recall": hits[k] / max(count, 1),
                "ndcg": ndcgs[k] / max(count, 1),
                "mrr": mrrs[k] / max(count, 1),
            }
            for k in TOPK_LIST
        }, count

    if do_grid:
        best = None
        print(f"\nGRID SEARCH on [{split_name}]")
        print("=" * 60)
        for alpha in ALPHAS:
            for beta in BETAS:
                metrics, eval_count = run_once(alpha, beta)
                score = (
                    metrics["20"]["recall"],
                    metrics["20"]["ndcg"],
                    metrics["20"]["mrr"],
                )
                print(
                    f"alpha={alpha:>4.2f} beta={beta:>4.2f} "
                    f"R@20={metrics['20']['recall']:.4f} "
                    f"NDCG@20={metrics['20']['ndcg']:.4f} "
                    f"MRR@20={metrics['20']['mrr']:.4f} n={eval_count}"
                )
                if best is None or score > best[0]:
                    best = (score, alpha, beta, metrics, eval_count)
        print_best(split_name, best[1], best[2], best[3], best[4])
        return {
            "alpha": best[1],
            "beta": best[2],
            "metrics": best[3],
            "eval_count": best[4],
        }

    if alpha_beta is None:
        raise ValueError("alpha_beta is required when do_grid=False.")
    alpha, beta = alpha_beta
    metrics, eval_count = run_once(alpha, beta)
    print_best(split_name, alpha, beta, metrics, eval_count)
    return {"alpha": alpha, "beta": beta, "metrics": metrics, "eval_count": eval_count}


def print_best(split_name, alpha, beta, metrics, eval_count):
    print(f"\nRESULTS on [{split_name}] alpha={alpha:.2f} beta={beta:.2f} n={eval_count}")
    print("=" * 60)
    for k in TOPK_LIST:
        row = metrics[str(k)]
        print(
            f"K={k:2d}: Recall={row['recall']:.4f} | "
            f"NDCG={row['ndcg']:.4f} | MRR={row['mrr']:.4f}"
        )


def train_model(train_dict, epochs):
    item_ids = sorted(train_dict.keys())
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    u2items = defaultdict(list)
    for item_id, info in train_dict.items():
        for user_id in info["users"]:
            u2items[user_id].append(item_id)
    user_ids = sorted(u2items.keys())
    u2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}

    train_img_raw, train_txt_raw = get_v(train_dict, item_ids)
    img_mean = train_img_raw.mean(dim=0)
    img_std = train_img_raw.std(dim=0).clamp_min(1e-9)
    txt_mean = train_txt_raw.mean(dim=0)
    txt_std = train_txt_raw.std(dim=0).clamp_min(1e-9)
    train_img = (train_img_raw - img_mean) / img_std
    train_txt = (train_txt_raw - txt_mean) / txt_std

    adj = to_sparse(
        build_thresh_adj(l2_norm(torch.cat([train_img, train_txt], dim=1)).detach().cpu(), tau=ARGS.tau_graph),
        DEVICE,
    )

    model = DeepGEV2(
        img_dim=train_img.shape[1],
        txt_dim=train_txt.shape[1],
        emb_dim=ARGS.embed_dim,
        n_users=len(user_ids),
        hidden_dim=ARGS.hidden_dim,
        n_layers=ARGS.gnn_layers,
        dropout=ARGS.dropout,
        queue_size=ARGS.queue_size,
    ).to(DEVICE)

    with torch.no_grad():
        v_init, _, _ = model.encode_items(train_img, train_txt, adj, mode="online")
        for user_id, user_item_ids in u2items.items():
            idxs = [item_to_idx[item_id] for item_id in user_item_ids if item_id in item_to_idx]
            if idxs:
                model.u_emb.weight.data[u2idx[user_id]] = v_init[idxs].mean(dim=0)

    pairs = [
        (u2idx[user_id], item_to_idx[item_id])
        for item_id, info in train_dict.items()
        for user_id in info["users"]
        if user_id in u2idx and item_id in item_to_idx
    ]
    loader = DataLoader(
        pairs,
        batch_size=ARGS.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=ARGS.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    rows, cols = [], []
    for user_id, user_idx in u2idx.items():
        for item_id in u2items[user_id]:
            item_idx = item_to_idx.get(item_id)
            if item_idx is not None:
                rows.append(user_idx)
                cols.append(item_idx)
    history_matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )

    print(f"Train items={len(item_ids)}, users={len(user_ids)}, pairs={len(pairs)}")
    print(f"Feature dims: img={train_img.shape[1]}, text={train_txt.shape[1]}")
    print(f"Training {epochs} epochs on {DEVICE} ...")
    sys.stdout.flush()
    start = time.time()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_users, batch_items in loader:
            batch_users = batch_users.to(DEVICE)
            batch_items = batch_items.to(DEVICE)
            loss = model(train_img, train_txt, adj, batch_users, batch_items)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            model.enqueue_target(train_img, train_txt, adj, batch_items)
            total_loss += float(loss.item())

        scheduler.step()
        if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == epochs:
            avg_loss = total_loss / max(len(loader), 1)
            print(f"Epoch {ep + 1:3d}/{epochs} | Loss={avg_loss:.4f} | Time={time.time() - start:.1f}s")
            sys.stdout.flush()

    return {
        "model": model,
        "train_img": train_img,
        "train_txt": train_txt,
        "train_img_raw": train_img_raw,
        "train_txt_raw": train_txt_raw,
        "img_mean": img_mean,
        "img_std": img_std,
        "txt_mean": txt_mean,
        "txt_std": txt_std,
        "item_ids": item_ids,
        "u2idx": u2idx,
        "history_matrix": history_matrix,
    }


def run_eval(split_name, split_raw, state, alpha_beta=None, do_grid=False):
    return eval_split(
        split_name=split_name,
        split_raw=split_raw,
        model=state["model"],
        train_img=state["train_img"],
        train_txt=state["train_txt"],
        train_img_raw=state["train_img_raw"],
        train_txt_raw=state["train_txt_raw"],
        img_mean=state["img_mean"],
        img_std=state["img_std"],
        txt_mean=state["txt_mean"],
        txt_std=state["txt_std"],
        train_item_ids=state["item_ids"],
        u2idx=state["u2idx"],
        history_matrix=state["history_matrix"],
        build_tau=ARGS.tau_graph,
        alpha_beta=alpha_beta,
        do_grid=do_grid,
    )


def main():
    set_seed(ARGS.seed)
    train_path = Path(ARGS.train_path)
    test_path = Path(ARGS.test_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path.resolve()}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path.resolve()}")

    print("=" * 70)
    print("GE-MV-MGDP V2 cold-item experiment")
    print(f"DEVICE: {DEVICE}")
    print(f"TRAIN: {train_path.resolve()}")
    print(f"TEST : {test_path.resolve()}")
    print("=" * 70)

    train_full = load_jsonl_items(train_path, max_items=ARGS.max_train_items)
    test_raw = load_jsonl_items(test_path, max_items=ARGS.max_test_items)
    if len(train_full) < 2:
        raise ValueError("Need at least 2 training items.")

    results = {"args": vars(ARGS), "device": str(DEVICE)}

    if ARGS.skip_stage1:
        best_alpha, best_beta = ARGS.alpha, ARGS.beta
        results["validation"] = None
        print(f"Skipping Stage-1. Using alpha={best_alpha:.2f}, beta={best_beta:.2f}")
    else:
        all_items = list(train_full.keys())
        random.shuffle(all_items)
        n_val = max(1, int(len(all_items) * ARGS.val_ratio))
        val_items = set(all_items[:n_val])
        sub_items = set(all_items[n_val:])
        train_sub = {item_id: train_full[item_id] for item_id in sub_items}
        val_raw = {item_id: train_full[item_id] for item_id in val_items}
        print(f"[Stage-1] train_sub={len(train_sub)} | val={len(val_raw)} | test={len(test_raw)}")
        stage1_state = train_model(train_sub, epochs=ARGS.epochs)
        val_result = run_eval("VAL", val_raw, stage1_state, do_grid=True)
        best_alpha, best_beta = val_result["alpha"], val_result["beta"]
        results["validation"] = val_result

    print("\n" + "=" * 70)
    print("[Stage-2] Retrain on FULL TRAIN and evaluate TEST once")
    print("=" * 70)
    full_state = train_model(train_full, epochs=ARGS.epochs)
    test_result = run_eval("TEST", test_raw, full_state, alpha_beta=(best_alpha, best_beta), do_grid=False)
    results["test"] = test_result

    output_path = Path(ARGS.output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
