import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ge_mv_mgdp_v2_kaggle as base


DEFAULT_OUTPUT_PATH = os.environ.get(
    "ABLATION_OUTPUT_PATH",
    "/kaggle/working/ge_mv_mgdp_ablations.json",
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    use_image: bool = True
    use_text: bool = True
    use_graph: bool = True
    use_gate: bool = True
    use_memory_queue: bool = True
    use_momentum_target: bool = True
    modal_loss_weight: float = 0.1
    pos_reg_weight: float = -0.05
    eval_alphas: Optional[Tuple[float, ...]] = None
    eval_betas: Optional[Tuple[float, ...]] = None
    graph_mode: Optional[str] = None


ABLATIONS = [
    AblationSpec(
        name="full",
        description="Full GE-MV-MGDP V2.",
    ),
    AblationSpec(
        name="no_graph",
        description="Remove GNN graph propagation by using zero GNN layers and identity adjacency.",
        use_graph=False,
    ),
    AblationSpec(
        name="no_gate",
        description="Replace gated image-text fusion with simple mean fusion.",
        use_gate=False,
    ),
    AblationSpec(
        name="image_only",
        description="Use image modality only; text features are zeroed.",
        use_text=False,
        eval_alphas=(1.0,),
    ),
    AblationSpec(
        name="text_only",
        description="Use text modality only; image features are zeroed.",
        use_image=False,
        eval_alphas=(0.0,),
    ),
    AblationSpec(
        name="no_queue",
        description="Replace the memory queue contrastive loss with in-batch negatives.",
        use_memory_queue=False,
    ),
    AblationSpec(
        name="no_momentum",
        description="Disable EMA momentum target encoder by copying online weights each step.",
        use_momentum_target=False,
    ),
    AblationSpec(
        name="no_modal_alignment",
        description="Remove the image-text alignment loss term.",
        modal_loss_weight=0.0,
    ),
    AblationSpec(
        name="no_pos_regularizer",
        description="Remove the positive-score regularization term.",
        pos_reg_weight=0.0,
    ),
    AblationSpec(
        name="learned_only_eval",
        description="Evaluate with learned user/item scores only; removes zero-shot late fusion.",
        eval_alphas=(0.0,),
        eval_betas=(1.0,),
    ),
    AblationSpec(
        name="zero_shot_only_eval",
        description="Evaluate with image/text zero-shot scores only; removes learned score.",
        eval_betas=(0.0,),
    ),
    AblationSpec(
        name="threshold_graph",
        description="Use threshold graph construction instead of KNN graph construction.",
        graph_mode="threshold",
    ),
]


class MeanFusion(nn.Module):
    def forward(self, img, txt):
        return 0.5 * (img + txt)


class AblationDeepGEV2(base.DeepGEV2):
    def __init__(
        self,
        *args,
        use_memory_queue=True,
        modal_loss_weight=0.1,
        pos_reg_weight=-0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_memory_queue = use_memory_queue
        self.modal_loss_weight = modal_loss_weight
        self.pos_reg_weight = pos_reg_weight

    def forward(self, image_feat, text_feat, adj, user_idx, item_idx):
        v_online, zi, zt = self.online_encoder(image_feat, text_feat, adj)

        with torch.no_grad():
            v_target, _, _ = self.target_encoder(image_feat, text_feat, adj)

        ue = base.l2_norm(self.user_proj(self.u_emb(user_idx)))
        temp = self.log_temp.exp().clamp(min=0.01, max=0.5)
        v_pos = v_target[item_idx]

        if self.use_memory_queue:
            l_pos = torch.einsum("nd,nd->n", ue, v_pos).unsqueeze(1)
            l_neg = torch.matmul(ue, self.queue.queue.T)
            logits = torch.cat([l_pos, l_neg], dim=1) / temp
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=ue.device)
        else:
            logits = torch.matmul(ue, v_pos.T) / temp
            labels = torch.arange(logits.size(0), dtype=torch.long, device=ue.device)

        loss = F.cross_entropy(logits, labels)

        if self.modal_loss_weight:
            loss = loss + self.modal_loss_weight * F.mse_loss(zi, zt)

        if self.pos_reg_weight:
            pos_scores = torch.einsum("nd,nd->n", ue, v_online[item_idx])
            loss = loss + self.pos_reg_weight * pos_scores.mean()

        return loss, v_pos.detach()


def parse_csv_values(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_csv(value):
    return [float(x) for x in parse_csv_values(value)]


def parse_fixed_params(value):
    params = {}
    if not value:
        return params

    for item in parse_csv_values(value):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Invalid --fixed-params entry. Use dataset:alpha:beta, "
                "for example baby:0.0:0.4,sports:0.0:0.4"
            )
        dataset, alpha, beta = parts
        params[dataset] = (float(alpha), float(beta))

    return params


def load_fixed_params_json(path):
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    params = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        if row.get("ablation") not in (None, "full"):
            continue
        dataset = row.get("dataset")
        alpha = row.get("best_alpha")
        beta = row.get("best_beta")
        if dataset is None or alpha is None or beta is None:
            continue
        params[str(dataset)] = (float(alpha), float(beta))

    return params


def build_fixed_params(args):
    fixed_params = {}

    fixed_params.update(load_fixed_params_json(args.fixed_params_json))
    fixed_params.update(parse_fixed_params(args.fixed_params))

    has_alpha = args.fixed_alpha is not None
    has_beta = args.fixed_beta is not None
    if has_alpha != has_beta:
        raise ValueError("--fixed-alpha and --fixed-beta must be provided together.")
    if has_alpha and has_beta:
        fixed_params["*"] = (float(args.fixed_alpha), float(args.fixed_beta))

    return fixed_params


def fixed_params_for_dataset(fixed_params, dataset):
    return fixed_params.get(dataset) or fixed_params.get("*")


def identity_adj(size, device):
    idx = torch.arange(size, dtype=torch.long, device=device)
    values = torch.ones(size, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([idx, idx]),
        values,
        (size, size),
        device=device,
    ).coalesce()


def standardize(train_raw, test_raw):
    mean = train_raw.mean(0)
    std = train_raw.std(0) + 1e-9
    return (train_raw - mean) / std, (test_raw - mean) / std


def apply_modal_ablation(train_image_raw, train_text_raw, test_image_raw, test_text_raw, spec):
    if not spec.use_image:
        train_image_raw = torch.zeros_like(train_image_raw)
        test_image_raw = torch.zeros_like(test_image_raw)
    if not spec.use_text:
        train_text_raw = torch.zeros_like(train_text_raw)
        test_text_raw = torch.zeros_like(test_text_raw)
    return train_image_raw, train_text_raw, test_image_raw, test_text_raw


def build_adj_for_spec(train_image, train_text, spec):
    if not spec.use_graph:
        return identity_adj(train_image.size(0), base.DEVICE)

    return base.build_adj(
        train_image,
        train_text,
        graph_mode=spec.graph_mode or base.GRAPH_MODE,
        k=base.KNN_K,
        tau=base.THRESHOLD_TAU,
        device=base.DEVICE,
    )


@contextmanager
def temporary_base_values(**changes):
    original = {name: getattr(base, name) for name in changes}
    try:
        for name, value in changes.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


@contextmanager
def temporary_identity_graph(spec):
    original_build_adj = base.build_adj
    if spec.use_graph:
        yield
        return

    def _identity_build_adj(image_feat, text_feat, graph_mode="knn", k=10, tau=0.3, device="cpu"):
        target_device = device if isinstance(device, torch.device) else torch.device(device)
        return identity_adj(image_feat.size(0), target_device)

    try:
        base.build_adj = _identity_build_adj
        yield
    finally:
        base.build_adj = original_build_adj


def replace_gate_with_mean(model):
    model.online_encoder.fusion = MeanFusion().to(base.DEVICE)
    model.target_encoder.fusion = MeanFusion().to(base.DEVICE)


def print_dataset_stats(dataset, spec, train_df, test_df, train_items, test_items, train_users, image_feat, text_feat):
    print("\n" + "=" * 100)
    print(f"ABLATION: {spec.name} | DATASET: {dataset}")
    print("=" * 100)
    print(spec.description)
    print(f"Train users: {len(train_users)}")
    print(f"Train items: {len(train_items)}")
    print(f"Test items : {len(test_items)}")
    print(f"Train pairs: {len(train_df)}")
    print(f"Test pairs : {len(test_df)}")
    print(f"Image dim  : {image_feat.shape[1]}")
    print(f"Text dim   : {text_feat.shape[1]}")
    print(f"Graph mode : {spec.graph_mode or base.GRAPH_MODE}")
    print(f"Graph used : {spec.use_graph}")
    print(f"Gate used  : {spec.use_gate}")
    print(f"Queue used : {spec.use_memory_queue}")


def run_one_ablation(dataset, spec, fixed_params=None):
    train_df, test_df, train_items, test_items, train_users, image_feat, text_feat = base.load_mmrec_dataset(
        base.DATA_ROOT,
        dataset,
        base.TRAIN_LABEL,
        base.TEST_LABEL,
    )

    print_dataset_stats(
        dataset,
        spec,
        train_df,
        test_df,
        train_items,
        test_items,
        train_users,
        image_feat,
        text_feat,
    )

    user2idx = {u: i for i, u in enumerate(train_users)}
    item2idx = {i: j for j, i in enumerate(train_items)}

    train_image_raw = torch.tensor(image_feat[train_items], dtype=torch.float32, device=base.DEVICE)
    train_text_raw = torch.tensor(text_feat[train_items], dtype=torch.float32, device=base.DEVICE)
    test_image_raw = torch.tensor(image_feat[test_items], dtype=torch.float32, device=base.DEVICE)
    test_text_raw = torch.tensor(text_feat[test_items], dtype=torch.float32, device=base.DEVICE)

    train_image_raw, train_text_raw, test_image_raw, test_text_raw = apply_modal_ablation(
        train_image_raw,
        train_text_raw,
        test_image_raw,
        test_text_raw,
        spec,
    )

    train_image, test_image = standardize(train_image_raw, test_image_raw)
    train_text, test_text = standardize(train_text_raw, test_text_raw)

    adj = build_adj_for_spec(train_image, train_text, spec)

    model = AblationDeepGEV2(
        image_dim=train_image.shape[1],
        text_dim=train_text.shape[1],
        embed_dim=base.EMBED_DIM,
        n_users=len(train_users),
        hidden_dim=base.HIDDEN_DIM,
        n_layers=base.GNN_LAYERS if spec.use_graph else 0,
        dropout=base.DROPOUT,
        queue_size=max(1, base.QUEUE_SIZE),
        momentum=0.995 if spec.use_momentum_target else 0.0,
        use_memory_queue=spec.use_memory_queue,
        modal_loss_weight=spec.modal_loss_weight,
        pos_reg_weight=spec.pos_reg_weight,
    ).to(base.DEVICE)

    if not spec.use_gate:
        replace_gate_with_mean(model)

    with torch.no_grad():
        init_items, _, _ = model.encode_items(train_image, train_text, adj, mode="online")

        user_hist = {}
        for row in train_df.itertuples(index=False):
            if row.userID in user2idx and row.itemID in item2idx:
                user_hist.setdefault(row.userID, []).append(item2idx[row.itemID])

        for user_id, local_items in user_hist.items():
            model.u_emb.weight.data[user2idx[user_id]] = init_items[local_items].mean(0)

        del init_items
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pairs = [
        (user2idx[row.userID], item2idx[row.itemID])
        for row in train_df.itertuples(index=False)
        if row.userID in user2idx and row.itemID in item2idx
    ]

    loader = DataLoader(
        pairs,
        batch_size=base.BATCH_SIZE,
        shuffle=True,
        drop_last=len(pairs) >= base.BATCH_SIZE,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LR,
        weight_decay=base.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(base.EPOCHS, 1),
        eta_min=1e-6,
    )

    print(f"\nTraining {base.EPOCHS} epochs on {base.DEVICE}...")
    start = time.time()

    for epoch in range(base.EPOCHS):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch_users, batch_items in loader:
            batch_users = batch_users.to(base.DEVICE)
            batch_items = batch_items.to(base.DEVICE)

            loss, queue_keys = model(train_image, train_text, adj, batch_users, batch_items)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.update_target()
            if spec.use_memory_queue:
                model.queue.enqueue_and_dequeue(queue_keys)

            total_loss += float(loss.item())
            steps += 1

        scheduler.step()

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == base.EPOCHS:
            print(
                f"Epoch {epoch + 1:03d}/{base.EPOCHS} | "
                f"Loss={total_loss / max(steps, 1):.4f} | "
                f"Time={time.time() - start:.1f}s"
            )

    fixed_eval = fixed_params_for_dataset(fixed_params or {}, dataset)
    if fixed_eval:
        eval_alpha, eval_beta = fixed_eval
        eval_alphas = [eval_alpha]
        eval_betas = [eval_beta]
        eval_mode = "fixed"
    else:
        eval_alphas = list(spec.eval_alphas) if spec.eval_alphas is not None else list(base.ALPHAS)
        eval_betas = list(spec.eval_betas) if spec.eval_betas is not None else list(base.BETAS)
        eval_mode = "grid"

    print(f"\nEvaluation mode: {eval_mode}")
    print(f"Eval alphas    : {eval_alphas}")
    print(f"Eval betas     : {eval_betas}")

    with temporary_base_values(
        ALPHAS=eval_alphas,
        BETAS=eval_betas,
        GRAPH_MODE=spec.graph_mode or base.GRAPH_MODE,
    ):
        with temporary_identity_graph(spec):
            best = base.evaluate_model(
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
                device=base.DEVICE,
            )

    print("\nBEST ABLATION RESULT")
    print("=" * 80)
    print(f"Ablation: {spec.name}")
    print(f"Dataset : {dataset}")
    print(f"Alpha   : {best['alpha']}")
    print(f"Beta    : {best['beta']}")
    print(f"Eval    : {best['eval_pairs']}")

    for k in base.TOPK_LIST:
        print(
            f"K={k:2d} | "
            f"Recall={best['metrics'][k]['recall']:.4f} | "
            f"NDCG={best['metrics'][k]['ndcg']:.4f} | "
            f"MRR={best['metrics'][k]['mrr']:.4f}"
        )

    return {
        "ablation": spec.name,
        "description": spec.description,
        "dataset": dataset,
        "best_alpha": best["alpha"],
        "best_beta": best["beta"],
        "eval_mode": eval_mode,
        "eval_pairs": best["eval_pairs"],
        "metrics": best["metrics"],
        "settings": {
            "use_image": spec.use_image,
            "use_text": spec.use_text,
            "use_graph": spec.use_graph,
            "use_gate": spec.use_gate,
            "use_memory_queue": spec.use_memory_queue,
            "use_momentum_target": spec.use_momentum_target,
            "modal_loss_weight": spec.modal_loss_weight,
            "pos_reg_weight": spec.pos_reg_weight,
            "graph_mode": spec.graph_mode or base.GRAPH_MODE,
            "eval_alphas": eval_alphas,
            "eval_betas": eval_betas,
            "fixed_alpha": fixed_eval[0] if fixed_eval else None,
            "fixed_beta": fixed_eval[1] if fixed_eval else None,
            "epochs": base.EPOCHS,
            "batch_size": base.BATCH_SIZE,
            "seed": base.SEED,
        },
    }


def metric_at(result, k, name):
    metrics = result.get("metrics", {})
    values = metrics.get(k) or metrics.get(str(k)) or {}
    return values.get(name)


def write_outputs(results, output_path):
    path = Path(output_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = path.with_suffix(".csv")
    fieldnames = [
        "ablation",
        "dataset",
        "best_alpha",
        "best_beta",
        "eval_mode",
        "eval_pairs",
        "recall@20",
        "ndcg@20",
        "mrr@20",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "ablation": result.get("ablation"),
                    "dataset": result.get("dataset"),
                    "best_alpha": result.get("best_alpha"),
                    "best_beta": result.get("best_beta"),
                    "eval_mode": result.get("eval_mode", ""),
                    "eval_pairs": result.get("eval_pairs"),
                    "recall@20": metric_at(result, 20, "recall"),
                    "ndcg@20": metric_at(result, 20, "ndcg"),
                    "mrr@20": metric_at(result, 20, "mrr"),
                    "error": result.get("error", ""),
                }
            )

    print(f"\nSaved JSON: {path}")
    print(f"Saved CSV : {csv_path}")


def select_ablations(only):
    by_name = {spec.name: spec for spec in ABLATIONS}
    if not only:
        return ABLATIONS

    selected = []
    for name in parse_csv_values(only):
        if name not in by_name:
            valid = ", ".join(by_name)
            raise ValueError(f"Unknown ablation '{name}'. Valid names: {valid}")
        selected.append(by_name[name])
    return selected


def apply_args_to_base(args):
    base.DATA_ROOT = args.data_root
    base.EPOCHS = args.epochs
    base.BATCH_SIZE = args.batch_size
    base.LR = args.lr
    base.WEIGHT_DECAY = args.weight_decay
    base.SEED = args.seed
    base.GRAPH_MODE = args.graph_mode
    base.KNN_K = args.knn_k
    base.THRESHOLD_TAU = args.threshold_tau
    base.GRAPH_CHUNK_SIZE = args.graph_chunk_size
    base.EVAL_CHUNK_SIZE = args.eval_chunk_size
    base.ALPHAS = parse_float_csv(args.alphas)
    base.BETAS = parse_float_csv(args.betas)


def print_summary(results):
    print("\n\nABLATION SUMMARY")
    print("=" * 100)
    for result in results:
        if result.get("error"):
            print(f"{result['dataset']:10s} | {result['ablation']:22s} | ERROR: {result['error']}")
            continue

        print(
            f"{result['dataset']:10s} | "
            f"{result['ablation']:22s} | "
            f"alpha={result['best_alpha']:.2f} | "
            f"beta={result['best_beta']:.2f} | "
            f"R@20={metric_at(result, 20, 'recall'):.4f} | "
            f"NDCG@20={metric_at(result, 20, 'ndcg'):.4f} | "
            f"MRR@20={metric_at(result, 20, 'mrr'):.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GE-MV-MGDP V2 ablation studies on MMRec cold-start datasets."
    )
    parser.add_argument("--data-root", default=base.DATA_ROOT)
    parser.add_argument("--datasets", default=",".join(base.RUN_DATASETS))
    parser.add_argument(
        "--only",
        default=os.environ.get("ABLATIONS", ""),
        help="Comma-separated ablation names. Empty means run all.",
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("ABLATION_EPOCHS", base.EPOCHS)))
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=base.LR)
    parser.add_argument("--weight-decay", type=float, default=base.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--graph-mode", choices=["knn", "threshold"], default=base.GRAPH_MODE)
    parser.add_argument("--knn-k", type=int, default=base.KNN_K)
    parser.add_argument("--threshold-tau", type=float, default=base.THRESHOLD_TAU)
    parser.add_argument("--graph-chunk-size", type=int, default=base.GRAPH_CHUNK_SIZE)
    parser.add_argument("--eval-chunk-size", type=int, default=base.EVAL_CHUNK_SIZE)
    parser.add_argument("--alphas", default=",".join(str(x) for x in base.ALPHAS))
    parser.add_argument("--betas", default=",".join(str(x) for x in base.BETAS))
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="Use one fixed alpha for every dataset and ablation. Requires --fixed-beta.",
    )
    parser.add_argument(
        "--fixed-beta",
        type=float,
        default=None,
        help="Use one fixed beta for every dataset and ablation. Requires --fixed-alpha.",
    )
    parser.add_argument(
        "--fixed-params",
        default="",
        help=(
            "Dataset-specific fixed params as dataset:alpha:beta entries, "
            "for example baby:0.0:0.4,sports:0.0:0.4,clothing:0.0:0.6."
        ),
    )
    parser.add_argument(
        "--fixed-params-json",
        default="",
        help=(
            "Read dataset alpha/beta from a full-run result JSON containing "
            "dataset, best_alpha, and best_beta fields."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List available ablations and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        for spec in ABLATIONS:
            print(f"{spec.name}: {spec.description}")
        return

    apply_args_to_base(args)
    base.set_seed(base.SEED)

    datasets = parse_csv_values(args.datasets)
    specs = select_ablations(args.only)
    fixed_params = build_fixed_params(args)
    results = []

    print("Selected datasets :", ", ".join(datasets))
    print("Selected ablations:", ", ".join(spec.name for spec in specs))
    if fixed_params:
        visible_params = {
            dataset: {"alpha": alpha, "beta": beta}
            for dataset, (alpha, beta) in fixed_params.items()
        }
        print("Fixed eval params :", visible_params)

    for dataset in datasets:
        for spec in specs:
            try:
                result = run_one_ablation(dataset, spec, fixed_params=fixed_params)
            except Exception as exc:
                print("\nERROR")
                print(f"Dataset : {dataset}")
                print(f"Ablation: {spec.name}")
                print(type(exc).__name__, str(exc))
                result = {
                    "ablation": spec.name,
                    "description": spec.description,
                    "dataset": dataset,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            results.append(result)
            write_outputs(results, args.output_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_summary(results)


if __name__ == "__main__":
    main()
