# GE-MV-MGDP V2 for MMRec Cold-Start

Code for running the GE-MV-MGDP V2 cold-item recommendation experiment on Kaggle.

## Run on Kaggle

Create a Kaggle notebook with the dataset mounted at:

```text
/kaggle/input/datasets/toanktx/mmrec-cold
```

Then run:

```bash
!git clone https://github.com/1101909/ge-mv-mgdp-v2-kaggle.git
%cd ge-mv-mgdp-v2-kaggle
!pip install -r requirements.txt
!python ge_mv_mgdp_v2_kaggle.py
```

Kaggle usually already includes `numpy`, `pandas`, `scipy`, and `torch`, so the install step can be skipped if those packages are available.

## Configuration

Main settings are near the top of [ge_mv_mgdp_v2_kaggle.py](ge_mv_mgdp_v2_kaggle.py):

```python
RUN_DATASETS = ["elec"]
EPOCHS = 100
BATCH_SIZE = 256
GRAPH_MODE = "knn"
KNN_K = 10
```

The script defaults to the Kaggle dataset path above. To use a different mounted path:

```bash
MMREC_DATA_ROOT=/kaggle/input/your-dataset/mmrec-cold python ge_mv_mgdp_v2_kaggle.py
```

Results are written to:

```text
/kaggle/working/ge_mv_mgdp_results.json
```

You can override that path with:

```bash
OUTPUT_PATH=/kaggle/working/my_results.json python ge_mv_mgdp_v2_kaggle.py
```

## Ablation Studies

Run all ablations:

```bash
!python ge_mv_mgdp_v2_ablations.py --datasets elec --epochs 100
```

Run only selected components:

```bash
!python ge_mv_mgdp_v2_ablations.py \
  --datasets elec \
  --only full,no_graph,no_gate,image_only,text_only,no_queue,no_modal_alignment \
  --epochs 100
```

Run component ablations with fixed alpha/beta from the full model:

```bash
!python ge_mv_mgdp_v2_ablations.py \
  --datasets baby,sports,clothing \
  --only full,no_graph,no_gate,image_only,text_only,no_queue,no_momentum,no_modal_alignment,no_pos_regularizer,threshold_graph \
  --fixed-params baby:0.0:0.4,sports:0.0:0.4,clothing:0.0:0.6 \
  --epochs 100
```

You can also read the fixed alpha/beta values directly from a full-run result JSON:

```bash
!python ge_mv_mgdp_v2_ablations.py \
  --datasets baby,sports,clothing \
  --only full,no_graph,no_gate,image_only,text_only,no_queue,no_momentum,no_modal_alignment,no_pos_regularizer,threshold_graph \
  --fixed-params-json /kaggle/working/ge_mv_mgdp_results.json \
  --epochs 100
```

List available ablations:

```bash
!python ge_mv_mgdp_v2_ablations.py --list
```

The ablation runner writes:

```text
/kaggle/working/ge_mv_mgdp_ablations.json
/kaggle/working/ge_mv_mgdp_ablations.csv
```

In the ablation runner, `full` is executed through the same `run_dataset()` path used by `ge_mv_mgdp_v2_kaggle.py`. The other entries apply the requested component removals before training/evaluation.

## Files

- `ge_mv_mgdp_v2_kaggle.py`: Kaggle-ready GE-MV-MGDP V2 experiment.
- `ge_mv_mgdp_v2_ablations.py`: Ablation runner for evaluating model components.
- `run_cold_item_experiment.py`: Legacy JSONL-based cold-item experiment.
- `requirements.txt`: Minimal Python dependencies.
