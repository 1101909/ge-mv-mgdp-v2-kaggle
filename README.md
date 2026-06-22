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
RUN_DATASETS = ["baby", "sports", "clothing", "elec"]
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

## Files

- `ge_mv_mgdp_v2_kaggle.py`: Kaggle-ready GE-MV-MGDP V2 experiment.
- `run_cold_item_experiment.py`: Legacy JSONL-based cold-item experiment.
- `requirements.txt`: Minimal Python dependencies.
