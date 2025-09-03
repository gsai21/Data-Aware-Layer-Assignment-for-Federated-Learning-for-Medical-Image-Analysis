# Data-Aware Layer Assignment for Secure and Efficient Communication in Federated Learning for Medical Image Analysis
Official Repository of "Data-Aware Layer Assignment for Secure and Efficient Communication in Federated Learning for Medical Image Analysis" which has been published in AAAI - Fall Symposia Series, 2025.

This repo implements a **train-all, send-one** federated protocol with **data-aware, layer-wise assignment**:
- Server estimates **per-layer influence** from a tiny root batch.
- Clients expose **lightweight data-quality** metadata.
- Layers are assigned to high-quality clients; each client uploads **one layer** per round.
- Server performs **per-layer blended aggregation**; we log accuracy, loss, influence, and comms.

> Datasets: Chest X-ray Pneumonia (Kermany et al.), ISIC Skin Lesions, Brain Tumor MRI (Nickparvar).

## Quickstart

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run (example: Pneumonia CXR)
python -m dafl.run \
  --dataset_path "/kaggle/input/chest-xray-pneumonia/chest_xray" \
  --num_clients 10 --rounds 50 --epochs_per_client 2 \
  --batch_size 16 --lr 1e-4 --alpha 0.5 \
  --max_influence_batches 10 --sample_count_weight 0.3 \
  --out_dir artifacts/pneumonia
