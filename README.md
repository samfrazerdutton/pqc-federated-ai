# Quantum-Safe Federated LLM Fine-Tuning

3 private edge nodes collaboratively fine-tune a shared LLM — raw data never leaves each node, weights are encrypted against quantum attacks.

## Key Numbers
| Metric | Value |
|---|---|
| Communication reduction | 99.4% (10MB vs 6GB) |
| Trainable parameters | 2,621,440 / 2,782,305,280 (0.094%) |
| Crypto overhead | < 2ms for 3 nodes |
| Privacy guarantee | (ε=8, δ=1e-5)-DP |

## How it works
- Each node (medical / legal / finance) trains only on its own private data
- Only the LoRA adapter (10MB) is transmitted, not the full 6GB model
- ML-KEM-768 (FIPS 203) encrypts every adapter against quantum attacks
- DP-FedAvg adds calibrated Gaussian noise before aggregation

## Stack
PyTorch · HuggingFace · PEFT · Opacus · CuPy · CUDA · Gradio

## Run it
```bash
pip install -r requirements.txt
python3 server/aggregator.py
python3 demo/app.py
```
