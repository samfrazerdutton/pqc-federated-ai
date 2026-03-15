# Quantum-Safe Federated Learning (QS-FL) Framework

A high-performance, hardware-accelerated Federated Learning architecture secured by NIST-standardized Post-Quantum Cryptography (FIPS 203 / FIPS 204).

## 📌 The Problem
Federated Learning allows massive swarms of edge devices (autonomous vehicles, IoT sensors, mobile devices) to train AI models locally and aggregate the weights on a central server without exposing raw data. However, the transmission of these model weights is highly vulnerable to interception and "model poisoning" attacks. Traditional classical cryptography introduces too much latency for real-time edge aggregation and is vulnerable to future quantum threats.

## 🚀 The Solution
This framework secures the PyTorch Federated Averaging (FedAvg) algorithm using a custom **C++ CUDA Post-Quantum Cryptography Bridge**. By offloading the heavy polynomial math of Lattice-Based Cryptography to the GPU, the system achieves sub-millisecond cryptographic overhead, enabling massively scalable, quantum-resistant AI training.

### Security Stack
* **Confidentiality:** ML-KEM-768 (FIPS 203) used to establish a quantum-resistant tunnel, wrapping the neural network weights in AES-256-GCM.
* **Integrity & Authenticity:** ML-DSA-65 (FIPS 204) digital signatures attached to every payload to guarantee zero tampering or model poisoning.
* **Hardware Acceleration:** Custom Number Theoretic Transform (NTT) CUDA kernels utilizing constant-time Montgomery reductions to prevent side-channel timing attacks.

## 📊 Performance Benchmarks (NVIDIA GPU)
The custom CUDA backend bypasses standard CPU limitations, achieving massive parallelization across the edge swarm:
* **Key Generation:** ~7,262 pairs/sec
* **Encapsulation:** ~88,857 pairs/sec
* **Decapsulation:** ~149,245 pairs/sec
* **Swarm Latency:** 10 nodes verified and decapsulated in **< 1.9ms**.

## 🏗️ Architecture
1. **`clients/edge_node.py`**: A PyTorch simulation of an edge device training a local neural network on sensor data and extracting the raw tensor byte-stream.
2. **`clients/pqc_client.py`**: The secure integration layer. Generates a FIPS 203 session key via the GPU bridge, encrypts the AI weights, and signs the payload.
3. **`server/aggregator.py`**: The cloud server that receives incoming swarm payloads, simultaneously verifies and decapsulates them on the GPU, and performs the FedAvg algorithm to update the global model.

## ⚙️ Quick Start
Ensure you have an NVIDIA GPU, CUDA Toolkit, and PyTorch installed.

**1. Run the Client Node**
`python3 clients/pqc_client.py`

**2. Run the Cloud Aggregator**
`python3 server/aggregator.py`
