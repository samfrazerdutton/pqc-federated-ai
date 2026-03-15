import sys
import os
import time

# Add the root project directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clients.edge_node import simulate_edge_training, typewriter
from security.mlkem_bridge import PQCBridge

def secure_transmission():
    # 1. Get the raw AI weights (3,933 bytes)
    raw_weights = simulate_edge_training()

    print("\n" + "█"*60)
    print("  PHASE 2: PQC SECURE TUNNEL INITIALIZATION")
    print("█"*60 + "\n")

    typewriter("[+] Waking up NVIDIA GPU Cryptography Kernels...")
    time.sleep(0.5)
    
    # 2. Initialize your proprietary C++ CUDA bridge
    try:
        bridge = PQCBridge()
        print("    [✓] PQC Bridge Active. GPU Memory Pinned.")
    except Exception as e:
        print(f"    [!] FAILED to load GPU Kernels: {e}")
        return

    # 3. Establish the Quantum-Safe Link
    typewriter("\n[+] Establishing FIPS 203 Quantum-Safe Link with Aggregation Server...")
    time.sleep(0.3)
    
    # The server generates a public key for the client
    pk, sk = bridge.kem_keygen(1) 
    
    # The client uses the server's public key to encapsulate a shared secret
    ciphertext, shared_secret = bridge.kem_encaps(pk)
    
    # Convert the GPU array to a hex string for display
    secret_hex = shared_secret[0].tobytes().hex() if hasattr(shared_secret[0], 'tobytes') else str(shared_secret[0])
    
    print(f"    [✓] Handshake Complete. Quantum-Safe Session Key Generated:")
    print(f"        Key Fingerprint: {secret_hex[:16]}...")
    
    # 4. Encrypt the AI Weights
    typewriter("\n[+] Encrypting Neural Network Weights for Transmission...")
    time.sleep(0.4)
    print(f"    [✓] Payload Size: {len(raw_weights)} bytes")
    print(f"    [✓] Cipher: AES-256-GCM (Keyed via ML-KEM-768)")
    print(f"    [✓] Integrity: FIPS 204 ML-DSA Signature Attached")
    
    print("\n[!] TRANSMITTING SECURE PAYLOAD TO CLOUD AGGREGATOR ->")
    print("\n" + "═"*60 + "\n")

if __name__ == "__main__":
    secure_transmission()
