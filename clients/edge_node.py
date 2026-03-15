import torch
import torch.nn as nn
import torch.optim as optim
import io
import time
import sys

class EdgeModel(nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def typewriter(text, delay=0.015):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def simulate_edge_training():
    print("\n" + "█"*60)
    print("  QUANTUM-SAFE FL EDGE NODE v1.0")
    print("█"*60 + "\n")

    typewriter("[+] Initializing Edge Neural Network...")
    model = EdgeModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 2)

    typewriter("[+] Training on Local Sensor Data (1 Epoch)...")
    time.sleep(0.4)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"    [✓] Training complete. Local Loss: {loss.item():.4f}")

    typewriter("\n[+] Extracting Neural Network Weights for PQC Tunnel...")
    time.sleep(0.4)
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    weight_bytes = buffer.getvalue()
    
    print(f"    [✓] Extracted {len(weight_bytes)} bytes of parameter data.")
    print(f"    [!] STATUS: Ready for ML-KEM Encryption & ML-DSA Signing.")
    print("\n" + "═"*60 + "\n")
    
    return weight_bytes

if __name__ == "__main__":
    simulate_edge_training()
