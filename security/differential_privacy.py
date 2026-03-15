import torch
import numpy as np

class DPLoRAWrapper:
    def __init__(self, epsilon=8.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=None):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier or self._calibrate_noise(epsilon, delta)
        print(f"[DP] Privacy budget: ε={epsilon}, δ={delta:.0e}")
        print(f"[DP] Noise multiplier σ={self.noise_multiplier:.4f}, clipping norm C={max_grad_norm}")

    def _calibrate_noise(self, epsilon, delta):
        return np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def clip_and_noise(self, lora_weights):
        total_norm = sum(v.float().norm(2).item()**2 for v in lora_weights.values()) ** 0.5
        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-6))
        noised = {}
        for k, v in lora_weights.items():
            clipped = v.float() * clip_factor
            noise = torch.randn_like(clipped) * self.noise_multiplier * self.max_grad_norm
            noised[k] = (clipped + noise).to(v.dtype)
        print(f"[DP] Clipping factor={clip_factor:.4f}, noise σ={self.noise_multiplier:.4f}")
        return noised

    def dp_fedavg(self, lora_weight_list):
        print(f"\n[DP] Running DP-FedAvg over {len(lora_weight_list)} clients...")
        noised_list = [self.clip_and_noise(w) for w in lora_weight_list]
        averaged = {}
        for key in noised_list[0].keys():
            stacked = torch.stack([w[key].float() for w in noised_list])
            averaged[key] = stacked.mean(dim=0)
        print(f"[DP] Done. Privacy guarantee: (ε={self.epsilon}, δ={self.delta:.0e})-DP")
        return averaged
