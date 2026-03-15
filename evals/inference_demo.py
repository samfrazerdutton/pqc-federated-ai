import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clients.edge_node import EdgeLLMNode

def generate(model, tokenizer, prompt, max_new_tokens=60):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

prompts = {
    "medical": "The patient was diagnosed with",
    "legal":   "The defendant hereby waives",
    "finance": "The yield curve inversion suggests",
}

print("\n" + "█"*60)
print("  INFERENCE DEMO — BEFORE vs AFTER FINE-TUNING")
print("█"*60)

for domain, prompt in prompts.items():
    print(f"\n{'─'*50}\n  {domain.upper()} | Prompt: \"{prompt}\"\n{'─'*50}")
    node = EdgeLLMNode(node_type=domain)
    before = generate(node.model, node.tokenizer, prompt)
    print(f"  BEFORE: {before}")
    node.train_on_local_data(epochs=3)
    after = generate(node.model, node.tokenizer, prompt)
    print(f"  AFTER:  {after}")
