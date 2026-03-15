import torch
import io
import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim import AdamW
from datasets import Dataset

def typewriter(text, delay=0.015):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# Simulated private datasets per node type
NODE_DATASETS = {
    "medical": [
        {"text": "Patient presents with acute myocardial infarction requiring immediate intervention."},
        {"text": "Diagnosis: Type 2 diabetes mellitus with peripheral neuropathy complications."},
        {"text": "Post-operative care following laparoscopic cholecystectomy shows normal recovery."},
        {"text": "MRI reveals herniated disc at L4-L5 causing radiculopathy symptoms."},
        {"text": "Prescription: Metformin 500mg twice daily for glycemic control management."},
    ],
    "legal": [
        {"text": "The defendant hereby waives all rights to appeal under Section 14(b) of the Act."},
        {"text": "Pursuant to the contractual obligation, indemnification clause applies retroactively."},
        {"text": "The court finds in favor of plaintiff on grounds of tortious interference with contract."},
        {"text": "Habeas corpus petition filed challenging the legality of continued detention."},
        {"text": "Arbitration clause mandates binding resolution outside of federal court jurisdiction."},
    ],
    "finance": [
        {"text": "Q3 earnings report indicates 12% revenue growth driven by recurring subscription model."},
        {"text": "Portfolio rebalancing recommended given current yield curve inversion signals."},
        {"text": "Credit default swap spreads widening suggests elevated counterparty risk exposure."},
        {"text": "Federal Reserve hawkish stance implies further rate hikes in upcoming FOMC meetings."},
        {"text": "Derivative hedging strategy reduces exposure to foreign exchange volatility risk."},
    ],
}

class EdgeLLMNode:
    def __init__(self, node_type="medical", model_name="microsoft/phi-2"):
        self.node_type = node_type
        self.model_name = model_name

        typewriter(f"[+] Loading base model ({model_name}) in 4-bit mode...")

        # 4-bit quantization config — keeps us well within 6GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA config — only trains ~0.1% of parameters
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )

        self.model = get_peft_model(base_model, lora_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        print(f"    [✓] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    def train_on_local_data(self, epochs=2):
        """Fine-tune on private local data — never leaves this node."""
        typewriter(f"\n[+] Training on private {self.node_type} data ({epochs} epochs)...")

        raw_data = NODE_DATASETS[self.node_type]
        texts = [d["text"] for d in raw_data]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)
        labels = input_ids.clone()

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=2e-4)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

        print(f"    [✓] Local training complete on {self.node_type} data.")
        return self.extract_lora_weights()

    def extract_lora_weights(self):
        """Extract only LoRA delta weights — ~4MB vs 6GB for full model."""
        lora_state = {
            k: v.detach().cpu() for k, v in self.model.state_dict().items()
            if "lora_" in k
        }
        size_bytes = sum(v.nelement() * v.element_size() for v in lora_state.values())
        print(f"    [✓] Extracted LoRA adapter: {size_bytes/1024:.1f} KB "
              f"({len(lora_state)} tensors)")
        print(f"    [!] STATUS: Ready for ML-KEM Encryption & ML-DSA Signing.")
        return lora_state

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  QUANTUM-SAFE FL EDGE NODE v2.0 — LLM EDITION")
    print("█"*60 + "\n")
    node = EdgeLLMNode(node_type="medical")
    lora_weights = node.train_on_local_data(epochs=2)
    print(f"\n[✓] Demo complete. LoRA keys: {list(lora_weights.keys())[:3]}...")
