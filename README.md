# Fine-Tuning TinyLlama with QLoRA

Fine-tuning TinyLlama-1.1B-Chat on the Guanaco instruction dataset using 4-bit quantization and LoRA adapters. The whole thing runs on a free T4 GPU — no expensive hardware needed.

> 🤗 Adapter on Hugging Face: [khaoula-ghz/tinyllama-guanaco-finetune](https://huggingface.co/khaoula-ghz/tinyllama-guanaco-finetune)

---

## Files

```
qlora-tinyllama-finetune/
│
├── Fine_Tune_Llama_2_QLoRA.ipynb    # training pipeline, evaluation, results
└── README.md
```

---

## What this project does

QLoRA lets you fine-tune a language model without loading it fully in float32. The model gets quantized to 4-bit, then small LoRA adapter weights are trained on top — you touch roughly 1% of the parameters and still get real improvements.

This notebook fine-tunes TinyLlama on 5,000 instruction-following examples from the Guanaco dataset and evaluates the result with BERTScore and an LLM judge.

---

## Setup

| Parameter | Value |
|---|---|
| Base model | TinyLlama-1.1B-Chat |
| Dataset | guanaco-llama2-5k |
| Precision | bf16 |
| LoRA rank | 16 |
| Sequence length | 512 |
| Packing | False |
| Gradient checkpointing | True |
| Quantization | 4-bit (QLoRA) |

LoRA rank 16 is a reasonable middle ground — low enough to keep the adapter small, high enough that the model actually learns something. I tried lower values and the quality dropped noticeably.

---

## Results

| Metric | Score |
|---|---|
| Perplexity | 4.56 |
| BERTScore F1 | 0.85 |
| LLM Judge | 4.26 / 5 |

A perplexity of 4.56 means the model is fairly confident about its outputs — lower is better. BERTScore F1 at 0.85 shows the responses are semantically close to the references. The LLM judge score (4.26/5) confirms it follows instructions well after fine-tuning.

---

## Libraries

- `transformers` — model loading
- `peft` — LoRA adapter
- `bitsandbytes` — 4-bit quantization
- `trl` (SFTTrainer) — training loop
- `huggingface_hub` — pushing the adapter
- `bert_score` — evaluation

---

## Load the adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "khaoula-ghz/tinyllama-guanaco-finetune")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

---

## Author

**Khaoula** — Data Science & Intelligent Systems 
📌 Specialization: NLP, Machine Learning, Data Analysis  
🔗 [Tableau Public](https://public.tableau.com/app/profile/khaoula.ghimouze/vizzes) | [Hugging Face](https://huggingface.co/khaoula-ghz)
