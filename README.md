# 🧠 MFactSum: Abstractive Dialogue Summarization via Multi-Level Factual
Consistency
> **MOEsum** is an innovative dialogue summarization framework that integrates **MoE (Mixture-of-Experts)** architecture, an **Event Factuality Module (Event-cross-attn)**, and **Contrastive Learning**.
> The system leverages **DocIE@UIEPrompter** for event extraction and employs diverse negative samples in contrastive learning, significantly improving the **factuality**, **fluency**, and **robustness** of generated summaries.

---

## 🌟 Project Overview

The summarization model implemented in this repository integrates three key techniques:

1. **MoE (Mixture-of-Experts) Sparse-Activated Expert Network**  
   An MoE structure is introduced into the final layer of the BART Encoder, using a standard FFN expert design. A Top-K sparse routing mechanism dynamically selects experts, while a Load Balancing Loss ensures balanced expert utilization, effectively enhancing representational capacity and training stability.

2. **Event Factuality Module**  
   Structured events are automatically extracted from dialogues using **DocIE@UIEPrompter**, and event information is fused during decoding through a dedicated cross-attention layer, ensuring that generated content aligns with real events.

3. **Decoder-Space Contrastive Learning**  
   Margin-based Ranking Loss is adopted for contrastive learning. Using 1–6 negative summary samples, semantic similarity is computed in the decoder output space to avoid semantic space misalignment. A Projection Head maps representations to a low-dimensional space, ensuring that positive samples are more similar than negative samples by at least a margin (default: 0.2).

---

## ⚙️ Environment Requirements

```bash
transformers==4.30.2
torch>=2.0
nltk
datasets
evaluate
filelock
sentence-transformers   # optional, used for contrastive embeddings
```

---

## 📂 Project Structure

```
MOEsum/
│
├── finetuning/
│   ├── run_summarization.py
│   ├── modeling_bart.py
│   ├── trainer.py
│   ├── moe_layer.py
│   ├── callbacks.py
│   └── data/
│       ├── samsum/
│       └── dialogsum/
│
├── facebook/
│   ├── bart-base/
│   └── bart-large/
│
└── README.md
```

---

## 🧠 Usage

### 🔹 Step 1: Data Preparation

Each training sample should contain:

```json
{
  "text": "dialogue text ...",
  "summary": "reference summary ...",
  "events": [
    {"subject": "Alice", "action": "called", "object": "Bob"},
    {"subject": "Bob", "action": "agreed", "object": "meeting"}
  ],
  "neg1": "negative summary 1",
  "neg2": "negative summary 2",
  "neg3": "negative summary 3"
}
```

### 🔹 Step 2: Run Training

```bash

python run_summarization.py \
  --model_name_or_path ../facebook/bart-large \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file data/samsum/train.json \
  --validation_file data/samsum/valid.json \
  --test_file data/samsum/test.json \
  --output_dir outputs/samsum_full \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --predict_with_generate \
  --max_source_length 1024 \
  --max_target_length 128 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --save_total_limit 2 \
  --fp16 \
  --load_best_model_at_end True \
  --text_column text \
  --summary_column summary \
  --num_beams 4 \
```
