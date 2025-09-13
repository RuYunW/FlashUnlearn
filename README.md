# FlashUnlearn  
**Training-Free LLM Unlearning via Subspace Distribution Filtering**


---

## 📋 Requirements
- Python 3.11+
- PyTorch 2.0+
- Transformers
- Flash-Attn 2.6.3

---

## 🛠️ Installation
```bash
# 1. Create & activate env
conda create -n unlearning python=3.11
conda activate unlearning

# 2. Install package + eval deps
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3
```

---

## 🔧 Quick Start
```bash
bash scripts/flash_unlearn.sh
```

---

## 🎯 What is FlashUnlearn?

FlashUnlearn is a **training-free** unlearning framework for Large Language Models.  
Instead of costly retraining, it **isolates and suppresses a low-dimensional “forgetting subspace”** in the model’s final hidden representations, achieving **precise knowledge removal** while preserving overall performance.


### Key Highlights
| Feature | Benefit |
|---------|---------|
| **Zero Training** | No gradient updates, no retraining. |
| **Subspace Filtering** | Remaps hidden states to disconnect unwanted knowledge. |
| **Scalable** | Works on any LLMs with minimal overhead. |



## 🏆 Contributions
1. **Training-Free Paradigm** 
2. **Subspace Mechanism** 
3. **Efficiency** 

