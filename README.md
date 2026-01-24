# 📰 Multi-Model News Article Summarization using Transformers

## 📌 Project Overview
This project implements an end-to-end system for **abstractive news article summarization** using multiple state-of-the-art **transformer-based models**.

The objective is to automatically generate concise and meaningful summaries from long news articles and to **compare the performance of different architectures**, especially for long-document handling.

The system:

- Cleans and preprocesses raw news articles
- Performs batch-wise tokenization for memory-efficient training
- Fine-tunes multiple transformer models
- Evaluates summaries using standard **ROUGE metrics**
- Supports long-document summarization using specialized models
- Is optimized to run on **Google Colab + Google Drive**

---

## 📂 Dataset Information

- **Dataset Type**: News Article Summarization Dataset  
- **Format**: Excel / CSV  
- **Columns Used**:
  - `article_text` – Full news article  
  - `human_summary` – Reference (gold) summary  

Due to GitHub file size limits and data licensing, the dataset is **not included** in this repository.

### 📥 How to Use Your Own Dataset

1. Prepare a dataset containing:
   - `article_text`
   - `human_summary`
2. Save it as `.xlsx` or `.csv`
3. Upload it to Google Drive
4. Update the dataset path in the notebook: DATA_PATH = "/content/drive/MyDrive/your_dataset.xlsx"
## 🧠 Models Implemented

This project includes training and evaluation notebooks for the following models:

### 1️⃣ FLAN-T5 (Small)
Instruction-tuned encoder-decoder model for efficient summarization.

### 2️⃣ Long-T5
Designed for handling longer input sequences than standard T5.

### 3️⃣ LED (Longformer Encoder-Decoder)
Optimized for **very long documents** using sparse attention and global tokens.

### 4️⃣ PRIMERA
Model specialized for multi-document and long-document summarization.

### 5️⃣ GPT-2
Decoder-only baseline for abstractive summarization.

### 6️⃣ Mistral
High-performance modern decoder-only transformer model.

### 7️⃣ LLaMA
Large-scale foundational language model fine-tuned for summarization.

### 8️⃣ Gemma
Lightweight open LLM optimized for efficient fine-tuning.

---
---

## 🧪 Novel Model Proposal: Article-Conditioned Decoder Training (Mistral-ASG)

In addition to the baseline models above, this project introduces a **novel training strategy for decoder-only models**, tailored for the NewsSumm dataset and Indian news clusters.

### 🔑 Core Idea

Instead of training the decoder only on reference summaries, we train it to generate summaries **conditioned directly on the full article text**:

> **Input:** `article_text`  
> **Target:** `human_summary`

This forces the decoder to learn strong source grounding before abstraction.

### 🧠 Why this helps

- Improves factual consistency and entity coverage  
- Reduces hallucination in long multi-document inputs  
- Better captures Indian news structure (locations, politics, events)  
- Works with existing LLM backbones (no architecture change required)

### 🏗 Implementation

- Backbone: **Mistral-7B** (decoder-only)  
- Fine-tuning: **QLoRA (4-bit)**  
- Training format: Article → Summary  

### 📈 Results (ROUGE-F1)

| Training Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------------|---------|----------|----------|
| Summary-only | 0.1137 | 0.0271 | 0.0749 |
| **Article → Summary (Proposed)** | **0.5254** | **0.3050** | **0.3951** |

This represents more than **4× improvement on ROUGE-1** and **10× on ROUGE-2**, demonstrating the importance of article-conditioned supervision.

**Proposed model name:** `Mistral-ASG` (Article-Supervised Generator)

---


## 📊 Evaluation Metrics

Each trained model is evaluated using:

- ROUGE-1 (F1 & Precision)
- ROUGE-2 (F1 & Precision)
- ROUGE-L (F1 & Precision)

Evaluation is performed **batch-wise on the test set** using generated summaries compared against reference summaries.

---
## 📁 Evaluation Results

The aggregated ROUGE evaluation results (F1 and Precision scores for all models) are stored in:


This file contains:

- ROUGE-1 F1 & Precision  
- ROUGE-2 F1 & Precision  
- ROUGE-L F1 & Precision  

These values are used to generate comparative performance plots across models using Python (pandas + seaborn / matplotlib).

The CSV file is included in this repository for reproducibility and result verification.

## 📈 Processing Pipeline

- Dataset loading from Google Drive  
- Text cleaning (HTML removal, URLs, emails, extra spaces)  
- Train / Test split  
- Batch-wise tokenization (`.pt` files)  
- Model fine-tuning  
- Summary generation  
- ROUGE evaluation
- Aggregation of ROUGE scores into CSV for visualization


### Execution:

  step1_clone_repository: "Clone the GitHub repository to your local machine"
    
    commands:
      - git clone https://github.com/Shriganga-Gokavi/MultiModel_Summarization-.git
      - cd MultiModel_Summarization-

  step2_open_in_colab: "paste required model url and run in collab"
  
    notebooks:
    
      - model: FLAN-T5
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/flan-t5.ipynb

      - model: LED
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/led.ipynb

      - model: Long-T5
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/longt5.ipynb

      - model: Gemma
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/gemma.ipynb

      - model: LLaMA
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/llama.ipynb

      - model: Mistral
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/mistral.ipynb

      - model: PRIMERA
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/PRIMERA.ipynb

      - model: GPT-2
        url: https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/gpt2.ipynb
        
      - compare:using graphs
        url:https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/comparision.ipynb

      - model: Novel Model (Mistral-ASG)
        url:https://colab.research.google.com/github/Shriganga-Gokavi/MultiModel_Summarization-/blob/main/novel_model.ipynb
  
   step3_install_dependencies: "Install required Python libraries in Google Colab"

     
   
          !pip install transformers rouge-score sentencepiece beautifulsoup4 pandas scikit-learn torch


step4_mount_google_drive: "Mount Google Drive to access the dataset"


  
    from google.colab import drive
    drive.mount('/content/drive')


