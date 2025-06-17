import os
import time
import fitz
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import configparser
import json

config = configparser.ConfigParser()
config.read('config.ini')

MODEL_PATH = config['paths']['MODEL_PATH']
PDF_PATHS = json.loads(config['paths']['PDF_PATH'])  # read as list

clean_up_tokenization_spaces=False
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Set device
device = torch.device("cpu")

# Model configs
# read from 
with open('embeddings.json', 'r') as f:
    embedding_model_configs = json.load(f)
# PDF paths
# create script that grabs al the pdfs in config.ini PDF_PATH []
def get_pdfs_from_dir(paths, recursively=True):
    pdfs = []
    for path in paths:
        if recursively:
            for root, _, files in os.walk(path):
                pdfs += [os.path.join(root, f) for f in files if f.lower().endswith(".pdf")]
        else:
            pdfs += [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf")]
    return pdfs

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Failed to read {pdf_path}: {e}")
    return text

def evaluate_embeddings(embeddings):
    cosine_sim = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2).numpy()
    avg_sim = float(np.mean(cosine_sim))
    try:
        k = min(4, len(embeddings) - 1)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embeddings)
        sil_score = float(silhouette_score(embeddings, kmeans.labels_))
    except Exception:
        sil_score = None
    return avg_sim, sil_score

def run_embedding(model_cfg, docs):
    # change working directory:
    model_directory = MODEL_PATH
    name = model_cfg["name"]
    model_type = model_cfg["type"]
    max_tokens = model_cfg["max_tokens"]
    model_path = os.path.join(model_directory, model_cfg["path"])

    # Measure model loading time
    load_start = time.time()

    if model_type == "sentence_transformer":
        model = SentenceTransformer(model_cfg["path"], trust_remote_code=True)
        tokenizer = model.tokenizer
        model.max_seq_length = max_tokens
    elif model_type == "qwen3":
        model = SentenceTransformer(model_cfg["path"])
        tokenizer = model.tokenizer
        model.max_seq_length = max_tokens
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    load_duration = time.time() - load_start

    # Measure embedding time
    embed_start = time.time()

    if name == "Qwen3":
        prompt_name = model_cfg.get("prompt_name", None)
        if prompt_name:
            print(f"Embedding documents for Qwen3 with prompt_name: '{prompt_name}'")
            embeddings = model.encode(docs, prompt_name=prompt_name)
        else:
            print("Embedding documents for Qwen3 without specific prompt_name (using default).")
            embeddings = model.encode(docs)
    else:
        embeddings = model.encode(docs)

    embed_duration = time.time() - embed_start

    return torch.tensor(embeddings), load_duration, embed_duration


def main():
    pdf_files = get_pdfs_from_dir(PDF_PATHS)
    docs = [extract_text_from_pdf(p) for p in pdf_files]
    results_df = pd.DataFrame(columns=["model", "load_time_sec", "embed_time_sec", "avg_cos_sim", "silhouette"])

    for emb_i, config in enumerate(embedding_model_configs):
        try:
            print(f"Running model: {config['name']}")
            embeddings, load_time, embed_time= run_embedding(config, docs)
            avg_sim, sil_score = evaluate_embeddings(embeddings)
            row = {
                    "model": config["name"],
                    "load_time_sec": round(load_time, 2),
                    "embed_time_sec": round(embed_time, 2),
                    "avg_cos_sim": round(avg_sim, 4),
                    "silhouette": round(sil_score, 4) if sil_score else "N/A"
                }             
            print(row)
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Error with model {config['name']}: {e}")
    # Save final results to CSV
    results_df.to_csv("embedding_results.csv", index=False)
    print("\n✅ Results saved to embedding_results.csv")

if __name__ == "__main__":
    main()
