import os
import time
import json
import torch
import configparser
import pandas as pd
import fitz  # PyMuPDF
import datetime
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document as LlamaDocument
from llama_index.core import VectorStoreIndex
# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")
MODEL_PATH = config["paths"]["MODEL_PATH"]
PDF_PATH = json.loads(config["paths"]["PDF_PATH"])

# Load embedding model configurations
with open("embeddings.json", "r") as f:
    embedding_model_configs = json.load(f)

def evaluate_embeddings(embeddings):
    cosine_sim = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
    ).numpy()
    avg_sim = float(cosine_sim.mean())
    try:
        k = min(4, len(embeddings) - 1)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embeddings)
        sil_score = float(silhouette_score(embeddings, kmeans.labels_))
    except Exception:
        sil_score = None
    return avg_sim, sil_score

def get_documents(pdf_dir):
    documents = []
    for file_name in os.listdir(pdf_dir):
        if not file_name.endswith(".pdf"):
            continue
        full_path = os.path.join(pdf_dir, file_name)
        with open(full_path, "rb") as f:
            content = f.read()
        pdf = fitz.open(stream=content, filetype="pdf")
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            metadata = {
                "file_name": file_name,
                "source": "legal_docs",
                "upload_time": datetime.datetime.now().isoformat(),
                "file_size": len(content),
                "file_type": "pdf",
                "text_length": len(text),
                "page_number": page_num,
                "popper": "legal_docs"
            }

            doc_pg = LlamaDocument(text=f"Absolute Page Number {page_num}\n\n{text}", metadata=metadata)
            doc_pg.excluded_llm_metadata_keys = ["file_name"]
            documents.append(doc_pg)
    return documents

def get_nodes(documents, chunk_size=512):    
    parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=int(0.1 * chunk_size))
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

def run_embedding(model_cfg, embed_model, documents, use_index=True):
    max_tokens = model_cfg.get("max_tokens")
    if use_index:        
        index = VectorStoreIndex.from_documents(documents)
        nodes = list(index.storage_context.docstore.docs.values())     
    else:
        parser = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=int(0.1 * max_tokens))
        nodes = parser.get_nodes_from_documents(documents)
    texts = [node.get_content(metadata_mode="EMBED") for node in nodes]
    embed_start = time.time()
    embeddings = embed_model.get_text_embedding_batch(texts)
    embed_duration = time.time() - embed_start
    embeddings = np.array(embeddings)
    return torch.tensor(embeddings), embed_duration

def get_embedding_model(model_cfg, local_model=True):
    if local_model:
        model_path = os.path.join(MODEL_PATH, model_cfg["path"])
    else:
        model_path = model_cfg["path"]
    load_start = time.time()
    embed_model = HuggingFaceEmbedding(model_name=model_path, trust_remote_code=True)
    load_duration = time.time() - load_start
    # setup Settings
    max_tokens = model_cfg.get("max_tokens")
    Settings.chunk_size = max_tokens
    Settings.chunk_overlap = int(0.1 * max_tokens)
    Settings.embed_model = embed_model
    return embed_model, load_duration


def main():
    # get documents page-wise
    documents=[]
    for files_path in PDF_PATH:
        documents1 = get_documents(files_path)
        documents.extend(documents1)
    results_df = pd.DataFrame({
        "model": pd.Series(dtype="str"),
        "load_time_sec": pd.Series(dtype="float"),
        "embed_time_sec": pd.Series(dtype="float"),
        "avg_cos_sim": pd.Series(dtype="float"),
        "silhouette": pd.Series(dtype="object"),  # Allow float or "N/A"
    })
    all_rows_data = []
    for config in embedding_model_configs:
        if config['name'].lower() == 'qwen3':
            continue
        try:
            print(f"Running model: {config['name']}")
            embed_model, load_time = get_embedding_model(config)
            embeddings, embed_time = run_embedding(config, embed_model, documents)
            avg_sim, sil_score = evaluate_embeddings(embeddings)
            row = {
                "model": config["name"],
                "load_time_sec": round(load_time, 2),
                "embed_time_sec": round(embed_time, 2),
                "avg_cos_sim": round(avg_sim, 4),
                "silhouette": round(sil_score, 4) if sil_score else "N/A",
            }
            all_rows_data.append(row)
        except Exception as e:
            print(f"Error with model {config['name']}: {e}")
        # break
    results_df = pd.DataFrame(all_rows_data)
    results_df.to_csv("embedding_results_metadata.csv", index=False)
    print("\n✅ Results saved to embedding_results_metadata.csv")

if __name__ == "__main__":
    main()
