# cli.py

import pandas as pd

def show_top_models(csv_file="embedding_results.csv", metric="avg_cos_sim", top_n=3):
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by=metric, ascending=False)
    print(f"\n📊 Top {top_n} Models by '{metric}':\n")
    print(df_sorted[["model", "load_time_sec", "embed_time_sec", "avg_cos_sim", "silhouette"]].head(top_n))

if __name__ == "__main__":
    # Hardcoded for debugging — you can change these values easily in the debugger
    csv_file = "embedding_results.csv"
    metric = "silhouette"
    top_n = 5

    show_top_models(csv_file, metric, top_n)
