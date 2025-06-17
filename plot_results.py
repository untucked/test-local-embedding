# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
def plot_results(filename='embedding_results'):
    csv_file="%s.csv"%(filename)
    output_file="%s.png"%(filename)
    df = pd.read_csv(csv_file)
    df = df[df["silhouette"] != "N/A"]
    df["silhouette"] = df["silhouette"].astype(float)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.grid(False) 
    
    ax1.set_title("Embedding Model Benchmark")
    ax1.set_ylabel("Average Cosine Similarity", color="tab:blue")
    ax1.bar(df["model"], df["avg_cos_sim"], color="tab:blue", alpha=0.6, label="Avg Cos Sim")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.set_ylabel("Silhouette Score", color="tab:red")
    ax2.plot(df["model"], df["silhouette"], color="tab:red", marker="o", label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.xticks(rotation=30)
    fig.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"\n✅ Plot saved as {output_file}")

if __name__ == "__main__":
    filename = "embedding_results"
    filename = "embedding_results_metadata"
    plot_results(filename=filename)
