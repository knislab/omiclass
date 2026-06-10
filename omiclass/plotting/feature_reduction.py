import io
import matplotlib.pyplot as plt
import numpy as np
from omiclass.utils.pdf_export import save_pdf_plot


def plot_feature_reduction_curve(result, title: str) -> str:
    title = title.replace("_", " ").title()
    plt.figure(figsize=(7, 5))

    for metric, mean_vals in result.mean.items():
        mean = np.array(mean_vals)
        std = np.array(result.std[metric])
        k = result.k_values

        plt.plot(k, mean, marker="o", label=metric)
        plt.fill_between(k, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Number of genes")
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()

    buf = io.StringIO()
    plt.savefig(buf, format="svg")
    save_pdf_plot(plt.gcf(), "feature_reduction_curve")
    plt.close()
    return buf.getvalue()
