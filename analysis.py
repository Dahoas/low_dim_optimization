import pathlib
import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils import make_plots


def log_linear_scaling():
    experiment_folder = "logs/parallel_bs_256"
    experiment_template = "run_*/*"
    files = list(pathlib.Path(experiment_folder).glob(experiment_template))
    print(files)
    results = dict()
    for exp in files:
        with open(os.path.join(str(exp), "stats.json")) as f:
            results[str(exp)] = json.load(f)
    print("Num results: ", len(results))

    # Compute averaged slope for each setup
    def compute_key(result):
        d, D = result["d"], result["D"]
        return f"{D}_{d}"

    grouped_results = defaultdict(list)
    for result in tqdm(results.values()):
        key = compute_key(result)
        log_linear_fit = make_plots(result["train_losses"],
                                    result["val_losses"],
                                    result["steps"],
                                    result["D"],
                                    result["d"],
                                    None,
                                    num_drop_start=5,)
        grouped_results[key].append(log_linear_fit)
        #grouped_results[key].append(result["log_linear_fit"])

    for k, v in grouped_results.items():
        grouped_results[k] = {"mean": np.mean(v), "max": np.max(v), "min": np.min(v)}
        
    print(json.dumps(grouped_results, indent=2))

    plt.clf()
    # First plot avg log_linear_fit with D held constant
    Ds = {
        2: [],
        4: [],
        8: [],
        16: [],
        32: [],
        }

    for k, v in grouped_results.items():
        D, d = k.split("_")
        D, d = int(D), int(d)
        print(D, d)
        Ds[D].append((d, v))

    for D, vs in Ds.items():
        print(D)
        vs = sorted(vs, key=lambda s: s[0])
        xs, ys = list(zip(*vs))
        means = [y["mean"] for y in ys]
        yerr = [y['max']-y['min'] for y in ys]
        print("yerr: ", len(yerr))
        print("means: ", len(means))
        print(xs)
        print(means)
        plt.plot(xs, means, "-o", label=f"D={D}")

        if D == 32:
            means = [round(y, 2) for y in means]
            print(list(zip(xs, means)))
            exit()

    plt.legend()
    plt.title("log linear vs. d")
    plt.savefig("figs/log_linear_vs_d.png")

    plt.clf()

    ds = {
        2: [],
        3: [],
        4: [],
        6: [],
        8: [],
        12: [],
        16: [],
        24: [],
        32: [],
        }

    for k, v in grouped_results.items():
        D, d = k.split("_")
        D, d = int(D), int(d)
        print(D, d)
        ds[d].append((D, v))

    for d, vs in ds.items():
        print(d)
        vs = sorted(vs, key=lambda s: s[0])
        xs, ys = list(zip(*vs))
        means = [y["mean"] for y in ys]
        yerr = [y['max']-y['min'] for y in ys]
        print("yerr: ", len(yerr))
        print("means: ", len(means))
        print(xs)
        print(means)
        plt.plot(xs, means, "-o", label=f"d={d}")

    plt.legend()
    plt.title("log linear vs. D")
    plt.savefig("figs/log_linear_vs_D.png")
    
    
def analyze_grads():
    grad_path = os.path.join("logs/binary_n_10000_D_8_d_8", "grads.pt")
    grads = torch.load(grad_path)
    ranks = []
    for grad in grads:
        rank = torch.linalg.matrix_rank(grad).type(torch.float32).reshape((1,))
        ranks.append(rank)
    ranks = torch.cat(ranks, dim=0)
    print("Mean: ", torch.mean(rank))
    print("Max: ", torch.max(ranks))
    print("Min: ", torch.min(ranks))
    print("STD: ", torch.std(ranks))
    
    
if __name__ == "__main__":
    log_linear_scaling()
    #analyze_grads()