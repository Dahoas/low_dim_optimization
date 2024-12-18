import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit


def linear_func(x, a, b):
            return a * x + b

def get_linear_fit(xs, ys):
    return curve_fit(linear_func, xs, ys)[0]

def make_plots(train_losses, 
               val_losses,
               steps,
               D, 
               d, 
               save_folder,
               num_drop_start,):
    if save_folder is not None:
        plt.clf()
        plt.plot(steps, train_losses, label="Train loss")
        plt.plot(steps, val_losses, label="Val loss")
        plt.legend()
        plt.title("Losses")
        
        file_path = os.path.join(save_folder, f"loss_D_{D}_d_{d}.png")
        plt.savefig(file_path)

    plt.clf()

    steps = steps[num_drop_start:]
    val_losses = val_losses[num_drop_start:]

    steps = np.array(steps)
    steps = np.log10(1+steps)
    val_losses = np.log10(val_losses)
    a, b = get_linear_fit(steps, val_losses)
    if save_folder is not None:
        linear_ys = linear_func(steps, a, b)
        plt.plot(steps, val_losses, "-o", label="Val loss")
        plt.plot(steps, linear_ys, "--", label=f"fit: a={a:.4f}, 1/a={1/a:.4f}")

        plt.title(f"Log Val Loss vs. Log Number of Train Samples")
        plt.xlabel("(log) Num steps")
        plt.ylabel("(log) Val loss")
        plt.legend()

        file_name = os.path.join(save_folder, f"loss_log_log_D_{D}_d_{d}.png")
        plt.savefig(file_name)
    return a