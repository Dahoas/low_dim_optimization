"""
Experiment goal: Optimize NN to approximate function on a manifold.
Define:
+ d: intrinsic dimension
+ D: extrinsic dimension
Observe:
1. Convergence rate as a function of d, D
2. Rank of gradient updates as a function of d, D
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from tqdm import tqdm
import scipy
import numpy as np
import json
import pathlib
from dataclasses import dataclass

from model import MLP
from data import SphereDataset
from utils import make_plots


device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_manifold_task(d, D, n):
    """ 
    Generates a binary classification dataset by calling 'sklearn.datasets.make_classification' and projecting the 
    resulting dataset to the d-dimensional sphere. We then rotate the result to embed the sphere in R^D
    + d: intrinsic data dimension
    + D: extrinsic data dimension
    + n: number of samples
    """
    # Sample random classification dataset
    X, y = make_classification(n_samples=n,
                               n_features=d,
                               n_informative=d,
                               n_classes=2,
                               n_redundant=0,)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long).reshape((-1))
    # Project to sphere
    X = X / X.norm(p="fro", dim=-1).reshape((-1, 1))
    # Rotate into higher dimension
    rotation = scipy.stats.special_ortho_group.rvs(D)
    X_D = np.zeros((n, D))
    X_D[:, :d] = X.numpy()
    X = torch.tensor((rotation @ X_D.T).T, dtype=torch.float32)
    return X, y

@dataclass
class HParams:
    """
    Dataclass defining hyper-parameters of experimental interest.
    """
    d: int = 2  # intrinsic data dimension
    D: int = 4  # extrinsic data dimension
    n: int = 10_000 # num samples
    
    batch_size: int = 32  # batch size
    save_grads: bool = False  # flag to save grads

    save_folder: str = "logs" # save folder


def run(hparams: HParams):
    d, D, n = hparams.d, hparams.D, hparams.n
    save_folder = hparams.save_folder
    batch_size = hparams.batch_size
    save_grads = hparams.save_grads
    # Sample the data
    train_size = 0.95
    train_size = int(0.95 * n)
    val_size = n - train_size
    X, y = generate_manifold_task(d=d, D=D, n=n)
    print(X.shape, y.shape)
    print(X[:2])
    print(y[:2])
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    dataset_train = SphereDataset(X_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataset_val = SphereDataset(X_val, y_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    def get_next_sample(dataiter, dataloader):
        try:
            X, y = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            X, y = next(dataiter)
        X, y = X.to(device), y.to(device)
        return X, y, dataiter

    # Define the model
    L = 3
    h_dim = 64
    out_dim = 2  # logits for binary classification
    model = MLP(L=L, in_dim=D, h_dim=h_dim, out_dim=out_dim).to(device)

    lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_iters = 10000
    n_val_iters = 100
    train_losses, val_losses, steps, loss_grads = [], [], [], []
    dataiter_train = iter(dataloader_train)
    for step in range(n_iters):
        X_batch, y_batch, dataiter = get_next_sample(dataiter_train, dataloader_train)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        
        loss_tensor = loss
        loss = loss.item()
        if step % n_val_iters == 0:
            steps.append(step)
            train_losses.append(loss)
            with torch.no_grad():
                val_loss = 0
                dataiter_val = iter(dataloader_val)
                n_val_batches = len(dataiter_val)
                for batch in dataiter_val:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    val_loss += criterion(out, y)
            val_loss = val_loss.item() / n_val_batches
            val_losses.append(val_loss)
            print(f"Step {step} / {n_iters}.....")
            print(f"Train loss: {loss:.4f}")
            print(f"Val loss: {val_loss:.4f}")
            
            if save_grads:
                offset = -4
                # We'll start by inspecting the grads of the output layer
                loss_grad = list(model.parameters())[offset].grad
                loss_grads.append(loss_grad.cpu())


    save_folder = os.path.join(save_folder, f"binary_n_{n}_D_{D}_d_{d}")
    save_folder = pathlib.Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    save_folder = str(save_folder)

    stats = {
        "steps": steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "d": d,
        "D": D,
        "n": n,
    }

    num_drop_start = 5 # Number of terms to drop towards start of training
    a = make_plots(train_losses=train_losses,
                   val_losses=val_losses,
                   steps=steps,
                   D=D,
                   d=d,
                   save_folder=save_folder,
                   num_drop_start=num_drop_start,)

    stats["log_linear_fit"] = a
    with open(os.path.join(save_folder, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    if save_grads:
        grads = torch.stack(loss_grads, dim=0)
        print("Grad shape: ", grads.shape)
        grad_path = os.path.join(save_folder, "grads.pt")
        torch.save(grads, grad_path)


#############################################################################


def configurator():
    import sys
    from ast import literal_eval
    overrides = dict()
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            overrides[key] = attempt
    return overrides


if __name__ == "__main__":
    overrides = configurator()
    hparams = HParams(**overrides)
    print("HParams: ", hparams)
    run(hparams)