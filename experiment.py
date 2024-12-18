from itertools import product
import os
from tqdm import tqdm
import json
from typing import List
import argparse
import subprocess
import pathlib
from dataclasses import dataclass


##### Experiment dependent params #####

params = {
            "d": [2, 3, 4, 6, 8, 12, 16, 24, 32],
            "D": [2, 4, 8, 16, 32],
            "save_folder": [os.path.join("logs/parallel_bs_256", f"run_{i}") for i in range(100)],
            "batch_size": [256],
         }

def filter_condition(arg):
    return arg["d"] <= arg["D"]

def launcher(args: List[dict]):
    from train import run, HParams
    for arg in tqdm(args):
        print(arg)
        hparams = HParams(**arg)
        run(hparams)

def dispatch_node(args_file_path):
    sbatch_file = "/storage/home/hcoda1/6/ahavrilla3/p-wliao60/alex/repos/Notes/experimentation/optimization/scripts/distributed_run.sh"
    subprocess.Popen(["sbatch", f"{sbatch_file}", f"{args_file_path}"])

##### Experiment independent params #####
               
def distributed_launcher(args: List[dict], num_procs: int, args_folder: str):
    arg_batch_size = len(args) // num_procs
    arg_batches = [args[arg_batch_size*i : arg_batch_size*(i+1)] for i in range(num_procs)]
    # Add last couple samples to last batch
    arg_batches[-1] += args[arg_batch_size*num_procs:]
    for i, arg_batch in enumerate(arg_batches):
        print(f"Len of {i}th arg_batch: ", len(arg_batch))
        # Save args to json file
        args_file_path = os.path.join(args_folder, f"{i}.json")
        with open(args_file_path, "w") as f:
            json.dump(arg_batch, f)
        # Launch job
        dispatch_node(args_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["local", "distributed"])
    parser.add_argument("--args_file", type=str, default=None, help="During local training, path to (optional) args file.\
                                                                     During distributed training, path to folder to dump arg files for each sub-process.")
    parser.add_argument("--num_procs", type=int, default=None)
    cmd_args = parser.parse_args()
    
    # Generate combination of args to sweep
    args = list(product(*params.values()))
    args = [{k: arg[i] for i, k in enumerate(params.keys())} for arg in args]
    args = [arg for arg in args if filter_condition(arg)]
    print("Total runs: ", len(args))
    
    if cmd_args.mode == "local":
        if cmd_args.args_file is not None:
            with open(cmd_args.args_file, "r") as f:
                args = json.load(f)
        print("Total (local) runs: ", len(args))
        launcher(args)
    elif cmd_args.mode == "distributed":
        assert cmd_args.args_file is not None and cmd_args.num_procs is not None
        pathlib.Path(cmd_args.args_file).mkdir(parents=True, exist_ok=True)
        distributed_launcher(args, args_folder=cmd_args.args_file, num_procs=cmd_args.num_procs)