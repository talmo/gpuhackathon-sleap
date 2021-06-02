"""This script is written from benchmark.ipynb. Do not edit directly."""

# Disable tensorflow spam (needs to happen before tensorflow gets imported)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # set to "2" to see TensorRT errors
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import system
from video import read_frames
from trtutils import OptimizedModel
from time import perf_counter


system.disable_preallocation()


def log_result(data, filename="benchmark.csv"):
    """Append a dictionary of scalar keys as a row to a CSV file."""
#     data = pd.DataFrame({k: [v] for k, v in data.items()})
    if os.path.exists(filename):
        data = pd.concat([
            pd.read_csv(filename),
            data,
        ])
    data.to_csv(filename, index=False)


def benchmark(model, precision, test_data_path, n_frames, batch_size, reps, grayscale):
    trt_model_path = f"{model}/trtmodel_FP{precision}"

    imgs = read_frames(test_data_path, np.arange(n_frames), grayscale=grayscale)
    trt_model = OptimizedModel(saved_model_dir=trt_model_path)

    res = None
    dts = []
    for rep in range(reps + 1):
        for i in range(0, n_frames, batch_size):
            t0 = perf_counter()
            preds = trt_model.predict(imgs[i:(i+batch_size)], numpy=True)
            dt = perf_counter() - t0
            if rep > 0:
                dts.append(dt)
    dts = np.array(dts)

    res_ = pd.DataFrame({"batch_size": np.full(dts.shape, batch_size), "dts": dts})
    if res is None:
        res = res_
    else:
        res = pd.concat([res, res_])

    res["fps"] = res["batch_size"] / res["dts"]
    res["model"] = model
    res["precision"] = precision
    res["trt_model_path"] = trt_model_path
    res["test_data_path"] = test_data_path
    res["n_frames"], res["img_height"], res["img_width"] = imgs.shape[:-1]
    
    log_result(res, filename="trt_benchmarks.csv")
    
    print(res.groupby("batch_size")[["fps", "dts"]].agg(["mean", "std"]).to_string())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("precision", type=str)
    parser.add_argument("test_data", type=str)
    parser.add_argument("--n_frames", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    parser.set_defaults(grayscale=True)
    args = parser.parse_args()

    benchmark(args.model, args.precision, args.test_data, args.n_frames, args.batch_size, args.reps, args.grayscale)
