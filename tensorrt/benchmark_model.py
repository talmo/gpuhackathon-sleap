"""This script is written from benchmark_models.ipynb. Do not edit directly."""

# Disable tensorflow spam (needs to happen before tensorflow gets imported)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # set to "2" to see TensorRT errors
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

import numpy as np
import tensorflow as tf
from time import perf_counter
from trtutils import OptimizedModel
import system
import argparse
import pandas as pd

system.disable_preallocation()
# system.summary()


def log_result(data, filename="benchmark.csv"):
    """Append a dictionary of scalar keys as a row to a CSV file."""
    df = pd.DataFrame({k: [v] for k, v in data.items()})
    
    if os.path.exists(filename):
        df = pd.concat([
            pd.read_csv(filename),
            df,
        ])
    df.to_csv(filename, index=False)


def benchmark(data, model, **kwargs):
    """Benchmark a single model.
    
    Args:
        data: Numpy array of data that will serve as input of shape
            (n_imgs, height, width, channels) and dtype float32.
        model: Anything with a .predict() method
    
    Returns:
        A tuple of (latency, fps).
        
        latency: time per image in milliseconds
        fps: images per second
        
        These values will be returned for the second run of this function
        to discount graph tracing/warmup time.
    """
    t0 = perf_counter()
    model.predict(data, **kwargs)
    first_run = perf_counter() - t0

    t0 = perf_counter()
    model.predict(data, **kwargs)
    dt = perf_counter() - t0

    N = len(data)
    latency = (first_run * 1000) / N
    fps = N / first_run
    print(f"  First run: {latency:.2f} ms/img -> {fps:.2f} FPS")

    latency = (dt * 1000) / N
    fps = N / dt
    print(f"   Real run: {latency:.2f} ms/img -> {fps:.2f} FPS")
    
    return latency, fps


def benchmark_model_set(model, precision, batch_size, batches, log_filename="benchmark.csv"):
    """Benchmark a pair of TensorFlow and TensorRT models."""
    saved_model_path = f"data/{model}_savedmodel"  # SavedModel proto folder
    opt_model_path = f"data/{model}_trt_{precision}"

    tf_model = tf.keras.models.load_model(saved_model_path)
    opt_model = OptimizedModel(saved_model_dir=opt_model_path)

    N = int(batch_size * batches)
    img_shape = tuple(tf_model.inputs[0].shape[1:])
    data = np.zeros((N,) + img_shape, dtype="float32")

    print("==========")
    print(f"Model: {model}")
    print(f"Precision: {precision}")
    print("==========")
    print(f"tf.keras.Model (batch size = {batch_size}):")
    tf_latency, tf_fps = benchmark(data, tf_model, batch_size=batch_size)
    print("==========")
    print()
    print("==========")
    print(f"TensorRT optimized model (batch size = {batch_size}):")
    trt_latency, trt_fps = benchmark(data, opt_model, batch_size=batch_size)
    
    latency_delta = trt_latency - tf_latency
    latency_prc = (latency_delta / tf_latency) * 100
    
    fps_delta = trt_fps - tf_fps
    fps_prc = (fps_delta / tf_fps) * 100
    
    print(f"   Latency: {latency_delta:.1f} ({latency_prc:.1f}%)")
    print(f"       FPS: +{fps_delta:.1f} (+{fps_prc:.1f}%)")
    print("==========")
    
    log_result({
        "model": model,
        "precision": precision,
        "batch_size": batch_size,
        "engine_base": "tf",
        "engine": "tf",
        "latency": tf_latency,
        "fps": tf_fps,
        "img_height": img_shape[0],
        "img_width": img_shape[1],
        "img_channels": img_shape[2],
        "batches": batches,
        "n_imgs": N,
        "model_path": saved_model_path,
    }, filename=log_filename)
    
    log_result({
        "model": model,
        "precision": precision,
        "batch_size": batch_size,
        "engine_base": "trt",
        "engine": f"trt_{precision.lower()}",
        "latency": trt_latency,
        "fps": trt_fps,
        "img_height": img_shape[0],
        "img_width": img_shape[1],
        "img_channels": img_shape[2],
        "batches": batches,
        "n_imgs": N,
        "model_path": opt_model_path,
    }, filename=log_filename)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("precision", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--log", type=str, default="benchmark.csv")
    args = parser.parse_args()

    benchmark_model_set(args.model, args.precision, args.batch_size, args.batches, log_filename=args.log)
