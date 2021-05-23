def test_tensorflow():
    # import sleap
    # sleap.versions()
    # sleap.system_summary()
    import system
    system.summary()
    return system.is_gpu_system()


def test_tensorrt_import():
    try:
        import tensorrt
        print("tensorrt imported!")
        print(f"tensorrt version: {tensorrt.__version__}")
    except:
        print("failed to import tensorrt")
        return False

    try:
        assert tensorrt.Builder(tensorrt.Logger())
        print("created tensorrt.Builder!")
        return True
    except:
        print("failed to create tensorrt.Builder")
        return False


def test_optimize_model():
    import os
    import tensorflow as tf

    h5_model_path = "td_fast.210505_012601.centered_instance.n=1800.best_model.h5"
    saved_model_path = "test_saved_model"  # SavedModel proto folder
    if not os.path.exists(saved_model_path):
        # wget https://storage.googleapis.com/sleap-data/reference/flies13/td_fast.210505_012601.centered_instance.n%3D1800.zip
        model = tf.keras.models.load_model(h5_model_path, compile=False)
        model.save(saved_model_path)
        print(f"Saved: {h5_model_path} -> {saved_model_path}")

    # https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/2.%20Using%20the%20Tensorflow%20TensorRT%20Integration.ipynb
    # wget https://raw.githubusercontent.com/NVIDIA/TensorRT/master/quickstart/IntroNotebooks/helper.py
    from helper import ModelOptimizer
    opt_model = ModelOptimizer(saved_model_path)
    print(f"Created ModelOptimizer with: {saved_model_path}")

    PRECISION = "FP32" # Options are "FP32", "FP16", or "INT8"
    opt_model_path = f"{saved_model_path}_{PRECISION}"
    model_fp32 = opt_model.convert(opt_model_path, precision=PRECISION)

    if os.path.exists(opt_model_path):
        print(f"Converted model: {opt_model_path}")
        return True
    else:
        print(f"failed to convert model: {opt_model_path}")
        return False


def test_benchmark():
    import os
    import numpy as np
    import tensorflow as tf
    from time import perf_counter
    from helper import OptimizedModel

    saved_model_path = "test_saved_model"  # SavedModel proto folder
    PRECISION = "FP32" # Options are "FP32", "FP16", or "INT8"
    opt_model_path = f"{saved_model_path}_{PRECISION}"

    model = tf.keras.models.load_model(saved_model_path, compile=False)
    opt_model = OptimizedModel(saved_model_dir=opt_model_path)

    batch_size = 128
    N = int(batch_size * 8)
    data = np.zeros([N, 160, 160, 1], dtype="float32")

    def benchmark(model, **kwargs):
        t0 = perf_counter()
        model.predict(data, **kwargs)
        first_run = perf_counter() - t0

        t0 = perf_counter()
        model.predict(data, **kwargs)
        dt = perf_counter() - t0

        N = len(data)
        ms_per_img = (first_run * 1000) / N
        fps = N / first_run
        print(f"  First run: {ms_per_img:.2f} ms/img -> {fps:.2f} FPS")

        ms_per_img = (dt * 1000) / N
        fps = N / dt
        print(f"   Real run: {ms_per_img:.2f} ms/img -> {fps:.2f} FPS")

    print("==========")
    print("tf.keras.Model:")
    benchmark(model, batch_size=batch_size)
    print("==========")
    print()
    print("==========")
    print("TensorRT optimized model:")
    benchmark(opt_model)
    print("==========")


has_gpu = test_tensorflow()
can_import_tensorrt = test_tensorrt_import()

if can_import_tensorrt:
    can_optimize = test_optimize_model()

    if can_optimize:
        test_benchmark()