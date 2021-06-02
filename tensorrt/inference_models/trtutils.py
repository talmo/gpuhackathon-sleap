# Adapted from:
# https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/helper.py
#
# (2021-05-22) talmo: Added batching functionality to OptimizedModel
#
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import tensorrt as trt
import os
import numpy as np

precision_dict = {
    "FP32": tf_trt.TrtPrecisionMode.FP32,
    "FP16": tf_trt.TrtPrecisionMode.FP16,
    "INT8": tf_trt.TrtPrecisionMode.INT8,
}

# For TF-TRT:

class OptimizedModel():
    def __init__(self, saved_model_dir = None):
        self.loaded_model_fn = None
        
        if not saved_model_dir is None:
            self.load_model(saved_model_dir)
    
    def predict(self, input_data, batch_size=None, numpy=True): 
        if self.loaded_model_fn is None:
            raise(Exception("Haven't loaded a model"))
            
        if batch_size is not None:
            all_inds = np.arange(len(input_data))
            all_preds = []
            for inds in np.array_split(all_inds, int(np.ceil(len(all_inds) / batch_size))):
                all_preds.append(self.predict(input_data[inds]))
            return all_preds
                
        x = tf.constant(input_data)
        preds = self.loaded_model_fn(x)
        if not numpy:
            return preds
        if type(preds) == dict:
            return {k: v.numpy() for k, v in preds.items()}
        else:
            return v.numpy()
        return preds
    
    def load_model(self, saved_model_dir):
        saved_model_loaded = tf.saved_model.load(saved_model_dir, tags=[tag_constants.SERVING])
        wrapper_fp32 = saved_model_loaded.signatures['serving_default']
        
        self.loaded_model_fn = wrapper_fp32

        
class ModelOptimizer():
    def __init__(self, input_saved_model_dir, calibration_data=None):
        self.input_saved_model_dir = input_saved_model_dir
        self.calibration_data = None
        self.loaded_model = None
        
        if not calibration_data is None:
            self.set_calibration_data(calibration_data)
        
        
    def set_calibration_data(self, calibration_data):
        
        def calibration_input_fn():
            yield (tf.constant(calibration_data.astype('float32')), )
        
        self.calibration_data = calibration_input_fn
        
        
    def convert(self, output_saved_model_dir, precision="FP32", max_workspace_size_bytes=8000000000, **kwargs):
        
        if precision == "INT8" and self.calibration_data is None:
            raise(Exception("No calibration data set!"))

        trt_precision = precision_dict[precision]
        conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_precision, 
                                                                       max_workspace_size_bytes=max_workspace_size_bytes,
                                                                       use_calibration= precision == "INT8")
        converter = tf_trt.TrtGraphConverterV2(input_saved_model_dir=self.input_saved_model_dir,
                                conversion_params=conversion_params)
        
        if precision == "INT8":
            converter.convert(calibration_input_fn=self.calibration_data)
        else:
            converter.convert()
            
        converter.save(output_saved_model_dir=output_saved_model_dir)
        
        return OptimizedModel(output_saved_model_dir)
    
    def predict(self, input_data):
        if self.loaded_model is None:
            self.load_default_model()
            
        return self.loaded_model.predict(input_data)
        
    def load_default_model(self):
        self.loaded_model = tf.keras.models.load_model('resnet50_saved_model')

        
def convert_to_savedmodel(model_path):
    import os
    import tensorflow as tf

    h5_model_path = f"{model}/best_model.h5"
    saved_model_path = f"{model}/savedmodel"  # SavedModel proto folder
    if not os.path.exists(saved_model_path):
        model = tf.keras.models.load_model(h5_model_path, compile=False)
        model.save(saved_model_path)
        print(f"Saved: {h5_model_path} -> {saved_model_path}")

        
def convert_to_trt(saved_model_path, opt_model_path, precision):
    # https://github.com/NVIDIA/TensorRT/blob/master/quickstart/IntroNotebooks/2.%20Using%20the%20Tensorflow%20TensorRT%20Integration.ipynb
    import os
    
    from trtutils import ModelOptimizer
    opt_model = ModelOptimizer(saved_model_path)
    print(f"Created ModelOptimizer with: {saved_model_path}")
    
    import numpy as np
    
    if precision == "INT8":
        import tensorflow as tf
        tf_model = tf.keras.models.load_model(saved_model_path)
        
        # not working:
        # InternalError:  Failed to feed calibration data
        # [[node TRTEngineOp_2_0 (defined at /mnt/helper.py:94) ]] [Op:__inference_pruned_20725]

        # Function call stack:
        # pruned
        N = 32
        calib_data = np.zeros((N,) + tuple(tf_model.inputs[0].shape[1:]))
        print("Set calibration data:", calib_data.shape)
        opt_model.set_calibration_data(calib_data)
    
    opt_model_path_ = f"{opt_model_path}_{precision}"
    opt_model_ = opt_model.convert(opt_model_path_, precision=precision)
    
    if os.path.exists(opt_model_path_):
        print(f"Converted model: {opt_model_path_}")
        return True
    else:
        print(f"Failed to convert model: {opt_model_path_}")
        return False