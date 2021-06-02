import tensorflow as tf
import numpy as np
from peak_finding import find_local_peaks, find_global_peaks, make_centered_bboxes, crop_bboxes, describe_tensors


class TopDownInferenceModel(tf.keras.Model):
    def __init__(
        self,
        centroid_base_model,
        centroid_input_size,
        centroid_input_scale,
        centroid_output_stride,
        crop_size,
        td_base_model,
        td_output_stride,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.centroid_base_model = centroid_base_model
        self.centroid_input_size = centroid_input_size
        self.centroid_input_scale = centroid_input_scale
        self.centroid_output_stride = centroid_output_stride
        self.crop_size = crop_size
        self.td_base_model = td_base_model
        self.td_output_stride = td_output_stride
        
    @tf.function
    def stage1(self, imgs):
        """Stage 1: Find centroids"""
        # Preprocessing
        X = tf.cast(imgs, tf.float32) / 255
        X = tf.image.resize(X, self.centroid_input_size)

        # Forward pass
        centroid_cms = self.centroid_base_model(X)

        # Find centroid points for the entire batch
        #     centroids: (n_centroids, 2)
        #     centroid_vals: (n_centroids,)
        #     centroid_sample_inds: (n_centroids,)
        centroids, centroid_vals, centroid_sample_inds, _ = find_local_peaks(centroid_cms, threshold=0.2, refinement="integral")

        # Adjust coordinates for output stride and input scale
        centroids = ((centroids * self.centroid_output_stride) / self.centroid_input_scale) + 0.5

        # TODO: Deal with case where no centroids are found (without resorting to RaggedTensors)

        # Make centered bboxes and crops
        #     bboxes: (n_centroids, 4)
        #     crops: (n_centroids, crop_size, crop_size, channels)
        bboxes = make_centered_bboxes(centroids, self.crop_size, self.crop_size)
        crops = crop_bboxes(imgs, bboxes, centroid_sample_inds)

        # Store crop offsets for coordinate adjustment in stage 2
        #     crop_offsets: (n_centroids, 2)
        crop_offsets = centroids - (self.crop_size / 2)

        return {"crops": crops, "crop_offsets": crop_offsets, "sample_inds": centroid_sample_inds}
    
    @tf.function
    def stage2(self, crops, crop_offsets):
        """Stage 2: Predict pose in each crop"""
        # Preprocessing
        X = tf.cast(crops, tf.float32) / 255

        # Forward pass
        cms = self.td_base_model(X)

        # Find keypoints in each crop
        #     pts: (n_centroids, n_nodes, 2)
        #     vals: (n_centroids, n_nodes)
        pts, vals = find_global_peaks(cms, threshold=0.2, refinement="integral")

        # Adjust coordinates for output stride
        pts = pts * self.td_output_stride

        # Adjust for bbox crops
        pts = pts + tf.expand_dims(crop_offsets, axis=1)

        return {"instance_peaks": pts, "instance_peak_vals": vals}
        
    def call(self, imgs):
        preds1 = self.stage1(imgs)
        preds2 = self.stage2(preds1["crops"], preds1["crop_offsets"])
        preds2["sample_inds"] = preds1["sample_inds"]
        return preds2


class TopDownIDInferenceModel(tf.keras.Model):
    def __init__(
        self,
        centroid_base_model,
        centroid_input_size,
        centroid_input_scale,
        centroid_output_stride,
        crop_size,
        td_base_model,
        td_output_stride,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.centroid_base_model = centroid_base_model
        self.centroid_input_size = centroid_input_size
        self.centroid_input_scale = centroid_input_scale
        self.centroid_output_stride = centroid_output_stride
        self.crop_size = crop_size
        self.td_base_model = td_base_model
        self.td_output_stride = td_output_stride
        
    @tf.function
    def stage1(self, imgs):
        """Stage 1: Find centroids"""
        # Preprocessing
        X = tf.cast(imgs, tf.float32) / 255
        X = tf.image.resize(X, self.centroid_input_size)

        # Forward pass
        centroid_cms = self.centroid_base_model(X)

        # Find centroid points for the entire batch
        #     centroids: (n_centroids, 2)
        #     centroid_vals: (n_centroids,)
        #     centroid_sample_inds: (n_centroids,)
        centroids, centroid_vals, centroid_sample_inds, _ = find_local_peaks(centroid_cms, threshold=0.2, refinement="integral")

        # Adjust coordinates for output stride and input scale
        centroids = ((centroids * self.centroid_output_stride) / self.centroid_input_scale) + 0.5

        # TODO: Deal with case where no centroids are found (without resorting to RaggedTensors)

        # Make centered bboxes and crops
        #     bboxes: (n_centroids, 4)
        #     crops: (n_centroids, crop_size, crop_size, channels)
        bboxes = make_centered_bboxes(centroids, self.crop_size, self.crop_size)
        crops = crop_bboxes(imgs, bboxes, centroid_sample_inds)

        # Store crop offsets for coordinate adjustment in stage 2
        #     crop_offsets: (n_centroids, 2)
        crop_offsets = centroids - (self.crop_size / 2)

        return {"crops": crops, "crop_offsets": crop_offsets, "sample_inds": centroid_sample_inds}
    
    @tf.function
    def stage2(self, crops, crop_offsets):
        """Stage 2: Predict pose in each crop"""
        # Preprocessing
        X = tf.cast(crops, tf.float32) / 255

        # Forward pass
        cms, class_probs = self.td_base_model(X)

        # Find keypoints in each crop
        #     pts: (n_centroids, n_nodes, 2)
        #     vals: (n_centroids, n_nodes)
        pts, vals = find_global_peaks(cms, threshold=0.2, refinement="integral")

        # Adjust coordinates for output stride
        pts = pts * self.td_output_stride

        # Adjust for bbox crops
        pts = pts + tf.expand_dims(crop_offsets, axis=1)

        return {"instance_peaks": pts, "instance_peak_vals": vals, "class_probabilities": class_probs}
        
    def call(self, imgs):
        preds1 = self.stage1(imgs)
        preds2 = self.stage2(preds1["crops"], preds1["crop_offsets"])
        preds2["sample_inds"] = preds1["sample_inds"]
        return preds2


class SingleInstanceInferenceModel(tf.keras.Model):
    def __init__(
        self,
        base_model,
        output_stride,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.output_stride = output_stride

    @tf.function
    def call(self, imgs):
        # Preprocessing
        X = tf.cast(imgs, tf.float32) / 255

        # Forward pass
        cms = self.base_model(X)

        # Find keypoints in each crop
        #     pts: (n, n_nodes, 2)
        #     vals: (n, n_nodes)
        pts, vals = find_global_peaks(cms, threshold=0.2, refinement="integral")

        # Adjust coordinates for output stride
        pts = pts * self.output_stride

        return {"instance_peaks": pts, "instance_peak_vals": vals}
