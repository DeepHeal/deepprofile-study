import tensorflow as tf
import numpy as np

class IntegratedGradients:
    """
    TensorFlow 2.x compatible implementation of Integrated Gradients.
    Calculates feature importance by integrating gradients along a path from a baseline to the input.
    """

    def __init__(self, model):
        """
        Args:
            model (tf.keras.Model): The model to explain.
        """
        self.model = model

    def interpolate_images(self, baseline, image, alphas):
        """
        Generates interpolated images between baseline and input image.
        """
        alphas_x = alphas[:, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(image, axis=0)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    @tf.function
    def compute_gradients(self, images, target_class_idx):
        """
        Computes gradients of the target class output with respect to inputs.
        """
        with tf.GradientTape() as tape:
            tape.watch(images)
            outputs = self.model(images)

            # Check output format. The VAE encoder in process_drug_data.py returns [z_mean, z_log_var, z].
            # We want to explain z_mean.
            if isinstance(outputs, (list, tuple)):
                z_mean = outputs[0]
            else:
                z_mean = outputs

            # Target output: The value of the specific latent dimension
            target_output = z_mean[:, target_class_idx]

        return tape.gradient(target_output, images)

    def explain(self, input_sample, target_idx, baseline=None, m_steps=50):
        """
        Calculates Integrated Gradients for a single input sample and a specific target output index.

        Args:
            input_sample (numpy.ndarray): The input feature vector (shape: (n_features,)).
            target_idx (int): The index of the latent dimension to explain.
            baseline (numpy.ndarray, optional): The baseline input. Defaults to zeros.
            m_steps (int): Number of steps for integration.

        Returns:
            numpy.ndarray: Feature importance scores (shape: (n_features,)).
        """
        input_sample = tf.convert_to_tensor(input_sample, dtype=tf.float32)

        if baseline is None:
            baseline = tf.zeros_like(input_sample)
        else:
            baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

        # 1. Generate alphas
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

        # 2. Generate interpolated images
        interpolated_images = self.interpolate_images(baseline=baseline,
                                                      image=input_sample,
                                                      alphas=alphas)

        # 3. Compute gradients for each interpolated image
        gradients = self.compute_gradients(interpolated_images, target_idx)

        # 4. Integral approximation (Riemann Trapezoidal)
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        avg_gradients = tf.reduce_mean(grads, axis=0)

        # 5. Scale by (input - baseline)
        integrated_gradients = (input_sample - baseline) * avg_gradients

        return integrated_gradients.numpy()
