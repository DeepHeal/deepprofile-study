import torch
import numpy as np

class IntegratedGradients:
    """
    PyTorch implementation of Integrated Gradients.
    Calculates feature importance by integrating gradients along a path from a baseline to the input.
    """

    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): The model to explain.
        """
        self.model = model
        self.device = next(model.parameters()).device

    def interpolate_images(self, baseline, image, alphas):
        """
        Generates interpolated images between baseline and input image.
        """
        alphas_x = alphas.unsqueeze(1) # (m_steps+1, 1)
        baseline_x = baseline.unsqueeze(0) # (1, features)
        input_x = image.unsqueeze(0) # (1, features)
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    def compute_gradients(self, inputs, target_class_idx):
        """
        Computes gradients of the target class output with respect to inputs.
        """
        inputs.requires_grad = True

        # Forward pass
        # The VAE model forward() returns (recon, mu, logvar).
        # We need to access the latent mean (mu) for explanation.
        outputs = self.model(inputs)

        if isinstance(outputs, (list, tuple)):
             mu = outputs[1] # mu is shape (batch_size, latent_dim)
        else:
             mu = outputs

        # Target output: The value of the specific latent dimension for each sample in the batch
        target_output = mu[:, target_class_idx].sum()

        # Backward pass
        # Compute gradient of the sum w.r.t inputs. This effectively computes individual gradients
        # because the samples in the batch are independent.
        grads = torch.autograd.grad(target_output, inputs)[0]

        return grads

    def explain(self, input_sample, target_idx, baseline=None, m_steps=50):
        """
        Calculates Integrated Gradients for a single input sample and a specific target output index.

        Args:
            input_sample (numpy.ndarray or torch.Tensor): The input feature vector (shape: (n_features,)).
            target_idx (int): The index of the latent dimension to explain.
            baseline (numpy.ndarray, optional): The baseline input. Defaults to zeros.
            m_steps (int): Number of steps for integration.

        Returns:
            numpy.ndarray: Feature importance scores (shape: (n_features,)).
        """
        # Ensure input is a tensor on the correct device
        if isinstance(input_sample, np.ndarray):
            input_sample = torch.tensor(input_sample, dtype=torch.float32)

        input_sample = input_sample.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_sample)
        elif isinstance(baseline, np.ndarray):
            baseline = torch.tensor(baseline, dtype=torch.float32).to(self.device)
        else:
            baseline = baseline.to(self.device)

        # 1. Generate alphas
        alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps+1).to(self.device)

        # 2. Generate interpolated images
        interpolated_images = self.interpolate_images(baseline=baseline,
                                                      image=input_sample,
                                                      alphas=alphas)

        # 3. Compute gradients for the batch of interpolated images
        gradients = self.compute_gradients(interpolated_images, target_idx)

        # 4. Integral approximation (Riemann Trapezoidal)
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        avg_gradients = torch.mean(grads, dim=0)

        # 5. Scale by (input - baseline)
        integrated_gradients = (input_sample - baseline) * avg_gradients

        return integrated_gradients.detach().cpu().numpy()
