import torch


class TrainingService:
    """Training Service for Simple Unet model"""

    @staticmethod
    def noising_process(x_0, t):
        """Adding noise to original image at eact time step"""

        noise = torch.rand_like(x_0) * t

        return x_0 + noise

    @staticmethod
    def time_step_creation(start: float, end: float, num_steps: int):
        """Create time step value for each time step -- Betas"""

        # Linear time step
        return torch.linspace(start, end, num_steps)


    @staticmethod
    def noisy_images(image, timesteps):
        """Create noisy image for each time step"""

        return [
            TrainingService.noising_process(image, t)
            for t in timesteps
        ]
    
    @staticmethod
    def plot_image_and_noisy_images(image, noisy_images):
        """Display image and noisy images of each time step"""
        
