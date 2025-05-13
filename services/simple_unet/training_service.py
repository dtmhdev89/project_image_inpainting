import torch
import matplotlib.pyplot as plt


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

        _fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        for i, noisy_img in enumerate(noisy_images):
            if i == 0:
                axes[0].imshow(image, cmap='gray')
                axes[0].set_title("Hình ảnh Gốc")
            else:
                axes[i].imshow(noisy_img, cmap='gray')
                axes[i].set_title(f"Hình ảnh với Nhiễu {i}")

        plt.show()
    
    @staticmethod
    def train_denoising_model(image, train_configs):
        for epoch in range(train_configs.num_epochs):
            # Lấy một giá trị t ngẫu nhiên
            t = torch.rand(1)
            noisy_image = TrainingService.noising_process(
                image,
                t
            ).unsqueeze(0).unsqueeze(0)
            _clean_image = image.unsqueeze(0).unsqueeze(0)

            train_configs.optimizer.zero_grad()
            output = train_configs.model(noisy_image)
            loss = train_configs.criterion(output, noisy_image)
            loss.backward()
            train_configs.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{train_configs.num_epochs}], \
                    Loss: {loss.item():.4f}")

        return loss, train_configs.model
