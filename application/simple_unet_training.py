import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os
from project_image_inpainting.services.simple_unet.training_service import TrainingService
from project_image_inpainting.models.train_configuration import TrainConfiguration
from project_image_inpainting.models.simple_unet import SimpleUnet


if __name__ == "__main__":
    image = torch.tensor([
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0]
    ], dtype=torch.float32)

    num_steps = 3
    start, end = 0, 1
    timesteps = TrainingService.time_step_creation(
        start=start,
        end=end,
        num_steps=num_steps
    )
    noisy_images = TrainingService.noisy_images(
        image=image,
        timesteps=timesteps
    )

    # TrainingService.plot_image_and_noisy_images(
    #     image=image,
    #     noisy_images=noisy_images
    # )

    # Training
    model = SimpleUnet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    num_epochs = 400

    train_configs = TrainConfiguration(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs
    )

    loss, trained_model = TrainingService.train_denoising_model(
        image=image,
        train_configs=train_configs
    )

    # 5. Dự đoán với ảnh nhiễu
    noisy_test = TrainingService.noising_process(
        x_0=image,
        t=torch.tensor(0.2)
    ).unsqueeze(0).unsqueeze(0)
    denoised_output = trained_model(noisy_test).squeeze().detach()

    # Hiển thị ảnh gốc, ảnh nhiễu nhẹ và ảnh sau khi khử nhiễu
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Gốc (số 5)")
    axes[1].imshow(noisy_test.squeeze(), cmap='gray')
    axes[1].set_title("Ảnh Nhiễu Nhẹ")
    axes[2].imshow(denoised_output, cmap='gray')
    axes[2].set_title("Ảnh Sau Khử Nhiễu")
    plt.savefig(
        os.path.join(
            "logs",
            f"predicted_image_{str(int(time.time()))}"
        )
    )
