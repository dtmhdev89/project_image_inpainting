import torch
import torchcfm.models.unet import UNetModel
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import time
import os
from project_image_inpainting.services.conditional_flow_matching.training_service import TrainingService
from project_image_inpainting.services.conditional_flow_matching.dataset_loader import DatasetLoader
from project_image_inpainting.models.train_configuration import TrainConfiguration


if __name__ == "__main__":
    device = TrainingService.DEVICE

    batch_size = 64

    dataset_loader = DatasetLoader(batch_size=batch_size)

    trainset = dataset_loader.build_trainset()
    train_loader = dataset_loader.build_train_loader(trainset=trainset)

    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        num_classes=10,
        class_cond=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 10

    train_configs = TrainConfiguration(
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        trainloader=train_loader
    )

    model, _ = TrainingService.perform_training(
        train_configs=train_configs
    )

    # Initial random image and class (optional)
    initial_image = torch.randn(100, 1, 28, 28, device=device)
    generated_class_list = torch.randint(0, 10, (100,), device=device)

    # Time parameters
    t_steps = torch.linspace(0, 1, 2, device=device)
    dt = t_steps[1] - t_steps[0]  # Time step

    # Solve the ODE using Euler method
    traj = TrainingService.euler_method(
        model=model,
        x=initial_image,
        t_steps=t_steps,
        dt=dt,
        generated_class_list=generated_class_list
    )

    grid = make_grid(
        traj[-1, :100].view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
    )
    img = ToPILImage()(grid)
    plt.imshow(img)
    save_path = os.path.join(
        "logs",
        f"cfm_with_advance_model_output_{str(int(time.time()))}.png"
    )
    plt.savefig(save_path)
    plt.show()
