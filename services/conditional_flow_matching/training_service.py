import torch
import project_image_inpainting.configs.logger import Logger


class TrainingService:
    """Training Service"""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger().get_logger()

    @staticmethod
    def perform_training(train_configs):
        """Perform training"""

        device = TrainingService.DEVICE
        model = train_configs.model.to(device)
        optimizer = train_configs.optimizer
        n_epochs = train_configs.num_epochs

        for epoch in range(n_epochs):
            losses = []

            for i, data in enumerate(train_configs.trainloader):
                optimizer.zero_grad()
                x1 = data[0].to(device)
                y = data[1].to(device)
                x0 = torch.randn_like(x1).to(device)
                t = torch.rand(1).to(device)
                xt = t * x1 + (1 - t) * x0 # Interpolate data point
                ut = x1 - x0 # True difference (velocity)

                # Model prediction
                vt = model(t, xt, y)
                loss = torch.mean((vt-ut)**2)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            avg_loss = sum(losses) / len(losses)
            TrainingService.logger(f"epoch: {epoch}, loss: {avg_loss:.4}")
        
        return model, avg_loss

    @staticmethod
    def euler_method(
        model,
        x,
        t_steps,
        dt,
        generated_class_list=None
    ):
        y = x
        y_values = [y]

        for t in t_steps[1:]:
            dy = model(t, y, generated_class_list)
            y = y + dy * dt
            y_values.append(y)

        return torch.stack(y_values)



