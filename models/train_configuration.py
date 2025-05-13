import torch.nn as nn


class PropertyInjectorMeta(type):
    def __new__(cls, name, bases, dct):
        allowed_attrs = dct.get("ALLOWED_ATTRS", [])

        def make_property(attr_name):
            return property(lambda self: getattr(self, f"_{attr_name}"))

        for attr in allowed_attrs:
            dct[attr] = make_property(attr)

        print(dct)

        return super().__new__(cls, name, bases, dct)


class TrainConfiguration(nn.Module, metaclass=PropertyInjectorMeta):
    """Train Configurations"""

    ALLOWED_ATTRS = [
        "model", "optimizer", "criterion", "num_epochs"
    ]

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        num_epochs
    ) -> None:
        super(TrainConfiguration, self).__init__()

        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._num_epochs = num_epochs
