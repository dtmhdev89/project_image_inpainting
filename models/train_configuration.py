import torch.nn as nn


class PropertyInjectorMeta(type):
    def __new__(cls, name, bases, dct):
        allowed_attrs = dct.get("ALLOWED_ATTRS", [])

        def make_property(attr_name):
            def getter(self):
                return getattr(self, f"_{attr_name}")

            def setter(self, value):
                setattr(self, f"_{attr_name}", value)

            return property(getter, setter)

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
        num_epochs,
        criterion=None,
        trainloader=None
    ) -> None:
        super(TrainConfiguration, self).__init__()

        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._num_epochs = num_epochs
        self._trainloader = trainloader
