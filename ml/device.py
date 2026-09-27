from importlib import import_module
from typing import Literal

import torch

DeviceBackend = Literal[
    "auto",
    "cpu",
    "cuda",
    "directml",
]


def get_device(
    backend: DeviceBackend = "auto",
) -> tuple[torch.device, str]:
    """
    Select the device used for neural-network computation.

    auto:
        1. CUDA, if available
        2. DirectML, if installed
        3. CPU otherwise

    A backend can also be requested explicitly.
    """

    if backend == "cpu":
        return torch.device("cpu"), "CPU"

    if backend in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA"

        if backend == "cuda":
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

    if backend in ("auto", "directml"):
        try:
            torch_directml = import_module(
                "torch_directml"
            )
        except ModuleNotFoundError:
            if backend == "directml":
                raise RuntimeError(
                    "DirectML was requested but "
                    "torch-directml is not installed."
                )
        else:
            return (
                torch_directml.device(),
                "DirectML",
            )

    return torch.device("cpu"), "CPU"