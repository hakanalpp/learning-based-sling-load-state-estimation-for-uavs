import cv2
import numpy as np
import torch


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)
