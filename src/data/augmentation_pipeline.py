import cv2
import numpy as np
import random
import torch
import torchvision.transforms as T
from typing import Tuple


class DepthAugmentation:
    """
    Applies data augmentation as per DepthPro paper - Appendix section C.
    Includes Gaussian blur, color jitter, horizontal flip, and field-of-view (FOV) augmentation.
    """

    def __init__(self, prob_blur: float = 0.3, prob_color_jitter: float = 0.3, prob_flip: float = 0.1,
                 prob_fov: float = 0.3, fov_range: Tuple[int, int] = (80, 100)):
        """
        Initializes the augmentation parameters.

        Args:
            prob_blur (float): Probability of applying Gaussian blur.
            prob_color_jitter (float): Probability of applying color jitter.
            prob_flip (float): Probability of applying horizontal flip.
            prob_fov (float): Probability of applying FOV augmentation.
            fov_range (Tuple[int, int]): Range for FOV scaling as percentages.
        """
        self.prob_blur = prob_blur
        self.prob_color_jitter = prob_color_jitter
        self.prob_flip = prob_flip
        self.prob_fov = prob_fov
        self.fov_range = fov_range

    @staticmethod
    def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
        """Applies Gaussian blur with a random kernel size."""
        ksize = random.choice([3, 5])  # Random kernel size
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    @staticmethod
    def apply_color_jitter(image: np.ndarray) -> np.ndarray:
        """Applies random color jittering using torchvision."""
        color_jitter = T.ColorJitter(brightness=random.choice(np.linspace(0.1, 0.3)),
                                     contrast=random.choice(np.linspace(0.1, 0.3)),
                                     saturation=random.choice(np.linspace(0.1, 0.2)),
                                     hue=random.choice(np.linspace(0.1, 0.17)))
        image_torch = T.ToPILImage()(torch.tensor(image.transpose(2, 0, 1)))
        image_torch = color_jitter(image_torch)
        return np.array(image_torch, dtype=np.float32)

    @staticmethod
    def apply_flip(image: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies horizontal flip to the image and depth map."""
        image_flipped = np.fliplr(image)
        depth_flipped = np.fliplr(depth)
        return image_flipped, depth_flipped

    def apply_fov_augmentation(self, image: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies FOV augmentation by scaling the image and depth map."""
        fov_factor = random.uniform(*self.fov_range) / 100.0
        h, w = image.shape[:2]
        new_w = int(w * fov_factor)
        new_h = int(h * fov_factor)

        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Center crop or pad to original size
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        image_resized = cv2.copyMakeBorder(image_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
        depth_resized = cv2.copyMakeBorder(depth_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)

        return image_resized, depth_resized

    def __call__(self, image: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies the selected augmentations based on their probabilities."""

        if random.random() < self.prob_blur:
            image = self.apply_gaussian_blur(image)

        if random.random() < self.prob_color_jitter:
            image = self.apply_color_jitter(image)

        if random.random() < self.prob_flip:
            image, depth = self.apply_flip(image, depth)

        if random.random() < self.prob_fov:
            image, depth = self.apply_fov_augmentation(image, depth)

        return image, depth


# Example Usage
if __name__ == "__main__":
    image = np.random.rand(256, 256, 3).astype(np.float32)  # Example image
    depth = np.random.rand(256, 256).astype(np.float32)  # Example depth map

    augmenter = DepthAugmentation()
    aug_image, aug_depth = augmenter(image, depth)
    print("Augmented Image Shape:", aug_image.shape)
    print("Augmented Depth Shape:", aug_depth.shape)
