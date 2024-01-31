import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from math import log10


def signal_to_noise_ratio(original_image, noisy_image):
    """
    Calculates the Signal-to-Noise Ratio (SNR) between two images.

    Args:
        original_image: The original image.
        noisy_image: The noisy image.

    Returns:
        The SNR value.
    """
    mse = mean_squared_error(original_image, noisy_image)
    snr = 10 * np.log10(np.mean(original_image ** 2) / mse)
    return snr


def structural_similarity_index(original_image, noisy_image, win_size):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.

    Args:
        original_image: The original image.
        noisy_image: The noisy image.

    Returns:
        The SSIM value.
    """
    return ssim(original_image, noisy_image, win_size=win_size, channel_axis=3, multichannel=True)


def mean_absolute_error(original_image, noisy_image):
    """
    Calculates the Mean Absolute Error (MAE) between two images.

    Args:
        original_image: The original image.
        noisy_image: The noisy image.

    Returns:
        The MAE value.
    """
    mae = np.mean(np.abs(original_image - noisy_image))
    return mae


def peak_signal_to_noise_ratio(original_image, noisy_image):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        original_image: The original image.
        noisy_image: The noisy image.

    Returns:
        The PSNR value.
    """
    mse = mean_squared_error(original_image, noisy_image)
    psnr = 10 * log10((np.max(original_image) ** 2) / mse)
    return psnr
