import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# --------------------- Weighted Mean Homomorphic Filter ---------------------
def weighted_mean_homomorphic_filter(image, gamma_low=0.5, gamma_high=2.0, c=1.0, d0=30):
    img_log = np.log1p(np.array(image, dtype="float"))  # log(1 + I)

    # Fourier Transform
    fft_image = fft2(img_log)
    fft_image = fftshift(fft_image)

    # Create Gaussian high-pass filter
    M, N = image.shape
    y, x = np.ogrid[:M, :N]
    center_y, center_x = M // 2, N // 2
    distance = (x - center_x) ** 2 + (y - center_y) ** 2
    H = (gamma_high - gamma_low) * (1 - np.exp(-c * distance / (d0 ** 2))) + gamma_low

    # Apply filter
    filtered = H * fft_image

    # Inverse FFT
    filtered = ifftshift(filtered)
    img_homomorphic = np.real(ifft2(filtered))
    img_homomorphic = np.expm1(img_homomorphic)  # inverse log

    # Normalize to [0, 255]
    img_homomorphic = np.clip((img_homomorphic - img_homomorphic.min()) /
                              (img_homomorphic.max() - img_homomorphic.min()) * 255, 0, 255)
    return img_homomorphic.astype(np.uint8)

# --------------------- Adaptive Histogram Equalization ---------------------
def adaptive_histogram_equalization(img, grid_size=(8, 8), clip_limit=40):
    h, w = img.shape
    tile_h, tile_w = h // grid_size[0], w // grid_size[1]
    output = np.zeros_like(img, dtype=np.uint8)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y0 = i * tile_h
            y1 = (i + 1) * tile_h if i != grid_size[0] - 1 else h
            x0 = j * tile_w
            x1 = (j + 1) * tile_w if j != grid_size[1] - 1 else w
            tile = img[y0:y1, x0:x1]

            # Histogram clipping
            hist, _ = np.histogram(tile.flatten(), bins=256, range=[0, 256])
            excess = np.maximum(hist - clip_limit, 0)
            clipped_hist = hist.copy()
            clipped_hist[clipped_hist > clip_limit] = clip_limit
            redistribute = excess.sum() // 256
            clipped_hist += redistribute
            diameter=5; sigma_color=0; sigma_space=0
            # Compute CDF
            cdf = np.cumsum(clipped_hist)
            cdf = 255 * cdf / cdf[-1]
            tile_equalized = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

    return tile_equalized

# --------------------- Preprocessing Pipeline ---------------------
def preprocess_image(image):
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    wm_filtered = weighted_mean_homomorphic_filter(image)
    ahe_result = adaptive_histogram_equalization(wm_filtered)
    return ahe_result

