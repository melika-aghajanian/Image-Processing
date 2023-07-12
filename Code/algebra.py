import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from PIL import Image


def denoise(image, k=100):
    if isinstance(image, str):
        # Load image and convert to numpy array
        img = Image.open(image).convert('RGB')
        pixels = np.array(img)
    else:
        # Input is already a numpy array
        pixels = image

    # Reshape pixel array to matrix and perform SVD decomposition
    matrix = pixels.reshape(-1, 3)
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Truncate singular values to retain top k components
    s[k:] = 0

    # Reconstruct matrix with truncated SVD and reshape to image shape
    recon_matrix = U.dot(np.diag(s)).dot(Vt)
    recon_pixels = np.clip(recon_matrix, 0, 255).astype(np.uint8)

    # Convert numpy array back to PIL image
    recon_image = Image.fromarray(recon_pixels.reshape(*pixels.shape))

    return recon_image


def gaussian_noise(shape, mean=0, stddev=1):
    # Generate normal noise with mean 0 and standard deviation 1
    noise = np.random.normal(0, 1, shape)
    # Scale and shift by mean and standard deviation
    return noise * stddev + mean


images = ['i1.jpg', 'i2.jpg', 'i3.jpg', 'i4.jpg', 'i5.jpg', 'i6.jpg', 'i7.jpg', 'i8.jpg', 'i9.jpg', 'i10.jpg', ]

for i in range(len(images)):
    # Load image
    img = plt.imread(images[i])
    # Generate Gaussian noise
    shape = img.shape
    noise = gaussian_noise(shape, random.randint(-1, 1))

    # Add noise to image
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # denoising
    # image = np.array(img)
    k = 100
    denoised_img = denoise(noisy_img, k)

    # Display original, noisy and denoised images side by side
    fig, ax = plt.subplots(1, 3, figsize=(25, 25))
    ax[0].imshow(img)
    ax[0].set_title('Original')
    ax[1].imshow(noisy_img)
    ax[1].set_title('Noisy')
    ax[2].imshow(denoised_img)
    ax[2].set_title('Denoised')
    plt.show()
