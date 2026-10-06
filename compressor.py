
import numpy as np
from sklearn.cluster import KMeans

def kmeans_quantize(image_array, k, seed=42):
    flat_pixels = image_array.reshape(-1, 3)

    model = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10
    )

    model.fit(flat_pixels)

    palette_k = np.round(
        model.cluster_centers_
    ).astype(np.uint8)

    new_pixels = palette_k[
        model.labels_
    ].reshape(image_array.shape)

    return new_pixels, palette_k

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def quality_metrics(original, compressed):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    psnr = peak_signal_noise_ratio(original, compressed, data_range=255)
    ssim = structural_similarity(
        original,
        compressed,
        channel_axis=2,
        data_range=255
    )
    return mse, psnr, ssim

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def quality_metrics(original, compressed):
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)

    psnr = peak_signal_noise_ratio(
        original,
        compressed,
        data_range=255
    )

    ssim = structural_similarity(
        original,
        compressed,
        channel_axis=2,
        data_range=255
    )

    return mse, psnr, ssim

from io import BytesIO
from PIL import Image

def png_rgb_size(image_array):
    buffer = BytesIO()

    image = Image.fromarray(image_array, mode="RGB")
    image.save(buffer, format="PNG")

    return len(buffer.getvalue())

def png_indexed_size(image_array):
    buffer = BytesIO()

    image = Image.fromarray(image_array, mode="RGB")
    image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    image.save(buffer, format="PNG")

    return len(buffer.getvalue())

def jpeg_size(image_array, quality=90):
    buffer = BytesIO()

    image = Image.fromarray(image_array, mode="RGB")
    image.save(buffer, format="JPEG", quality=quality)

    return len(buffer.getvalue())

def process_image(image_array, k):
    compressed, palette = kmeans_quantize(image_array, k)
    mse, psnr, ssim = quality_metrics(image_array, compressed)

    metrics = {
        "unique_colours": len(np.unique(compressed.reshape(-1, 3), axis=0)),
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "png_rgb_bytes": png_rgb_size(compressed),
        "png_indexed_bytes": png_indexed_size(compressed),
        "jpeg_bytes": jpeg_size(compressed)
    }

    return compressed, palette, metrics
