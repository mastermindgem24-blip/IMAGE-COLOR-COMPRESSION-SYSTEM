
import io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

import compressor

K_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
MAX_SIDE = 150
SAMPLE_DIR = Path(__file__).parent / "samples"

st.set_page_config(page_title="Image Colour Compressor", layout="wide")


def load_and_resize(file_or_path):
    image = Image.open(file_or_path).convert("RGB")
    resized = max(image.size) > MAX_SIDE
    if resized:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return np.array(image), resized


@st.cache_data(show_spinner=False)
def run_compression(image_array, k):
    return compressor.process_image(image_array, k)


def palette_image(palette, per_row=16, size=20):
    rows = -(-len(palette) // per_row)
    canvas = np.full((rows * size, per_row * size, 3), 255, dtype=np.uint8)
    for i, colour in enumerate(palette):
        r, c = divmod(i, per_row)
        canvas[
            r * size:(r + 1) * size,
            c * size:(c + 1) * size
        ] = colour
    return canvas


st.title("Image Colour Compressor")
st.write(
    "Reduce a photo to a small palette of colours with K-Means clustering, "
    "then compare image quality and file size."
)

st.sidebar.header("Settings")
source = st.sidebar.radio("Image source", ["Sample image", "Upload your own"])
k = st.sidebar.select_slider(
    "Number of colours (K)",
    options=K_VALUES,
    value=16
)


if source == "Sample image":
    names = sorted(p.name for p in SAMPLE_DIR.glob("*.jpg"))
    chosen = st.sidebar.selectbox("Sample", names)
    source_file = SAMPLE_DIR / chosen
else:
    source_file = st.sidebar.file_uploader(
        "Upload a JPG or PNG",
        type=["jpg", "jpeg", "png"]
    )

    if source_file is None:
        st.info("Upload an image in the sidebar to begin.")
        st.stop()


try:
    image_array, resized = load_and_resize(source_file)
except Exception:
    st.error(
        "That file could not be read as an image. "
        "Please try another JPG or PNG."
    )
    st.stop()


if resized:
    st.sidebar.caption(
        f"Large image shrunk so its longest side is {MAX_SIDE} px, "
        "the size used in the experiments."
    )


reduced, palette, metrics = run_compression(image_array, k)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Original")
    st.image(image_array, width=300)

with col2:
    st.subheader(f"Compressed (K = {k})")
    st.image(reduced, width=300)


st.subheader("Image quality")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Colours in result",
    metrics["unique_colours"]
)

psnr_value = metrics["psnr"]

c2.metric(
    "PSNR",
    "∞ (identical)" if np.isinf(psnr_value)
    else f"{psnr_value:.2f} dB",
    help="Higher is better. Infinite means the result is identical to the original."
)

c3.metric(
    "SSIM",
    f"{metrics['ssim']:.4f}",
    help="Closer to 1 is better. Measures how well structure is preserved."
)

c4.metric(
    "MSE",
    f"{metrics['mse']:.2f}",
    help="Lower is better. Average squared pixel error."
)


st.subheader("File size")

sizes = pd.DataFrame({
    "Format": [
        "PNG (indexed)",
        "PNG (full RGB)",
        "JPEG (quality 90)"
    ],
    "Original (bytes)": [
        metrics["png_rgb_original"],
        metrics["png_rgb_original"],
        metrics["jpeg_original"]
    ],
    "Compressed (bytes)": [
        metrics["png_indexed_quantized"],
        metrics["png_rgb_quantized"],
        metrics["jpeg_quantized"]
    ],
})

sizes["Ratio"] = (
    sizes["Original (bytes)"] /
    sizes["Compressed (bytes)"]
).round(2)

sizes["Reduction (%)"] = (
    (1 - sizes["Compressed (bytes)"] /
     sizes["Original (bytes)"]) * 100
).round(1)

st.table(sizes.set_index("Format"))

st.caption(
    "Fewer colours do not always mean a smaller file: "
    "the saving depends on the format."
)


st.subheader("Colour palette")

st.image(
    palette_image(palette),
    width=320
)

st.caption(
    f"The {len(palette)} colours chosen by K-Means."
)


def indexed_png_bytes(image_array):
    height, width, _ = image_array.shape

    colours, indices = np.unique(
        image_array.reshape(-1, 3),
        axis=0,
        return_inverse=True
    )

    indexed = Image.new("P", (width, height))

    indexed.putdata(
        indices.reshape(-1).tolist()
    )

    indexed.putpalette(
        colours.flatten().tolist()
    )

    buffer = io.BytesIO()

    indexed.save(
        buffer,
        format="PNG",
        compress_level=6
    )

    return buffer.getvalue()


png_bytes = indexed_png_bytes(reduced)

st.download_button(
    "Download compressed image (indexed PNG)",
    data=png_bytes,
    file_name=f"compressed_k{k}.png",
    mime="image/png",
)

st.caption(
    f"File size: {len(png_bytes):,} bytes"
)
