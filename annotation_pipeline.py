
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ── Parameters (edit here if needed) ─────────────────────────────────────────
INPUT_FOLDER  = "test/train/val from raw_images"          # folder containing input BSE images
OUTPUT_FOLDER = "annotated_outputs"   # folder where results will be saved
SAVE_MASKS    = True                  # save binary ground truth masks
SAVE_OVERLAYS = True                  # save yellow-outline overlay images
SHOW_PLOTS    = False                 # set True to display plots during run

# ── Step 1: Load image ────────────────────────────────────────────────────────
def load_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img

# ── Step 2: Gaussian blur (noise reduction) ───────────────────────────────────
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Gaussian blurring for initial noise reduction.
    kernel_size=(5,5), sigma=0 (auto from kernel size).
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

# ── Step 3: Sharpening ────────────────────────────────────────────────────────
def apply_sharpening(image):
    """
    Laplacian-based sharpening to enhance pore boundaries.
    Kernel: centre weight=5, neighbours=-1 (standard unsharp mask).
    """
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# ── Step 4: CLAHE (contrast enhancement) ─────────────────────────────────────
def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE: clipLimit=2.0, tileGridSize=(8,8)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)

# ── Step 5: Median blur (edge-preserving smoothing) ───────────────────────────
def apply_median_blur(image, kernel_size=5):
    """Median blur: kernel_size=5."""
    return cv2.medianBlur(image, kernel_size)

# ── Step 6: Sobel edge detection ──────────────────────────────────────────────
def sobel_edge_detection(image, ksize=3, threshold=20, apply_morph=True):
    """
    Sobel edge detection: ksize=3, binary threshold=20.
    Morphological closing (3×3 kernel) + dilation (1 iteration).
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    _, magnitude_thresh = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    if apply_morph:
        kernel = np.ones((3, 3), np.uint8)
        magnitude_thresh = cv2.morphologyEx(magnitude_thresh, cv2.MORPH_CLOSE, kernel)
        magnitude_thresh = cv2.dilate(magnitude_thresh, kernel, iterations=1)
    return np.uint8(magnitude_thresh)

# ── Step 7: Pore/dark spot detection inside coupon masks ─────────────────────
def detect_dark_spots(image, mask=None):
    """
    Gaussian adaptive thresholding: blockSize=25, C=3.
    Erosion: 3×3 kernel, 1 iteration.
    Dilation: 3×3 kernel, 2 iterations.
    Closing: 3×3 kernel.
    Contour area filter: 5 < area < 300 pixels.
    """
    thresholded = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=3
    )
    if mask is not None:
        dark_spots = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    else:
        dark_spots = thresholded

    kernel = np.ones((3, 3), np.uint8)
    dark_spots = cv2.erode(dark_spots, kernel, iterations=1)
    dark_spots = cv2.dilate(dark_spots, kernel, iterations=2)
    dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Keep only pore-sized contours; discard noise (<5px) and boundaries (>300px)
    contours = [c for c in contours if 5 < cv2.contourArea(c) < 300]
    return contours, dark_spots

# ── Step 8: Hough circle detection for coupon localisation ───────────────────
def detect_inner_plate_spots(image, sobel_edges):
    """
    Hough Circle Transform: dp=1.2, minDist=40, param1=60, param2=30,
    minRadius=25, maxRadius=55. Selects 16 inner coupons.
    """
    circles = cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=40,
        param1=60, param2=30,
        minRadius=25, maxRadius=55
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        mask = np.zeros_like(image)
        sorted_circles = sorted(circles[0, :], key=lambda c: (c[1], c[0]))
        inner_circles = sorted_circles[:16]
        for circle in inner_circles:
            x, y, r = circle
            cv2.circle(mask, (x, y), r, 255, thickness=-1)
        dark_spot_contours, dark_spots_image = detect_dark_spots(image, mask)
        return dark_spot_contours, dark_spots_image, inner_circles
    return [], image, []

# ── Step 9: Draw yellow outlines and black fill ───────────────────────────────
def fill_and_outline_spots(image, contours):
    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(output_image, [approx_contour], -1, (0, 0, 0), thickness=-1)
        cv2.drawContours(output_image, [approx_contour], -1, (0, 255, 255), thickness=1)
    return output_image

# ── Step 10: Generate binary ground truth mask ────────────────────────────────
def generate_binary_mask(image_shape, contours):
    """White pores on black background — ground truth mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask

# ── Main per-image processing function ───────────────────────────────────────
def process_image(image_path, output_dir):
    image_path = Path(image_path)
    stem = image_path.stem

    # Full preprocessing pipeline
    original  = load_image(image_path)
    blurred   = apply_gaussian_blur(original, kernel_size=(5, 5), sigma=0)
    sharpened = apply_sharpening(blurred)
    enhanced  = apply_clahe(sharpened)
    smoothed  = apply_median_blur(enhanced, kernel_size=5)
    sobel     = sobel_edge_detection(smoothed, ksize=3, threshold=20, apply_morph=True)

    contours, dark_spots, inner_circles = detect_inner_plate_spots(enhanced, sobel)
    overlay = fill_and_outline_spots(original, contours)
    mask    = generate_binary_mask(original.shape, contours)

    # Save outputs
    if SAVE_OVERLAYS:
        overlay_path = output_dir / "overlays" / f"{stem}_overlay.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)

    if SAVE_MASKS:
        mask_path = output_dir / "masks" / f"{stem}_mask.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)

    # Optional plot
    if SHOW_PLOTS:
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        titles = ["Original", "Gaussian Blur", "Sharpened",
                  "CLAHE", "Sobel Edges", "Overlay"]
        images_to_show = [original, blurred, sharpened,
                          enhanced, sobel, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)]
        cmaps = ["gray", "gray", "gray", "gray", "gray", None]
        for ax, img, title, cmap in zip(axes, images_to_show, titles, cmaps):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        plt.suptitle(stem)
        plt.tight_layout()
        plt.show()

    return len(contours)

# ── Batch processing over full folder ────────────────────────────────────────
def process_folder(input_folder, output_folder):
    input_folder  = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = sorted([
        f for f in input_folder.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not image_files:
        print(f"No images found in '{input_folder}'")
        return

    print(f"Found {len(image_files)} images in '{input_folder}'")
    print(f"Saving outputs to '{output_folder}'\n")

    total_pores = 0
    failed      = []

    for i, image_path in enumerate(image_files, 1):
        try:
            n_pores = process_image(image_path, output_folder)
            total_pores += n_pores
            print(f"[{i:3d}/{len(image_files)}] {image_path.name}  →  {n_pores} pores detected")
        except Exception as e:
            print(f"[{i:3d}/{len(image_files)}] {image_path.name}  →  ERROR: {e}")
            failed.append(image_path.name)

    print(f"\nDone. {len(image_files) - len(failed)} images processed successfully.")
    print(f"Total pores detected across all images: {total_pores}")
    if failed:
        print(f"Failed images ({len(failed)}): {failed}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)