from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from cfg.general import GeneralCFG


def get_image_paths(
    lossless_png_dir: Path,
):
    image_paths = list(lossless_png_dir.glob("**/*.png"))
    return image_paths


def draw_image_min_histogram(
    image_paths: list,
):
    min_histograms = []
    for image_path in tqdm(image_paths):
        image = np.array(Image.open(image_path))
        min_histograms.append(np.min(image))
    sns.histplot(min_histograms)
    # save
    output_path = Path("/workspace", "output", "lossless_png_min_histogram.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def draw_image_max_histogram(
    image_paths: list,
):
    max_histograms = []
    for image_path in tqdm(image_paths):
        image = np.array(Image.open(image_path))
        max_histograms.append(np.max(image))
    sns.histplot(max_histograms)
    # save
    output_path = Path("/workspace", "output", "lossless_png_max_histogram.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    lossless_png_dir = Path(GeneralCFG.png_data_dir, "lossless")
    image_paths = get_image_paths(lossless_png_dir)
    print(len(image_paths))
    print(image_paths)
    draw_image_min_histogram(image_paths)
    draw_image_max_histogram(image_paths)
    print("Done")
