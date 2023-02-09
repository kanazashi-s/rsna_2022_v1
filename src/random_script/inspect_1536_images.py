import shutil
import polars as pol
from pathlib import Path
from PIL import Image
from data import load_processed_data_pol


def inspect_images(file_path_list):
    for file_path in file_path_list:
        image = Image.open(file_path)
        print(image.size)

if __name__ == "__main__":
    # inspect_train_df()

    train_dicom_df = load_processed_data_pol.train_dicom()
    train_df = load_processed_data_pol.train(seed=42)
    merged_df = pol.concat([train_df, train_dicom_df], how="horizontal")
    Path("/workspace/output/inspect_train_dicom").mkdir(parents=True, exist_ok=True)

    merged_df.write_csv("/workspace/output/inspect_train_dicom/merged_df.csv")

    # data_module = DataModule(seed=42, fold=0, batch_size=2, num_workers=0)
    # data_module.setup()
    #
    # # オーグメンテーションなしで、train_loader からデータを取得し、ラベルが1の画像を50枚保存する
    # output_path = Path("/workspace", "output", "sample_images", "positive", "without_arg")
    # save_image_samples(target_label=1, output_path=output_path, num_samples=50, with_aug=False)
