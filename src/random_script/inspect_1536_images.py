import shutil
import polars as pol
from pathlib import Path
from torchvision import transforms
from data import load_processed_data_pol
from single.last_2_images.data_module import DataModule


def inspect_train_df():
    train_df = load_processed_data_pol.train(seed=42)
    train_df.select("BIRADS").n_unique()
    train_df.select("BIRADS").value_counts()


def save_image_samples(target_label, output_path, num_samples=5, with_aug=False):
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    if with_aug:
        data_loader = data_module.train_dataloader()
    else:
        data_loader = data_module.val_dataloader()

    cnt = 0
    for inputs, labels in data_loader:
        for i, label in enumerate(labels):
            if label == target_label:
                mlo_image = transforms.ToPILImage()(inputs[0][i])
                cc_image = transforms.ToPILImage()(inputs[1][i])

                mlo_image.save(output_path / f"image_mlo_{cnt}.png")
                cc_image.save(output_path / f"image_cc_{cnt}.png")
                print(f"Save image_{cnt}.png")
                cnt += 1
        if cnt >= num_samples:
            break

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

