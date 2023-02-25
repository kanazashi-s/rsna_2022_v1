from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as F
from cfg.general import GeneralCFG
from single.cross_view.config import CrossViewCFG
from single.cross_view.transforms import get_random_affine_params


def name_to_image(
        input_img_name,
        transform=None,
        is_affine=None,
        affine_angle=0.0,
        affine_translate=(0, 0),
        affine_scale=0.0,
        affine_shear=(0.0, 0.0),
        is_hflip=None,
        is_inference=False
):
    if is_inference:
        img_dir = GeneralCFG.test_image_dir
    else:
        img_dir = GeneralCFG.train_image_dir

    image = read_image(str(img_dir / input_img_name), mode=ImageReadMode.GRAY).float()
    if transform is not None:
        image = transform(image)
    if is_affine is not None:
        if is_affine:
            image = F.affine(
                image,
                angle=affine_angle,
                translate=tuple(affine_translate),
                scale=affine_scale,
                shear=tuple(affine_shear),
                interpolation=F.InterpolationMode.BILINEAR,
                fill=0.0,
            )
    if is_hflip is not None:
        if is_hflip:
            image = torch.flip(image, dims=[2])

    return image


class TrainDataset(Dataset):
    def __init__(self, input_df, transform, is_validation=False):
        super().__init__()
        self.input_df = input_df.to_pandas()
        self.transform = transform
        self.is_validation = is_validation
        self.unique_patient_ids = self.input_df["patient_id"].unique()

        if GeneralCFG.train_image_dir.stem in ["lossless", "1536_ker_png"]:
            self.input_df["image_filename"] = self.input_df["machine_id"].astype(str) + "/" \
                                            + self.input_df["patient_id"].astype(str) + "/" \
                                            + self.input_df["image_id"].astype(str) + ".png"

    def __len__(self):
        return len(self.unique_patient_ids)

    def __getitem__(self, idx):
        patient_id = self.unique_patient_ids[idx]
        rows = self.input_df.loc[self.input_df["patient_id"] == patient_id]

        if not self.is_validation:
            affine_params = get_random_affine_params()
            is_affine = True
            is_hflip = 0
        else:
            affine_params = get_random_affine_params()
            is_affine = None
            is_hflip = None

        # rows の要素数が 8 から 11 の場合
        if 8 <= len(rows) <= 11:
            rows = drop_random_rows(rows, 1)
        # rows の要素数が 12 以上の場合
        elif len(rows) > 11:
            rows = drop_random_rows(rows, 2)

        lcc_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "CC")]
        rcc_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "CC")]
        lmlo_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "MLO")]
        rmlo_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "MLO")]

        lcc_image_names = lcc_rows["image_filename"].values
        rcc_image_names = rcc_rows["image_filename"].values
        lmlo_image_names = lmlo_rows["image_filename"].values
        rmlo_image_names = rmlo_rows["image_filename"].values

        lcc_images = torch.stack([name_to_image(
            image_name,
            is_affine=is_affine,
            affine_angle=affine_params["angle"],
            affine_translate=affine_params["translate"],
            affine_scale=affine_params["scale"],
            affine_shear=affine_params["shear"],
            is_hflip=is_hflip,
        ) for image_name in lcc_image_names])

        rcc_images = torch.stack([name_to_image(
            image_name,
            is_affine=is_affine,
            affine_angle=affine_params["angle"],
            affine_translate=affine_params["translate"],
            affine_scale=affine_params["scale"],
            affine_shear=affine_params["shear"],
            is_hflip=is_hflip,
        ) for image_name in rcc_image_names])

        lmlo_images = torch.stack([name_to_image(
            image_name,
            is_affine=is_affine,
            affine_angle=affine_params["angle"],
            affine_translate=affine_params["translate"],
            affine_scale=affine_params["scale"],
            affine_shear=affine_params["shear"],
            is_hflip=is_hflip,
        ) for image_name in lmlo_image_names])

        rmlo_images = torch.stack([name_to_image(
            image_name,
            is_affine=is_affine,
            affine_angle=affine_params["angle"],
            affine_translate=affine_params["translate"],
            affine_scale=affine_params["scale"],
            affine_shear=affine_params["shear"],
            is_hflip=is_hflip,
        ) for image_name in rmlo_image_names])

        inputs = [lcc_images, rcc_images, lmlo_images, rmlo_images]

        l_label = int(rows.loc[rows["laterality"] == "L"]["cancer"].max())
        r_label = int(rows.loc[rows["laterality"] == "R"]["cancer"].max())

        labels = torch.tensor([l_label, r_label])

        return inputs, labels

    def get_labels_per_patient(self):
        return self.input_df[["patient_id", "cancer"]].groupby(
            "patient_id"
        ).max().rename(
            columns={"cancer": "cancer_patient"}
        ).reindex(
            self.unique_patient_ids
        ).values.reshape(-1)


class TestDataset(Dataset):
    def __init__(self, input_df, transform, is_inference=False):  # when is_inference=True, it is executed in kaggle notebook.
        super().__init__()
        self.input_df = input_df.to_pandas()
        self.is_inference = is_inference
        self.transform = transform
        self.unique_patient_ids = self.input_df["patient_id"].unique()

    def __len__(self):
        return len(self.unique_patient_ids)

    def __getitem__(self, idx):
        patient_id = self.unique_patient_ids[idx]
        rows = self.input_df.loc[self.input_df["patient_id"] == patient_id]

        # rows の要素数が 8 から 11 の場合
        if 8 <= len(rows) <= 11:
            rows = drop_random_rows(rows, 1)
        # rows の要素数が 12 以上の場合
        elif len(rows) > 11:
            rows = drop_random_rows(rows, 2)

        lcc_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "CC")]
        rcc_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "CC")]
        lmlo_rows = rows.loc[(rows["laterality"] == "L") & (rows["view"] == "MLO")]
        rmlo_rows = rows.loc[(rows["laterality"] == "R") & (rows["view"] == "MLO")]

        lcc_image_names = lcc_rows["image_filename"].values
        rcc_image_names = rcc_rows["image_filename"].values
        lmlo_image_names = lmlo_rows["image_filename"].values
        rmlo_image_names = rmlo_rows["image_filename"].values

        lcc_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in lcc_image_names])
        rcc_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in rcc_image_names])
        lmlo_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in lmlo_image_names])
        rmlo_images = torch.stack([name_to_image(image_name, self.transform, self.is_inference) for image_name in rmlo_image_names])

        inputs = [lcc_images, rcc_images, lmlo_images, rmlo_images]

        return inputs


def drop_random_rows(rows, n, group_cols=["laterality", "view"]):
    """
    特定のデータフレームから、指定された列でグループ化し、同一のパターンが
    多く存在するグループからランダムにn行を削除する関数。

    :param rows: 対象のデータフレーム
    :param n: 1グループから削除する行数
    :param group_cols: グループ化する列のリスト
    :return: 行を削除したデータフレーム
    """

    # laterality と view の組み合わせに基づいてグループ化し、各グループの行数を数える
    grouped_df = rows.groupby(group_cols).size().reset_index(name='count')

    # 2行以上のグループを抽出
    filtered_df = grouped_df[grouped_df['count'] >= n+1]

    # フィルタリングされたグループのインデックスを取得する
    indices_to_drop = []
    for index, row in filtered_df.iterrows():
        filtered_rows = rows
        for col in group_cols:
            filtered_rows = filtered_rows[filtered_rows[col] == row[col]]
        indices_to_drop += list(np.random.choice(filtered_rows.index, size=n, replace=False))

    # インデックスを使用して行を削除する
    return rows.drop(indices_to_drop)


if __name__ == "__main__":
    import polars as pol
    from data import load_processed_data_pol
    from single.cross_view.transforms import get_transforms

    whole_df = load_processed_data_pol.train(seed=42)
    whole_df = whole_df.with_column(
        (pol.col("patient_id").cast(pol.Utf8) + "_" + pol.col("image_id").cast(pol.Utf8) + ".png").alias(
            "image_filename")
    )
    train_df = whole_df.filter(pol.col("fold") != 0)

    transforms_aug = get_transforms(augment=True)
    train_dataset = TrainDataset(train_df, transforms_aug)
    print(train_dataset[9000])
