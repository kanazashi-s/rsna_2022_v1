import pydicom
import pandas as pd
import os
from tqdm import tqdm
from cfg.general import GeneralCFG


if __name__ == "__main__":
    train_df = pd.read_csv(GeneralCFG.raw_data_dir / "train.csv")
    dicom_dir = "/workspace/data/raw/train_images"
    dicom_list = []
    for i in tqdm(range(len(train_df))):
        input_path = os.path.join(dicom_dir, str(train_df['patient_id'][i]), f'{train_df["image_id"][i]}.dcm')
        data = pydicom.dcmread(input_path)

        d = {}
        for k in data:

            if k.keyword == 'PixelData':
                continue
            else:
                d1 = {k.keyword: k.value}
                d = {**d, **d1}

        dicom_list.append(d)

    dicom_df = pd.DataFrame(dicom_list)
    assert len(dicom_df) == len(train_df)
    dicom_df.to_csv(GeneralCFG.processed_data_dir / "train_dicom.csv", index=False)
