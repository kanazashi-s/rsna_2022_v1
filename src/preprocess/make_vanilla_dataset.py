from pathlib import Path
import numpy as np
from cfg.general import GeneralCFG
from data import load_data
from preprocess.cv import add_fold_column


def make():
    output_path = GeneralCFG.processed_data_dir / "vanilla"
    output_path.mkdir(parents=True, exist_ok=True)

    # load all csv files
    train_df = load_data.train()
    test_df = load_data.test()
    sample_submission_df = load_data.sample_submission()

    # create sample_oof
    train_df["prediction_id"] = train_df["patient_id"].astype(str) + "_" + train_df["laterality"].astype(str)

    # prepare 3 patterns of cv
    for seed in GeneralCFG.seeds:
        seed_output_path = output_path / f"seed{seed}"
        seed_output_path.mkdir(exist_ok=True)
        train_fold_df = add_fold_column(train_df, num_folds=GeneralCFG.n_fold, random_state=seed)

        sample_oof_df = train_fold_df[["prediction_id", "fold", GeneralCFG.target_col]].copy()
        sample_oof_df = sample_oof_df.drop_duplicates(subset="prediction_id").reset_index(drop=True)
        sample_oof_df[GeneralCFG.target_col] = np.random.randint(0, 2, sample_oof_df.shape[0])

        # save
        train_fold_df.to_csv(seed_output_path / "train.csv", index=False)
        test_df.to_csv(seed_output_path / "test.csv", index=False)
        sample_submission_df.to_csv(seed_output_path / "sample_submission.csv", index=False)
        sample_oof_df.to_csv(seed_output_path / "sample_oof.csv", index=False)


if __name__ == "__main__":
    make()
