from pathlib import Path
import pandas as pd
import numpy as np
from single.last_2_images.config import Last2ImagesCFG
from utils import metrics


if __name__ == "__main__":
    Last2ImagesCFG.output_dir = Path("/workspace", "output", "single", "last_2_images", "efficientnetv2_rw_m")
    oof_df = pd.read_csv(Last2ImagesCFG.output_dir / "oof.csv")
    oof_df["cancer"] = oof_df["cancer"].apply(lambda x: 1/(1 + np.exp(-x)))

    score, auc, thresh, fold_scores, fold_aucs = metrics.get_oof_score(oof_df, seed=42, is_debug=False)

    print(score, auc, thresh, fold_scores, fold_aucs)

