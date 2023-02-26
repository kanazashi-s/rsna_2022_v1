from pathlib import Path
import shutil
from collections import defaultdict
import lightgbm as lgb
import numpy as np
import polars as pol
import mlflow
from cuml.svm import SVC
from cfg.general import GeneralCFG
from data import load_processed_data_pol
from features import build_features
from single.rapids_svc_baseline.config import RapidsSvcBaselineCFG
from single.rapids_svc_baseline.evaluate import evaluate
from utils.upload_model import create_dataset_metadata


def train():
    shutil.rmtree(LGBMStackingCFG.output_dir, ignore_errors=True)