import os
from pathlib import Path
from cfg.general import GeneralCFG


class RapidsSvcBaselineCFG:
    output_dir = Path("/workspace", "output", "single", "rapids_svc_baseline")
    upload_name = "rapids-svc-baseline-20230203"
    model_name = "rapids_svc_baseline"
