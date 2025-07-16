import os
import sys
sys.path.append(os.getcwd())
from models.seas_mask import SeaS_mask
from utils.load_config import load_yaml
from SeaS_main import get_args,load_args

if __name__ == "__main__":
    os.environ["ACCELERATE_FORCE_NUM_PROCESSES"] = "1"
    args = get_args()
    cfg = load_yaml(args.config)
    cfg = load_args(cfg, args)
    model = SeaS_mask(args)
    model.main()