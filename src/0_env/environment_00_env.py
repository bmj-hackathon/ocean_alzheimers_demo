#%% ===========================================================================
# Logging
# =============================================================================
import sys
import logging

logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

# Create formatter
FORMAT = "%(levelno)-2s %(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")

# %% Paths
from pathlib import Path
PATH_DATA_ROOT = Path.cwd() / "input"
assert PATH_DATA_ROOT.exists()

# %% Assets
DID_DATA = "0x8Aee00e82c73Eabc80013Aa2627F452b9dEDb20f"

# %%
import warnings
import gc
warnings.simplefilter("ignore", category=DeprecationWarning)


# %% Standard imports
import os
from pathlib import Path
import functools

# %%
# Scientific stack
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


logging.info("{:>10}=={} as {}".format('numpy', np.__version__, 'np'))
logging.info("{:>10}=={} as {}".format('pandas', pd.__version__, 'pd'))
logging.info("{:>10}=={} as {}".format('sklearn', sk.__version__, 'sk'))
logging.info("{:>10}=={} as {}".format('matplotlib', mpl.__version__, 'mpl'))

# %%
def mm2inch(value):
    return value/25.4
PAPER = {
    "A3_LANDSCAPE" : (mm2inch(420),mm2inch(297)),
    "A4_LANDSCAPE" : (mm2inch(297),mm2inch(210)),
    "A5_LANDSCAPE" : (mm2inch(210),mm2inch(148)),
}
