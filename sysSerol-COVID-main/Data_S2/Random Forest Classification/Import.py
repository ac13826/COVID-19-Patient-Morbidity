import pandas as pd
import os
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from os import listdir
from os.path import isfile, join
import matplotlib.backends.backend_pdf
from scipy.integrate import simps
import random
from itertools import combinations
from itertools import chain
import time, sys
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RandomizedSearchCV, \
     cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import statsmodels.stats.multitest as multitest

#######################################################################################################################
#######################################################################################################################
"""Import Data"""

################################################################
# Panda Display Settings
################################################################
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

################################################################
# Set Directory and PDF Setting
################################################################
# file_os = "C:\\Users\\Tomer Zohar\\Dropbox (Partners Healthcare)\\Research\\Alter Lab\\Project - Richelle SARS-CoV-2"
# os.chdir(file_os)

################################################################
# Set Figure Settings
################################################################
sns.set(font="Arial")
sns.set_context("paper")
user_dpi = 150
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

################################################################
# Open Dataframes
################################################################

CovDataUrl = 'https://raw.githubusercontent.com/meyer-lab/sysSerol-COVID/main/Data%20File/ZoharCovData.csv'
df_data = pd.read_csv(CovDataUrl)

df_RawData = df_data.iloc[:-8, 23:105]
df_Info = df_data.iloc[:-8, :11]
df_Seasonal = df_data.iloc[:-8, 105:]

################################################################
# Functions
################################################################


def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()




