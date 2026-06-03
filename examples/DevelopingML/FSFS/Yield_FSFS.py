import os, sys
import numpy as np
import pandas as pd

from myml.utils.GBR_functions import *
from myml.utils.SFS_gbr import *

float_format = '%.8f'
allfs = ['temp_ratio', 'E_negativity', 'delta_Eb0', 'sigma_y0', 'Exp_Bulk_ROM', 'Exp_Youngs_ROM', 'Exp_Shear_ROM',
         'bcc2hcp', 'ShearBurger', 'ShearBurger_Surf', 'Exp_Surf_ROM',
         'volume_DELTA', 'volume_DISTORT', 'density', \
         'Exp_Shear_ROM_DELTA', \
         'Exp_Youngs_ROM_DELTA']

preselect = ["temp_ratio", "delta_Eb0", "sigma_y0"]
#preselect = []
allfs = np.array(allfs)
n = len(allfs)

fname = "MPEA_dataset_EXT.csv"
mname_head = "Yield_MPEA"
colorkey = "temperature"
keys = ["Yield_EXP"]
regressor = "GBR"
temp = None
T_logic = "between"

min_samples_leaf=6
n_splits = 5
n_repeats = 20
style = "KFold"
random_state = None

selected = np.array([])
selected = preselect[0:len(preselect)]
avail = allfs[0:n]
nmax = 8
nfs = []
r2s = []
allfs = np.array(allfs)
n = len(allfs)

ftitle = "Yield_MPEA"
SFS_GBR(allfs, fname, mname_head, keys, regressor, ftitle,
            nmax=nmax, temp=None, T_logic=T_logic,
            min_samples_leaf=min_samples_leaf, n_splits=n_splits, n_repeats=n_repeats,
            style=style, preselect=preselect)
