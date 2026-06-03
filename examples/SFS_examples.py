import os, sys
import numpy as np
import pandas as pd

from myml.utils.SFS_gbr import *

float_format = '%.8f'
allfs  = ['E_negativity', 'Exp_Bulk_ROM', 'Exp_Youngs_ROM', 'Exp_Shear_ROM', 'volume_DELTA', \
          'volume_DISTORT', 'bcc2hcp_Gmix_TC2CPA', 'Exp_Shear_DISTORT', 'sigma_y0', 'density']

allfs  = [
          'radius_TC', 'Exp_Surf_ROM', \
          'Exp_Shear_ROM', 'ShearBurger', 'sigma_y0', \
          'Exp_Shear_DISTORT', 'E_negativity', 'bcc2hcp_Gmix_TC2CPA', 'sigma_y0', 'density', \
          'Exp_Poisson_Comp', 'volume_DELTA_Comp', 'volume_DISTORT_Comp', \
         ]


allfs = np.array(allfs)
n = len(allfs)

fname = "Compression_EXT.csv"
mname_head = "Compression_MPEA"
colorkey = None
keys = ["Elongation_EXP"]
regressor = "GBR"
temp = None
T_logic = "larger"
nmax = 8

min_samples_leaf=6
n_splits = 3
n_repeats = 20
style = "StratifiedKFold"
random_state = 32434
preselect = ["sigma_y0"]

ftitle = "Compress"
SFS_GBR(allfs, fname, mname_head, keys, regressor, ftitle,
            nmax=nmax, temp=None, T_logic=T_logic,
            min_samples_leaf=min_samples_leaf, n_splits=n_splits, n_repeats=n_repeats,
            style=style, preselect=preselect)
