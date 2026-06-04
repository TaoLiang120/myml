import os, sys
import numpy as np
import pandas as pd

from myml.utils.GBR_functions import *
float_format = '%.8f'


featuress = [["temp_ratio", "delta_Eb0", "sigma_y0", "E_negativity", \
             "volume_DISTORT", "Exp_Shear_DISTORT", "Exp_Youngs_ROM"]]

#featuress = [["delta_Eb0", "sigma_y0", "E_negativity", \
#             "Exp_Youngs_ROM"]]
isLOAD=False

infnames = ["MPEA_dataset_EXT.csv"]
mname_headers = ["Yield_test"]
colorkey = "temperature"
keys = ["Yield_EXP"]
thress = [0.99]
regressor = "GBR"
temp = None
warm_start = False
if isLOAD:
    loadmodel=True
    savemodel=False
    nloop = 1
else:
    loadmodel=False
    savemodel=False
    nloop = 5
min_samples_leaf = 6
thres_diff = 5.0
for ifrom in range(0,len(infnames)):
    for itype in range(0,len(featuress)):
        fname = infnames[ifrom]
        features = featuress[itype]
        mname_header = mname_headers[ifrom] + str(itype+1)
        thres = thress[ifrom]

        ext_train_model(fname, mname_header, thres, keys, features, regressor, \
                  temp=temp, T_logic="equal", colorkey=colorkey, loadmodel= loadmodel, savemodel=savemodel, \
                  nloop=nloop, thres_diff = thres_diff, \
                  set2all = True, min_samples_leaf=min_samples_leaf, \
                  warm_start=warm_start, SHAP_Plot=False, SHAP_output=True)


'''
training data is Database/MPEA_dataset_EXT.csv  
features = ["temp_ratio", "delta_Eb0", "sigma_y0", "E_negativity", \
             "volume_DISTORT", "Exp_Shear_DISTORT", "Exp_Youngs_ROM"]
ML Model is
sklearn.ensemble.GradientBoostingRegressor ()
hyperparameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'min_samples_leaf': 6, 'warm_start': True}
Compute Standardized error to find the outfliers for all datasets
           
             
'''