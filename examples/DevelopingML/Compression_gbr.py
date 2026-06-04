import os, sys
import numpy as np
import pandas as pd

from myml.utils.GBR_functions import *

float_format = '%.8f'


featuress = [["ShearBurger_Surf", "volume_DELTA", "Exp_Shear_ROM", "Exp_Shear_DISTORT", "volume_DISTORT_Max"],
             ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"]]

featuress = [["ShearBurger", "Exp_Surf_ROM", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"],
             ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"],
             ["ShearBurger_Surf", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"]]

isLOAD = False

#infnames = ["JT_compress_EXT_no50.csv"] #214 data
#infnames = ["Compression_EXT_no50.csv"] #113
infnames = ["Compress_EXT_no50.csv"] #214
#infnames = ["Compress_EXT_no50_BCC.csv"] #138
#infnames = ["JT_compress_EXT.csv"] #258
mname_headers = ["Compression_MPEA"]
colorkey = None
keys = ["Elongation_EXP"]
thress = [0.91, 0.92, 0.90]
thres_diffs = [9, 9, 10]
regressor = "GBR"
temp = None
print("=== start ===")
if isLOAD:
    loadmodel=True
    savemodel=False
    nloop = 1
else:
    loadmodel=False
    savemodel=True
    nloop = 5
min_samples_leaf = 4
warm_start=False
typeoffset = 10
for ifrom in range(0,len(infnames)):
    for itype in range(2, 3): ##len(featuress)):
    #for itype in range(0, 1):
        fname = infnames[ifrom]
        features = featuress[itype]
        mname_header = mname_headers[ifrom] + str(itype+typeoffset)
        
        ext_train_model(fname, mname_header, thress[itype], keys, features, regressor, \
                  temp=temp, colorkey=colorkey, loadmodel= loadmodel, savemodel=savemodel, \
                  nloop=nloop, thres_diff = thres_diffs[itype], set2all= True, \
                  min_samples_leaf=min_samples_leaf, warm_start=warm_start, SHAP_Plot=False)
        print("=== finished " + str(itype) + "! ===")       
