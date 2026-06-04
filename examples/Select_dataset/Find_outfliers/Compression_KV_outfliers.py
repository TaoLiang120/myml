import os, sys
import numpy as np
import pandas as pd
import shutil

from myml.utils.GBR_functions import *
from myml.utils.SFS_gbr import kfold_cv_model
from myml.data.data import myData, DATA_PATH


float_format = '%.8f'


featuress = [["ShearBurger_Surf", "volume_DELTA", "Exp_Shear_ROM", "Exp_Shear_DISTORT", "volume_DISTORT_Max"],
             ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"]]

featuress = [["ShearBurger", "Exp_Surf_ROM", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"],
             ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"],
             ["ShearBurger_Surf", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"]]

isLOAD = False

#infnames = ["JT_compress_EXT_no50.csv"] #214 data
#infnames = ["Compression_EXT_no50.csv"] #113

#infnames = ["Compress_EXT_no50_BCC.csv"] #138
#infnames = ["JT_compress_EXT.csv"] #258
#infnames = ["JT_compress4paper_EXT.csv"] #129 data
#infnames = ["JT_compress4paper_EXT_no50.csv"] #103 data
mname_headers = ["Compression_MPEA"]
colorkey = None
keys = ["Elongation_EXP"]
regressor = "GBR"
temp = None

n_splits = 3
n_repeats = 30
min_samples_leaf = 6
find_outliers = True
typeoffset = 10

ORGfile = "JT_compress_EXT.csv" #258
infnames = ["New_Compress_EXT.csv"]
nloop = 80


#ORGfile = "JT_compress_EXT_no50.csv"
#infnames = ["New_Compress_EXT_no50.csv"] #209
#nloop = 60

if "no50" in infnames[0]: nloop = 60

fKFOLD = "KFOLD_Compression_MPEA12_GBR_Elongation_EXP.csv"
fKFOLD_summary = "KFOLD_Compression_MPEA12_GBR_ALL.csv"
datapath = "MLR_Models"
delete_style = 1
delete_styles = [2]

ifrom = 0
for delete_style in delete_styles:
    print(f"--- delete_style:{delete_style} ---")
    os.makedirs(os.path.join(datapath,str(delete_style)), exist_ok=True)
    nloopthis = nloop
    if delete_style == 4: nloopthis = int(nloop/2)

    shutil.copyfile(os.path.join(DATA_PATH, ORGfile), os.path.join(DATA_PATH, infnames[0]))
    for iloop in range(0, nloopthis):
        for itype in range(2, 3):  ##len(featuress)):
            # for itype in range(0, 1):
            fname = infnames[ifrom]
            features = featuress[itype]
            mname_header = mname_headers[ifrom] + str(itype + typeoffset)
            kfold_cv_model(fname, mname_header, keys, features, regressor, n_splits, n_repeats,
                           temp=None, T_logic="larger",
                           min_samples_leaf=None, warm_start=True,
                           style="StratifiedKFold", random_state=None, find_outfliers=find_outliers)

        df = pd.read_csv(os.path.join(datapath, fKFOLD))

        inds4e = []
        inds4re = []
        inds4std = []
        inds4all = []

        for i in range(0, len(df)):
            inds4e.append(df["imax_abs"][i])
            inds4re.append(df["imax_rabs"][i])
            inds4std.append(df["imax_std"][i])
            inds4all.append(df["imax_abs"][i])
            inds4all.append(df["imax_rabs"][i])
            inds4all.append(df["imax_std"][i])

        inds4e = np.array(inds4e)
        inds4re = np.array(inds4re)
        inds4std = np.array(inds4std)
        inds4all = np.array(inds4all)

        unique_inds4e, counts4e = np.unique(inds4e, return_counts=True)
        unique_inds4re, counts4re = np.unique(inds4re, return_counts=True)
        unique_inds4std, counts4std = np.unique(inds4std, return_counts=True)
        unique_inds4all, counts4all = np.unique(inds4all, return_counts=True)
        imax4e = np.argmax(counts4e)
        imax4re = np.argmax(counts4re)
        imax4std = np.argmax(counts4std)
        imax4all = np.argmax(counts4all)

        idfe = unique_inds4e[imax4e]
        idfre = unique_inds4re[imax4re]
        idfstd = unique_inds4std[imax4std]
        idfa = unique_inds4all[imax4all]

        df_data = pd.read_csv(os.path.join(DATA_PATH, infnames[0]))
        compstre = df_data.iloc[idfe]['Composition']
        compstrre = df_data.iloc[idfre]['Composition']
        compstrstd = df_data.iloc[idfstd]['Composition']
        compstrall = df_data.iloc[idfa]['Composition']
        targete = df_data.iloc[idfe][keys[0]]
        targetre = df_data.iloc[idfre][keys[0]]
        targetstd = df_data.iloc[idfstd][keys[0]]
        targetall = df_data.iloc[idfa][keys[0]]


        if iloop % 20 == 0:
            print(f"--- start {iloop} ---")
            print("--- absolute error ---")
            #print(f"unique_inds4e:{unique_inds4e} counts4e:{counts4e}")
            print(f"imax4e:{imax4e} counts4e:{counts4e[imax4e]} ind4e:{idfe}")
            print(f"compstre:{compstre} target:{targete}")
            print("--- relative error ---")
            #print(f"unique_inds4re:{unique_inds4re} counts4re:{counts4re}")
            print(f"imax4re:{imax4re} counts4re:{counts4re[imax4re]} ind4re:{idfre}")
            print(f"compstrre:{compstrre} target:{targetre}")
            print("--- standardized error ---")
            #print(f"unique_inds4std:{unique_inds4std} counts4std:{counts4std}")
            print(f"imax4std:{imax4std} counts4std:{counts4std[imax4std]} ind4std:{idfstd}")
            print(f"compstrstd:{compstrstd} target:{targetstd}")
            print("--- all ---")
            #print(f"unique_inds4all:{unique_inds4all} counts4all:{counts4all}")
            print(f"imax4all:{imax4all} counts4all:{counts4all[imax4all]} ind4all:{idfa}")
            print(f"compstrall:{compstrall} target:{targetall}")
        print(f"--- end {iloop} ---" + "\n")

        if delete_style == 0:
            df_data = df_data.drop(df_data.index[idfe])
        elif delete_style == 1:
            df_data = df_data.drop(df_data.index[idfre])
        elif delete_style == 2:
            df_data = df_data.drop(df_data.index[idfstd])
        elif delete_style == 3:
            df_data = df_data.drop(df_data.index[idfa])
        else:
            print("error")
            sys.exit()

        if iloop == nloopthis - 1:
            final_fname = infnames[0].replace(".csv", "_" + str(delete_style) + "_final.csv")
            df_data.to_csv(os.path.join(DATA_PATH, final_fname), index=False)
        else:
            df_data.to_csv(os.path.join(DATA_PATH, infnames[0]), index=False)
        fKFOLD_out = "KFOLD_"+str(delete_style)+"_"+str(iloop)+".csv"
        shutil.copyfile(os.path.join(datapath, fKFOLD_summary), os.path.join(datapath, str(delete_style), fKFOLD_out))








