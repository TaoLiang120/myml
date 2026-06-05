import os, sys
import numpy as np
import pandas as pd
import shutil

from myml.utils.GBR_functions import *
from myml.utils.SFS_gbr import kfold_cv_model
from myml.data.data import myData, DATA_PATH

float_format = '%.8f'
LOCAL_DATA_PATH = "MLR_Models"


def Get_KFOLD_filename(mname_header, key, regressor):
    fKFOLD = "KFOLD_" + mname_header + "_" + regressor + "_" + key + ".csv"
    fKFOLD_summary = "KFOLD_" + mname_header + "_" + regressor + "_" + "ALL" + ".csv"
    return fKFOLD, fKFOLD_summary

def Get_outliers_from_KFOLD_CV(fname, mname_header, key, features, regressor, n_splits, n_repeats,
                     min_samples_leaf=None, warm_start=True,
                     style="StratifiedKFold", random_state=None, find_outliers=True):

    kfold_cv_model(fname, mname_header, [key], features, regressor, n_splits, n_repeats,
                   temp=None, T_logic="larger",
                   min_samples_leaf=None, warm_start=True,
                   style="StratifiedKFold", random_state=None, find_outliers=find_outliers)

def Parsing_KFOLD_CV_results(fKFOLD):
    def tuple2list(instring):
        instring = instring.replace('"', '')
        instring = instring.replace('(', "")
        instring = instring.replace(")", "")
        instring = instring.replace("[", "")
        instring = instring.replace("]", "")
        instring = instring.replace(" ","")
        strlist = instring.split(",")
        thislist = []
        for i in range(0, len(strlist)):
            try: thislist.append(int(strlist[i]))
            except: pass
        return thislist

    df = pd.read_csv(os.path.join(LOCAL_DATA_PATH, fKFOLD))
    inds4e = []
    inds4re = []
    inds4std = []
    inds4outliers = []

    for i in range(0, len(df)):
        inds4e.append(df["imax_abs"][i])
        inds4re.append(df["imax_rabs"][i])
        inds4std.append(df["imax_std"][i])
        inds4outliers += tuple2list(df["outliers"][i])
    inds4e = np.array(inds4e)
    inds4re = np.array(inds4re)
    inds4std = np.array(inds4std)
    inds4outliers = np.array(inds4outliers)
    return [inds4e, inds4re, inds4std, inds4outliers]

def handling_outliers(fname, key, fKFOLD, fKFOLD_summary, delete_style=3, print_results=False, iloop=0):
    df_data = pd.read_csv(os.path.join(DATA_PATH, fname))
    indss = Parsing_KFOLD_CV_results(fKFOLD)
    idfs = []
    for i in range(0, len(indss)):
        if len(indss[i]) == 0:
            idfs.append(None)
        else:
            uniques, counts = np.unique(indss[i], return_counts=True)
            imax = np.argmax(counts)
            idf = uniques[imax]
            idfs.append(idf)
            if print_results:
                if i == 0:
                    print("--- max absolute error ---")
                elif i == 1:
                    print("--- max relative error ---")
                elif i == 2:
                    print("--- max standardized error ---")
                else:
                    print("--- outliers ---")
                print(f"uniques:{uniques} counts:{counts}")
                print(f"imax:{imax} count4max:{counts[imax]} idf:{idf}")
                print(f"compstr:{df_data.iloc[idf]['Composition']} target:{df_data.iloc[idf][key]}")

    if delete_style in (None, False):
        pass
    else:
        if delete_style < 4:
            if idfs[delete_style] is None:
                pass
            else:
                df_data = df_data.drop(df_data.index[idfs[delete_style]])
        else:
            print("error")
            sys.exit()
        df_data.to_csv(os.path.join(DATA_PATH, fname), index=False)
        fKFOLD_out = "KFOLD_" + str(delete_style) + "_" + str(iloop) + ".csv"
        os.makedirs(os.path.join(LOCAL_DATA_PATH, str(delete_style)), exist_ok=True)
        shutil.copyfile(os.path.join(LOCAL_DATA_PATH, fKFOLD_summary), os.path.join(LOCAL_DATA_PATH, str(delete_style), fKFOLD_out))

featuress = [["ShearBurger_Surf", "volume_DELTA", "Exp_Shear_ROM", "Exp_Shear_DISTORT", "volume_DISTORT_Max"],
             ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"]]

featuress = [["ShearBurger", "Exp_Surf_ROM", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"],
             ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"],
             ["ShearBurger_Surf", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"]]


#infnames = ["JT_compress_EXT_no50.csv"] #214 data
#infnames = ["Compression_EXT_no50.csv"] #113

#infnames = ["Compress_EXT_no50_BCC.csv"] #138
#infnames = ["JT_compress_EXT.csv"] #258
#infnames = ["JT_compress4paper_EXT.csv"] #129 data
#infnames = ["JT_compress4paper_EXT_no50.csv"] #103 data

mname_header = "Compression_test"
key = "Elongation_EXP"
regressor = "GBR"
features = ["ShearBurger_Surf", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"]

n_splits = 3
n_repeats = 30
min_samples_leaf = 6
find_outliers = True
iattempt = 10

ORGfile = "JT_compress_EXT.csv" #258
infname = "New_test_EXT.csv"
nloop = 80
delete_style = 1
delete_styles = [2, 3] ##0 max absolute error, 1 max relative error, 2 max standardized error, 3 delete the outliers

for delete_style in delete_styles:
    print(f"--- delete_style:{delete_style} ---")
    os.makedirs(os.path.join(LOCAL_DATA_PATH,str(delete_style)), exist_ok=True)
    fKFOLD, fKFOLD_summary = Get_KFOLD_filename(mname_header, key, regressor)
    for iloop in range(0, nloop):
        print(f"--- iloop:{iloop} ---")

        Get_outliers_from_KFOLD_CV(infname, mname_header, key, features, regressor, n_splits, n_repeats,
                     min_samples_leaf=min_samples_leaf, warm_start=True,
                     style="StratifiedKFold", random_state=None, find_outliers=find_outliers)
        print_results = False
        if iloop % 10 == 0:
            print_results = True
        handling_outliers(infname, key, fKFOLD, fKFOLD_summary, delete_style=delete_style,
                          print_results=True, iloop=iloop)


