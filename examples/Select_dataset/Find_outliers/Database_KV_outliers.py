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
    pathKFOLD = os.path.join(LOCAL_DATA_PATH, mname_header)
    os.makedirs(pathKFOLD, exist_ok=True)
    return fKFOLD, fKFOLD_summary, pathKFOLD

def Get_outliers_from_KFOLD_CV(fname, mname_header, key, features, regressor, n_splits, n_repeats,
                     min_samples_leaf=None, warm_start=True,
                     style="StratifiedKFold", random_state=None, find_outliers=True, thres4outliers=3.0):

    kfold_cv_model(fname, mname_header, [key], features, regressor, n_splits, n_repeats,
                   temp=None, T_logic="larger",
                   min_samples_leaf=None, warm_start=True,
                   style="StratifiedKFold", random_state=None,
                   find_outliers=find_outliers, thres4outliers=thres4outliers)

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
            try:
                thislist.append(int(strlist[i]))
            except:
                pass
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

def handling_outliers(fname, key, fKFOLD, fKFOLD_summary, pathKFOLD, delete_style=3,
                      print_results=False, iloop=0, R2_thres=0.9):
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

    df_summary = pd.read_csv(os.path.join(LOCAL_DATA_PATH, fKFOLD_summary))
    if delete_style in (None, False):
        pass
    else:
        if delete_style < 4:
            if idfs[delete_style] is None:
                df_data = df_data.drop(df_data.index[idfs[2]])
            else:
                df_data = df_data.drop(df_data.index[idfs[delete_style]])
        else:
            print("error")
            sys.exit()
        df_data.to_csv(os.path.join(DATA_PATH, fname), index=False)
        fKFOLD_out = "KFOLD_" + str(delete_style) + "_" + str(iloop) + ".csv"
        os.makedirs(os.path.join(pathKFOLD, str(delete_style)), exist_ok=True)
        shutil.copyfile(os.path.join(LOCAL_DATA_PATH, fKFOLD_summary),
                        os.path.join(pathKFOLD, str(delete_style), fKFOLD_out))

    isStop = False
    df_summary = pd.read_csv(os.path.join(LOCAL_DATA_PATH, fKFOLD_summary))
    thisR2 = df_summary.at[0, "R2_AVG_ALL"]
    if thisR2 > R2_thres:
        isStop = True
    return isStop

mname_header = "Compression"
key = "Elongation_EXP" #key for target property
features = ["ShearBurger_Surf", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"]
ORGfile = "JT_compress_EXT.csv" #Original database file
infname = "New_Compress_EXT.csv" #database file after removing outliers
nloop = 80 #number of iterations to remove outliers.
# Each iteration will remove 1 outlier has the highest frequencies showing in the repeated k-fold cross validation.
R2_thres = 0.9 #stop crierion for stopping the iteration.

regressor = "GBR"
n_splits = 3
n_repeats = 30
## 3*30 = 90 evaluations, it will count the frequency of each outlier in these 90 evaluations.
#And then remove the outlier with the highest frequency.
min_samples_leaf = 6
find_outliers = True
thres4outliers = 3.0 #threshold for the standardized error for outliers.
##If the absolute standardized error is larger than this threshold, it will be considered as an outlier.
##The standardized error is calculated by the formula: (residuals) / stdev(residuals)
iattempt = 10
delete_style = 3
##type of errors: 0 max absolute error, 1 max relative error, 2 max standardized error, 3 delete the outliers


fKFOLD, fKFOLD_summary, pathKFOLD = Get_KFOLD_filename(mname_header, key, regressor)
shutil.copyfile(os.path.join(DATA_PATH, ORGfile), os.path.join(DATA_PATH, infname))
isStop = False
for iloop in range(0, nloop):
    Get_outliers_from_KFOLD_CV(infname, mname_header, key, features, regressor, n_splits, n_repeats,
                                min_samples_leaf=min_samples_leaf, warm_start=True,
                                style="StratifiedKFold", random_state=None,
                                find_outliers=find_outliers, thres4outliers=thres4outliers)
    print_results = False
    if iloop % 10 == 0:
        print_results = True
    isStop = handling_outliers(infname, key, fKFOLD, fKFOLD_summary, pathKFOLD, delete_style=delete_style,
                            print_results=True, iloop=iloop, R2_thres=R2_thres)

    print(f"--- ENO OF iloop:{iloop} ---")
    if isStop: break






