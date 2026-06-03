import os

import numpy as np
import pandas as pd

from myml.data.data import myData, DATA_PATH
from myml.models.models import MLRegressor

float_format = '%.8f'
from_DATA = False


def init_columns(m):
    cols = []
    for i in range(m):
        cols.append(str(i))
    return cols


def init_thisdict(features):
    thisdict = {}
    for i in range(len(features)):
        thisdict[str(i)] = features[i]
    return thisdict


def kfold_cv_model(fname, mname_header, keys, features, regressor, n_splits, n_repeats,
                   temp=None, T_logic="larger",
                   min_samples_leaf=None, warm_start=True,
                   style="StratifiedKFold", random_state=None, find_outfliers=False):
    Source = "EXTERNAL"
    thisdata = myData(os.path.join(DATA_PATH, fname), from_DATA=from_DATA, Source=Source)
    if temp is None:
        pass
    else:
        key = "temperature"
        if T_logic[0:4].upper() == "LARG":
            condition = "values>" + str(temp)
            thisdata.gooddf, baddf = thisdata.select_by_DF(thisdata.gooddf, key, condition, set2gooddf=True)
        elif T_logic[0:4].upper() == "SMAL":
            condition = "values<" + str(temp)
            thisdata.gooddf, baddf = thisdata.select_by_DF(thisdata.gooddf, key, condition, set2gooddf=True)
        else:
            condition = "values<" + str(temp + 50.0)
            thisdata.gooddf, baddf = thisdata.select_by_DF(thisdata.gooddf, key, condition, set2gooddf=True)
            condition = "values>" + str(temp - 50.0)
            thisdata.gooddf, baddf = thisdata.select_by_DF(thisdata.gooddf, key, condition, set2gooddf=True)

    learning_rate = 0.01
    max_iter = 1000
    thisMLR = MLRegressor(thisdata, keys, mname_header=mname_header, modelname=regressor,
                          features=features, learning_rate=learning_rate,
                          min_samples_leaf=min_samples_leaf, warm_start=warm_start)
    perform_df = thisMLR.kfold_crossvalidation(n_splits, n_repeats=n_repeats, style=style, random_state=random_state,
                                               find_outfliers=find_outfliers)

    return perform_df


def SFS_GBR(allfs, fname, mname_head, keys, regressor, ftitle,
            nmax=8, temp=None, T_logic="between",
            min_samples_leaf=6, n_splits=5, n_repeats=20,
            style="StratifiedKFold", preselect=None):
    if isinstance(keys, str):
        keys = [keys]
    allfs = np.array(allfs)
    n = len(allfs)
    random_state = 32434

    if isinstance(preselect, list) or isinstance(preselect, np.ndarray):
        if len(preselect) > 0:
            selected = preselect[0:len(preselect)]
            selected = np.array(selected)
        else:
            selected = np.array([])
    else:
        selected = np.array([])
    avail = allfs[0:n]
    nmax = min(nmax, n + len(selected))
    nfs = []
    r2s = []
    r2stds = []
    r2pots = []
    while len(selected) < nmax:
        nfeat = len(selected)
        cols = init_columns(nfeat + 1)
        cols += ["R2_AVG_ALL", "R2_STDEV", "R2_POT"]
        df = pd.DataFrame(columns=cols)
        for i in range(len(avail)):
            features = np.append(selected, [avail[i]])
            mname_header = mname_head + "Test" + str(nfeat + 1)
            thisdict = init_thisdict(features)
            thisdf = kfold_cv_model(fname, mname_header, keys, features, regressor, n_splits, n_repeats,
                                    temp=temp, T_logic=T_logic,
                                    min_samples_leaf=min_samples_leaf, warm_start=True,
                                    style=style, random_state=random_state)

            thisdict["R2_AVG_ALL"] = thisdf.at[0, "R2_AVG_ALL"]
            thisdict["R2_STDEV"] = thisdf.at[0, "R2_STDEV"]
            thisdict["R2_POT"] = thisdf.at[0, "R2_AVG_ALL"] + thisdf.at[0, "R2_STDEV"]
            df.loc[len(df)] = thisdict

        df = df.sort_values(["R2_POT"])
        df = df.reset_index()
        bestdict = df.loc[len(df) - 1].to_dict()
        nfs.append(nfeat + 1)
        r2s.append(bestdict["R2_AVG_ALL"])
        r2stds.append(bestdict["R2_STDEV"])
        r2pots.append(bestdict["R2_POT"])
        thisfeats = np.array([])
        for i in range(nfeat + 1):
            thisfeats = np.append(thisfeats, bestdict[str(i)])
        for i in range(nfeat):
            thisfeats = np.delete(thisfeats, np.where(thisfeats == selected[i]))
        selected = np.append(selected, thisfeats)
        avail = np.delete(avail, np.where(avail == thisfeats[0]))

        df.to_csv("Feature_eval4" + ftitle + "_" + str(nfeat + 1) + ".csv")

    sfs_df = pd.DataFrame(columns=["nfeat", "R2_AVG_ALL", "R2_STDEV", "R2_POT"])
    sfs_df["nfeat"] = nfs
    sfs_df["R2_AVG_ALL"] = r2s
    sfs_df["R2_STDEV"] = r2stds
    sfs_df["R2_POT"] = r2pots
    sfs_df.to_csv("SFS4" + ftitle + ".csv")
