import os

import numpy as np
import pandas as pd

from myml.myglobal import Element_external
from myml.myglobal import Element_negativity, b2h4ml, thres4b2h
from myml.data.data import myData, DATA_PATH
from myml.models.models import MLRegressor
from myml.expansion.expansion import DataExpansion


float_format = '%.8f'

from_DATA = False
performance_keys = ["iloop", "key", "mse_test", "rmse_test", "mae_test", "rmae_test", "r2_test",
                    "mse_all", "rmse_all", "mae_all", "rmae_all", "r2_all"]


def ext_train_model(fname, mname_header, thres, keys, features, regressor,
                    temp=None, T_logic="larger", colorkey=None, loadmodel=False, savemodel=True, nloop=100,
                    perform_df=None, elements=None,
                    thres_diff=5.0, set2all=False, min_samples_leaf=None, warm_start=True, TEST=False,
                    SHAP_Plot=True, SHAP_output=False, PDP_output=False, **kwargs):
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

    if perform_df is None: perform_df = pd.DataFrame(columns=performance_keys)
    perform_df = perform_df.set_index(performance_keys[0])

    normalization = False
    extended = False
    Visualize_Results = False
    Find_PDP = True
    Find_interPDP = False
    interactive_features = None
    if temp is None:
        Save_Pred2Data = True
    else:
        Save_Pred2Data = False
    savefig = True

    activation = "relu"
    learning_rate_style = "adaptive"  #"constant"
    learning_rate = 0.01
    max_iter = 1000

    plot_predict = savemodel
    plot_predict_all = True
    compute_R2 = True
    if TEST:
        Find_PDP = False
        SHAP_Plot = False
        Save_Pred2Data = False
    R2 = 0.0
    R2_diff = 100.0
    iloop = 0
    r2_avg = 0.0
    while R2 < thres and iloop < nloop:
        thisMLR = MLRegressor(thisdata, keys, mname_header=mname_header, modelname=regressor,
                              elements=elements, features=features, Find_PDP=Find_PDP, Find_interPDP=Find_interPDP,
                              interactive_features=interactive_features, SHAP_Plot=SHAP_Plot,
                              activation=activation, learning_rate_style=learning_rate_style,
                              learning_rate=learning_rate, min_samples_leaf=min_samples_leaf, warm_start=warm_start)
        thisMLR.regression_models(Save_Pred2Data=Save_Pred2Data, loadmodel=loadmodel, savemodel=savemodel,
                                  savefig=savefig, plot_predict=plot_predict, plot_predict_all=plot_predict_all,
                                  colorkey=colorkey, SHAP_data="ALL", SHAP_output=SHAP_output, PDP_output=PDP_output,
                                  **kwargs)

        R2 = thisMLR.performance_df.loc[keys[0], "r2_test"]
        R2_all = thisMLR.performance_df.loc[keys[0], "r2_all"]
        R2_diff = abs(R2 - R2_all) * 100.0
        print("00000 GBR FITTING 00000")
        print(f"ndata:{thisMLR.ndata}")
        print(f"mname:{mname_header} iloop:{iloop} R2: {R2} R2_All: {R2_all} R2_diff:{R2_diff}")

        thisdf = thisMLR.performance_df.copy()
        thisdf[performance_keys[0]] = [iloop]
        thisdf = thisdf.reset_index().set_index(performance_keys[0])
        perform_df.loc[iloop] = thisdf.loc[iloop]

        if set2all: R2 = R2_all
        if R2_diff > thres_diff: R2 = thres - 0.01
        r2_avg += R2_all
        print(f"*********** R2:  {R2} *****************")
        iloop += 1
    r2_avg /= iloop
    print(f"===== r2_avg: {r2_avg}  =====")
    perform_df["R2_AVG_ALL"] = r2_avg
    return perform_df


def data_expansion(fname, keys, mname_header, OutKey_dict, features, regressor,
                   FOLDER=DATA_PATH,  Source="INTERNAL",
                   Save_OrgData=True, Save_Pred2Data=True, Scale_Eform=False):
    from_ncompon = 3
    thisdata = myData(os.path.join(FOLDER, fname), from_DATA=from_DATA, Source=Source)
    thisDExp = DataExpansion(thisdata, keys, from_ncompon,
                             modelname=regressor, mname_header=mname_header,
                             features=features, Scale_Eform=Scale_Eform)
    thisDExp.fill_data(Add_Predict=False, Save_Pred2Data=True, OutKey_dict=OutKey_dict, Save_OrgData=Save_OrgData)
    print(f"finished {fname}!")


def merge_elongations(fname, thres=thres4b2h, scale=False, FOLDER=DATA_PATH):
    print("start " + fname)
    df = pd.read_csv(os.path.join(FOLDER, fname))
    df["Compress_M1"] = np.select([df["Compress_M1"] < 0.0, df["Compress_M1"] >= 0.0],
                                  [0.0, df["Compress_M1"]])
    if "Compress_M2" in df.columns:
        df["Compress_M2"] = np.select([df["Compress_M2"] < 0.0, df["Compress_M2"] >= 0.0],
                                  [0.0, df["Compress_M2"]])

    if "Compress_M3" in df.columns:
        df["Compress_M3"] = np.select([df["Compress_M3"] < 0.0, df["Compress_M3"] >= 0.0],
                                  [0.0, df["Compress_M3"]])

    df["Elong_M1"] = np.select([df["Elong_M1"] < 0.0, df["Elong_M1"] >= 0.0],
                               [0.0, df["Elong_M1"]])

    if "Elong_M2" in df.columns:
        df["Elong_M2"] = np.select([df["Elong_M2"] < 0.0, df["Elong_M2"] >= 0.0],
                               [0.0, df["Elong_M2"]])

    if "Elong_M3" in df.columns:
        df["Elong_M3"] = np.select([df["Elong_M3"] < 0.0, df["Elong_M3"] >= 0.0],
                                   [0.0, df["Elong_M3"]])

    if "Elong_M4" in df.columns:
        df["Elong_M4"] = np.select([df["Elong_M4"] < 0.0, df["Elong_M4"] >= 0.0],
                                   [0.0, df["Elong_M4"]])

    if scale:
        if "Compress_M2" in df.columns:
            xs = df[b2h4ml].to_numpy()
            df["Compress_M2"] = np.select(
                                [xs < thres, xs >= thres],
                                [df["Compress_M2"], df["Compress_M2"] / 10.0])

    df.to_csv(os.path.join(FOLDER, fname), index=False, float_format=float_format)
    print("Finish " + fname)


def normalize_yield(fname, ykey, thres=1.0, FOLDER=DATA_PATH):
    print("start " + fname)
    df = pd.read_csv(os.path.join(FOLDER, fname))
    xs = df["temp_ratio"].to_numpy()
    df[ykey] = np.select([xs < thres, xs >= thres], [df[ykey], 0.0])
    df.to_csv(os.path.join(FOLDER, fname), index=False, float_format=float_format)
    print("Finish " + fname)


def compute_hv_yield(fname, FOLDER=DATA_PATH):
    print("start " + fname)
    df = pd.read_csv(os.path.join(FOLDER, fname))
    if "Yield_M1" in df.columns:
        ys = df["Yield_M1"].to_numpy()
    elif "Predicted_Yield_EXP" in df.columns:
        ys = df["Predicted_Yield_EXP"].to_numpy()
    else:
        ys = np.zeros(len(df))
    ys = ys * 3.0 / 1000.0
    df["Tabor_M1"] = ys
    df.to_csv(os.path.join(FOLDER, fname), index=False, float_format=float_format)


def int_train_model(fname, keys, regressor, temp=None, savefig=False, figapp=None, thres=None,
                    loadmodel=False, savemodel=True, elements=None, SHAP_Plot=True, SHAP_output=False, PDP_output=False):
    Source = "INTERNAL"
    thisdata = myData(os.path.join(DATA_PATH, fname), from_DATA=from_DATA, Source=Source)
    if temp is None:
        pass
    else:
        key = "temperature"
        condition = "values>" + str(temp - 100.0)
        thisdata.gooddf, baddf = thisdata.select_by_DF(thisdata.gooddf, key, condition, set2gooddf=True)
        condition = "values<" + str(temp + 100.0)
        thisdata.gooddf, baddf = thisdata.select_by_DF(thisdata.gooddf, key, condition, set2gooddf=True)

    normalization = False
    extended = False
    Visualize_Results = False
    Importance_mode = [0, 1]
    Vertical = False
    Find_PDP = True
    subsample = 100
    valid_size = 0.1
    learning_rate = 0.01
    n_estimators = 500  ##similar to iterations
    max_iter = 500
    random_state4data = None

    random_state4eval = None
    n_repeats = 20
    mname_header = None

    if thres is None: thres = 0.1
    R2 = 0.0
    while R2 < thres:
        thisMLR = MLRegressor(thisdata, keys, mname_header=mname_header, modelname=regressor,
                              elements=elements, ncompons=None,
                              normalization=normalization, extended=extended, SHAP_Plot=SHAP_Plot,
                              Visualize_Results=False, Importance_mode=Importance_mode, Find_PDP=Find_PDP,
                              Vertical=Vertical, subsample=subsample,
                              valid_size=valid_size, random_state=random_state4data,
                              n_estimators=n_estimators, max_iter=max_iter)

        thisMLR.regression_models(random_state=random_state4eval, n_repeats=n_repeats, SHAP_data="TEST",
                                  loadmodel=loadmodel, savemodel=savemodel, savefig=savefig, figapp=figapp,
                                  SHAP_output=SHAP_output, PDP_output=PDP_output)

        R2 = thisMLR.performance_df.loc[keys[0], "r2_test"]
        print(f"temp:{temp} R2: {R2}")
        print("===")


def concat_dfs(YS=True):
    if not YS:
        fnames = ["bccC3_TC_300_IP.csv", "bccC4_from_NDIP.csv", "bccC5_from_NDIP.csv", "bccC6_from_NDIP.csv",
                  "bccC7_from_NDIP.csv", "bccC8_from_NDIP.csv", "bccC9_from_NDIP.csv"]

        outfile = "bccCall_HF_GBR.csv"
        for ifile in range(len(fnames)):
            fname = fnames[ifile]
            thisdf = pd.read_csv(os.path.join(DATA_PATH, fname))
            if ifile == 0:
                df = thisdf.copy(deep=True)
            else:
                df = pd.concat([df, thisdf])
        df.to_csv(os.path.join(DATA_PATH, outfile), index=False, float_format=float_format)
    else:
        fnames = ["bccC3_TC_300_IP_YS.csv", "bccC4_from_NDIP_YS.csv", "bccC5_from_NDIP_YS.csv", "bccC6_from_NDIP_YS.csv",
                  "bccC7_from_NDIP_YS.csv", "bccC8_from_NDIP_YS.csv", "bccC9_from_NDIP_YS.csv"]
        outfile = "bccCall_YS_GBR.csv"
        for ifile in range(len(fnames)):
            fname = fnames[ifile]
            thisdf = pd.read_csv(os.path.join(DATA_PATH, fname))
            if ifile == 0:
                df = thisdf.copy(deep=True)
            else:
                df = pd.concat([df, thisdf])
        df.to_csv(os.path.join(DATA_PATH, outfile), index=False, float_format=float_format)


FHEADERS = ["bccC3_TC_300_IP", "bccC4_from_NDIP", "bccC5_from_NDIP", "bccC6_from_NDIP",
            "bccC7_from_NDIP", "bccC8_from_NDIP", "bccC9_from_NDIP"]


def data_compute_features(Recompute, Compute_YS, fheaders=FHEADERS, FOLDER=DATA_PATH):
    for fheader in fheaders:
        if "EXT" in fheader or "bccC2_" in fheader:
            Source = "EXTERNAL"
            elements = Element_external
        else:
            Source = "INTERNAL"
            elements = Element_negativity

        fname = fheader + ".csv"
        thisdata = myData(os.path.join(FOLDER, fname), from_DATA=from_DATA,
                          Source=Source, Recompute=Recompute, elements=elements)

        if "bcc" in fheader:
            if not Compute_YS:
                thisdata.compute_Hardness_features(style="ExpSCALED")
                outfile = fheader + ".csv"
                outfile = os.path.join(FOLDER, outfile)
                thisdata.save_to(outfile, df=thisdata.df)
                print(f"finished {fheader} hardness")
            else:
                thisdata.compute_Yield_and_features(style="ExpSCALED", UseCxx=False)
                outfile = fheader + "_YS.csv"
                outfile = os.path.join(FOLDER, outfile)
                thisdata.save_to(outfile, df=thisdata.df)
                print(f"finished {fheader} Yield strength")
        else:
            outfile = fheader + ".csv"
            outfile = os.path.join(FOLDER, outfile)
            thisdata.save_to(outfile, df=thisdata.df)

        print(f"finished {fheader}")
        print("=====")

fnames4ADD = ["bccCall_ADD_T0.csv", "bccCall_ADD_T300.csv", "bccCall_ADD_T600.csv", "bccCall_ADD_T900.csv",
              "bccCall_ADD_T1200.csv", "bccCall_ADD_T1500.csv", "bccCall_ADD_T1800.csv", "bccCall_ADD_T2100.csv"]
