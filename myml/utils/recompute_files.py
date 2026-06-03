import os, sys
import numpy as np
import pandas as pd


float_format = '%.8f'
from myml.myglobal import TEMPERATURE_LEVEL
from myml.data.data import myData, DATA_PATH 

bccC2_fnames = ["bccC2_TC_300_IP.csv"]

bcc_fnames = ["bccC3_TC_300_IP.csv", "bccC4_from_NDIP.csv", "bccC5_from_NDIP.csv", "bccC6_from_NDIP.csv", \
          "bccC7_from_NDIP.csv", "bccC8_from_NDIP.csv", "bccC9_from_NDIP.csv"]

hcp_fnames = ["hcpC3_IP.csv", "hcpC4_from_NDIP.csv", "hcpC5_from_NDIP.csv", "hcpC6_from_NDIP.csv", \
          "hcpC7_from_NDIP.csv", "hcpC8_from_NDIP.csv", "hcpC9_from_NDIP.csv"]


bcc_YS_fnames = ["bccC3_TC_300_IP_YS.csv", "bccC4_from_NDIP_YS.csv", "bccC5_from_NDIP_YS.csv", "bccC6_from_NDIP_YS.csv", \
          "bccC7_from_NDIP_YS.csv", "bccC8_from_NDIP_YS.csv", "bccC9_from_NDIP_YS.csv"]

bcc_all_fnames = ["bccCall_HF_GBR.csv", "bccCall_YS_GBR.csv"]


ts = TEMPERATURE_LEVEL[1:len(TEMPERATURE_LEVEL)]
nt = len(ts)
fheader = "bccCall_ADD_T"
bcc_ADD_fnames=[]
for i in range(nt):
    fname = fheader+str(int(ts[i]))+".csv"
    bcc_ADD_fnames.append(fname)

EXT_fnames = ["Beniwal_Hardness_EXT.csv", "Giles_YS_EXT.csv", "MPEA_dataset_EXT.csv",
              "MPEA_Hardness_EXT.csv", "Compression_EXT.csv", "Elongation_EXT.csv"]
EXT_fnames += ["JT_compress_EXT.csv", "JT_compress4paper_EXT.csv", "JT_tensile_0compress4paper_EXT.csv",
               "JT_tensile_compress0_9eles_EXT.csv", "JT_tensile_compress0_EXT.csv",
               "JT_tensile_EXT.csv", "JT_tensile4paper_EXT.csv"]
EXT_fnames += ["Merged_ductility.csv", "MPEA_dataset_noAug_EXT.csv",
               "MPEA_dataset_noAug4paper_EXT.csv", "MPEA_dataset4paper_EXT.csv"]


def recompute_df_gmix_TC(df):
    isvalid = True
    if "temperature" in df.columns:
        ts = df["temperature"].to_numpy()
    else:
        ts = np.array([300.0]*len(df))
    if "Sconf" in df.columns:
        s = df["Sconf"].to_numpy()
    else:
        isvalid = False
    if "Hmix_TC" in df.columns:
        h = df["Hmix_TC"].to_numpy()
    else:
        isvalid = False
    if isvalid:
        gmix = h - ts * s
        df["Gmix_TC"] = gmix
    return df

def drop_cols(fname, dropcols):
    print("Start " + fname)
    df = pd.read_csv(os.path.join(DATA_PATH, fname))
    thisdrop = []
    for key in dropcols:
        if key in df.columns: thisdrop.append(key)

    df = df.drop(columns=thisdrop)
    df.to_csv(os.path.join(DATA_PATH, fname), index=False, float_format=float_format)
    print("Finish " + fname)

def rename_df(fname):
    rename_dict = {"Elong_M1": "Compress_M1"}
    print("Start " + fname)
    df = pd.read_csv(os.path.join(DATA_PATH, fname))
    cols = df.columns.tolist()
    if "Elong_M1" in cols:
        df = df.rename(columns=rename_dict)
        df.to_csv(os.path.join(DATA_PATH, fname), index=False, float_format=float_format)
    print("Finish " + fname)


from myml.sublatt.sublatt import Laves, B2Latt, Sub_SS
from myml.expansion.expansion import load_all_NDIP_models
from myml.utils.GBR_functions import data_expansion, merge_elongations

def recompute_sublattice(fname, FOLDER=DATA_PATH, isScaled=False, Scale_Eform=False, fine_tune=True):
    all_NDIP_dict = load_all_NDIP_models(keys=["Eform_BOKAS", "Gmix_TC"])

    print("Starting " + fname)
    df = pd.read_csv(os.path.join(FOLDER, fname))
    compstrs = df["Composition"].to_numpy()
    eforms = df["Eform_BOKAS"].to_numpy()
    gmixes = df["Gmix_TC"].to_numpy()
    b2l = []
    lsubstr = []
    b2l2 = []
    lsubstr2 = []
    b2b = []
    b2substr = []
    b2b2 = []
    b2substr2 = []
    subsss = []
    subsubstr = []
    subsss2 = []
    subsubstr2 = []
    for i in range(len(compstrs)):
        compstr = compstrs[i]
        eform = eforms[i]
        LV = Laves(compstr, eform, temp=300.0, style=1, isScaled=isScaled, Scale_Eform=Scale_Eform)
        e1, substr = LV.get_estimation4CCSub()
        e2, substr_2 = LV.get_estimation(all_NDIP_dict=all_NDIP_dict)
        b2l.append(e1)
        b2l2.append(e2)
        lsubstr.append(substr)
        lsubstr2.append(substr_2)
        BL = B2Latt(compstr, eform, temp=300.0, style=1, isScaled=isScaled, Scale_Eform=Scale_Eform)
        e1, substr = BL.get_estimation4CCSub()
        e2, substr_2 = BL.get_estimation(all_NDIP_dict=all_NDIP_dict)
        b2b.append(e1)
        b2b2.append(e2)
        b2substr.append(substr)
        b2substr2.append(substr_2)

        gmix = gmixes[i]
        SubSS = Sub_SS(compstr, gmix, temp=300.0, style=0, isScaled=isScaled,
                       Scale_Eform=Scale_Eform, fine_tune=fine_tune)
        e, substr_2 = SubSS.get_estimation(all_NDIP_dict=all_NDIP_dict)
        subsss.append(SubSS.ediff2)
        subsubstr.append(SubSS.substrs_2)
        subsss2.append(SubSS.ediff_fine)
        subsubstr2.append(SubSS.substrs_fine)


        if i % 1000 == 0:
            print("Finished sublattice calculation of " + str(i) + " compositions!")
    df["bcc2laves"] = b2l
    df["laves_substrs"] = lsubstr
    df["bcc2laves_2"] = b2l2
    df["laves_substrs_2"] = lsubstr2
    df["bcc2b2"] = b2b
    df["b2_substrs"] = b2substr
    df["bcc2b2_2"] = b2b2
    df["b2_substrs_2"] = b2substr2
    df["bcc2subss"] = subsss
    df["subss_substrs"] = subsubstr
    df["bcc2subss_2"] = subsss2
    df["subss_substrs_2"] = subsubstr2
    df.to_csv(os.path.join(FOLDER,fname), index=False, float_format=float_format)
    print("finished " + fname)

def recompute_elongations(fname, FOLDER=DATA_PATH):
    #df = pd.read_csv(os.path.join(folder, fname))
    print("Starting " + fname)
    keys = ["Elongation_EXP"]
    EL_featuress = [["ShearBurger", "D_Surf_GBR_ROM", "Exp_Shear_DISTORT", "volume_DISTORT"],
                        ["ShearBurger", "Exp_Surf_ROM", "Exp_Shear_DISTORT", "volume_DISTORT"],
                        None,
                        ["ShearBurger", "D_Surf_GBR_ROM", "Exp_Shear_DISTORT", "volume_DISTORT"]]

    OutKey_dicts = [{"Elongation_EXP": "Elong_M"}]
    mname_headers = ["Elongation_MPEA"]
    modelname = "GBR"
    ifrom = 0
    for itype in range(0, len(EL_featuress)):
        features = EL_featuress[itype]
        mname_header = mname_headers[ifrom] + str(itype + 1)
        tmp_dict = OutKey_dicts[ifrom]
        OutKey_dict = {}
        OutKey_dict["Elongation_EXP"] = tmp_dict["Elongation_EXP"] + str(itype + 1)
        data_expansion(fname, keys, mname_header, OutKey_dict, features, modelname,
                       FOLDER=FOLDER)
    merge_elongations(fname, FOLDER=FOLDER)
    print("finished " + fname)

def recompute_YS(fname, FOLDER=DATA_PATH):
    def normalization_YS(fname, ykey, thres=1.0):
        thisdata = myData(fname, Source="INTERNAL")
        ys = thisdata.df[ykey].to_numpy()
        xs = thisdata.df["temp_ratio"].to_numpy()
        thisdata.df[ykey] = np.select([xs < thres, xs >= thres], [ys, 0.0])
        thisdata.save_to(fname, df=thisdata.df)

    def compute_Tabor_yield(fname, ykey):
        thisdata = myData(fname, Source="INTERNAL")
        ys = thisdata.df[ykey].to_numpy()
        outkey = ykey.split("_")
        outkey = "Tabor_" + outkey[-1]
        thisdata.df[outkey] = ys * 3.0 / 1000.0
        thisdata.save_to(fname, df=thisdata.df)

    print("Starting " + fname)

    YS_featuress = [["temp_ratio", "delta_Eb0", "sigma_y0", "E_negativity",
                    "volume_DISTORT", "Exp_Shear_DISTORT", "Exp_Youngs_ROM"]]
    OutKey_dicts = [{"Yield_EXP": "Yield_M"}]
    mname_headers = ["Yield_MPEA"]
    keys = ["Yield_EXP"]
    modelname = "GBR"
    ifrom = 0
    for itype in range(0, len(YS_featuress)):
        features = YS_featuress[itype]
        mname_header = mname_headers[ifrom] + str(itype + 1)
        tmp_dict = OutKey_dicts[ifrom]
        OutKey_dict = {}
        OutKey_dict["Yield_EXP"] = tmp_dict["Yield_EXP"] + str(itype + 1)
        data_expansion(fname, keys, mname_header, OutKey_dict, features, modelname, FOLDER=FOLDER)
    normalization_YS(os.path.join(FOLDER, fname), "Yield_M1")
    compute_Tabor_yield(os.path.join(FOLDER, fname), "Yield_M1")
    print("finished " + fname)


def Ref2ADD_file(refname, outfname, Temp, FOLDER=DATA_PATH):
    dropcols = ["temperature", "temp_ratio", "Yield_MC", "Yield_M1", "Tabor_M1"]
    df_ref = pd.read_csv(os.path.join(FOLDER, refname))

    df_ref = df_ref.drop(columns=dropcols)
    df_ref.to_csv(os.path.join(FOLDER, outfname), index=False, float_format=float_format)

    Temp_Level = [Temp]
    thisdata = myData(os.path.join(FOLDER, outfname),
                      Recompute=False,
                      Temp_Level=Temp_Level)
    thisdata.compute_YieldStrength(thisdata.df, style="Exp", Recompute=True)
    thisdata.save_to(os.path.join(FOLDER, outfname), df=thisdata.df)
    print(f"finish compute yield strength for {outfname} at T:{Temp_Level}.")
    recompute_YS(outfname, FOLDER=FOLDER)

