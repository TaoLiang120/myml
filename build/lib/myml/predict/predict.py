import os
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

from myml.myglobal import Element_Additional, config_vars, Constants, TEMPERATURE_LEVEL
from myml.myglobal import load_complist_from_json, b2h4ml, b2h4phase, thres4b2h
from myml.data.data import myData
from myml.expansion.expansion import CSExpansion, DataExpansion, load_all_NDIP_models
from myml.sublatt.sublatt import Laves, B2Latt, Sub_SS

DATA_PATH = config_vars["DATA_PATH"]
float_format = Constants["float_format"]

showcols1 = ["Composition", "ReducedFormula", "LattPara", "Eform_BOKAS", "bcc2hcp_Gmix_TC2CPA",
             "sigma_y0", "delta_Eb0", "DOS_EF_ROM", "ShearBurger", "Exp_Surf_ROM", "E_USF_ROM", "E_GBR_ROM",
             "temperature", "Tm_ROM"]

showcols2 = ["Composition", "radius_TC", "volume_DELTA", "volume_DISTORT_Max", "Gmix_TC", "bcc2hcp_Gmix_TC",
             "bcc2laves", "bcc2b2", "Compress_M1", "Vicker_M1", "Elong_M1", "Elong_M2", "Elong_M3",
             "Yield_M1"]

shortcols= ["Composition", "ReducedFormula", "LattPara", "radius_TC", "volume_DELTA",
            "Exp_Shear_ROM", "Exp_Youngs_ROM",
            "Gmix_TC", b2h4phase,
            "bcc2laves", "laves_substrs", "bcc2laves_2", "laves_substrs_2",
            "bcc2b2", "b2_substrs", "bcc2b2_2", "b2_substrs_2",
            "bcc2subss", "subss_substrs", "bcc2subss_2", "subss_substrs_2",
            "Tm_ROM", "temperature", "Yield_M1",
            "Vicker_M1", "Compress_M1", "Compress_M2",
            "Elong_M1", "Elong_M2", "Elong_M3"
            ]

rename_dict = {"ReducedFormula": "RedComp", "radius": "r_CPA", "radius_TC": "r_TC",
               "Exp_Bulk_ROM": "Bulk", "Exp_Poisson_ROM": "Poisson",
               "Exp_Shear_ROM": "Shear", "Exp_Youngs_ROM": "Youngs",
               "Eform": "Hmix_CPA", "Eform_str": "Hmix_str_CPA",  "Eform_BOKAS": "Hmix_VASP", "bcc2hcp": "b2h_CPA",
               "Hmix_TC": "Hmix", "Gmix_TC": "Gmix", "bcc2hcp_Gmix_TC": "b2h", "bcc2hcp_Gmix_TC2CPA": "b2h_TC2CPA",
               "Hmix_str_TC": "Hmix_str", "Gmix_str_TC": "Gmix_str",
               "bcc2laves": "b2laves", "bcc2b2": "b2b2",
               "volume_DELTA": "v_DELTA", "radius_TC_DELTA": "r_DELTA",
               "volume_DISTORT": "v_DISTORT", "radius_TC_DISTORT": "r_DISTORT",
               "volume_DISTORT_Max": "v_DIS_Max", "ShearBurger": "ShearBurg", "Exp_Surf_ROM": "eSurf",
               "ShearBurger_Surf": "ShearB_Surf", "Exp_Poisson_Comp": "1+P_1_P", "DOS_EF_ROM": "DOS",
               "E_USF_ROM": "USF", "USF_Surf": "USF_Surf", "E_GBR_ROM": "eGB",
               "temperature": "T4YS", "Tm_ROM": "Tm",
               "Vicker_M1": "VH_M1", "Yield_MC": "YS_MC",
               "Yield_M1": "YS_M1"}


def compute_df_composite_Poisson(df, style="Exp"):
    p = df["Exp_Poisson_ROM"].to_numpy()
    v = (1 + p) / (1 - p)
    df["Exp_Poisson_Comp"] = v

def display_df(df, sort_by=["Yield_M1"], DisplayCols=None, Index="Composition"):
    pd.set_option('display.max_rows', 100)
    df = df.sort_values(sort_by)
    if DisplayCols is None:
        print(df[showcols1].rename(columns=rename_dict).set_index(Index))
        print(df[showcols2].rename(columns=rename_dict).set_index(Index))
        print("************")
    else:
        print(df[DisplayCols])
        print("************")

class BasicPredictor:
    def __init__(self, indict, style=1, modelname="NDIP", style4sub=1, Scale_Eform=False, fine_tune=True):
        self.indict = indict
        self.style = style
        self.modelname = modelname
        self.style4sub = style4sub
        self.Scale_Eform = Scale_Eform
        self.fine_tune = fine_tune

    @staticmethod
    def read_predict_input(fname):
        outdict = {}
        outdict["InFile"] = "compstrs.json"
        outdict["Temp_Level"] = TEMPERATURE_LEVEL
        outdict["YS_MID"] = 1
        outdict["OutFile"] = "prediction_out.csv"
        outdict["Scale_Eform"] = False
        outdict["fine_tune"] = True
        outdict["style"] = 1
        outdict["style4sub"] = 1
        with open(fname, 'r') as fh:
            for line in fh.readlines():
                line = line.replace(" ", "")
                line = line.replace('\n', '')
                if not line.startswith("#") and "=" in line:
                    key, value = line.split('=')
                    tmplist = value.split('#')
                    value = tmplist[0]
                    value = value.replace('\'', '')
                    value = value.replace('\"', '')
                    value = value.replace('[', '')
                    value = value.replace(']', '')
                    if key == "InFile" or key == "OutFile":
                        outdict[key] = value
                    if key == "YS_MID":
                        try:
                            outdict[key] = int(value)
                        except:
                            outdict[key] = 1
                    elif key == "Temp_Level":
                        tmplist = value.split(',')
                        for m in range(len(tmplist)):
                            tmplist[m] = float(tmplist[m])
                        outdict[key] = tmplist
                    elif key == "fine_tune" or key == "Scale_Eform":
                        if value[0:1].upper() == "T":
                             outdict[key] = True
                        elif value[0:1].upper() == "F":
                            outdict[key] = False
                    elif key == "style" or key == "style4sub":
                        try:
                            outdict[key] = int(value[0:1])
                        except:
                            pass

        return outdict

    @classmethod
    def from_finput(cls, finput, style=1, modelname="NDIP", style4sub=1):
        indict = cls.read_predict_input(finput)
        thisobj = cls(indict, style=style, modelname=modelname, style4sub=style4sub,
                      Scale_Eform=indict["Scale_Eform"], fine_tune=indict["fine_tune"])
        return thisobj

    def get_basic_properties(self, fill_transform=True):
        fdatabase = "bccC3_TC_300_IP.csv"
        compstrs = load_complist_from_json(self.indict["InFile"])
        compstrs = np.array(compstrs)
        thisdata = myData(os.path.join(DATA_PATH, fdatabase), from_DATA=True, Source="INTERNAL")

        mode = "elastic"
        CSE = CSExpansion(compstrs, thisdata, modelname=self.modelname, mode=mode, fill_transform=fill_transform,
                          Scale_Eform=self.Scale_Eform)
        CSE.fill_data(style=self.style, loadmodel=True, savemodel=False, savefile=True, outfile=self.indict["OutFile"],
                      Load_all_NDIP=True)

    def display_predictions(self):
        pd.set_option('display.max_rows', 100)
        thisdata = myData(self.indict["OutFile"], from_DATA=False, Source="INTERNAL")
        if "Hmix_TC" in thisdata.df.columns:
            print(thisdata.df[showcols1].rename(columns=rename_dict).set_index("Composition"))
        else:
            print(thisdata.df[showcols2].rename(columns=rename_dict).set_index("Composition"))
        thisdata.save_to(self.indict["OutFile"], df=thisdata.df)



class ExtendPredictor(BasicPredictor):
    def __init__(self, indict, style=1, modelname="NDIP", style4sub=1, Scale_Eform=False, fine_tune=True):
        super().__init__(
            indict,
            style=style,
            modelname=modelname,
            style4sub = style4sub,
            Scale_Eform=Scale_Eform,
            fine_tune=fine_tune,
        )

    def compute_extend_features(self):
        thisdata = myData(self.indict["OutFile"], from_DATA=False, Source="INTERNAL",
                          Recompute=True, Temp_Level=self.indict["Temp_Level"])
        thisdata.compute_Hardness_features(style="Exp")
        thisdata.save_to(self.indict["OutFile"], df=thisdata.df)

    def compute_temp_features(self, FileSeparation=False):
        thisdata = myData(self.indict["OutFile"], from_DATA=False, Source="INTERNAL",
                          Recompute=False, Temp_Level=self.indict["Temp_Level"])
        thisdata.compute_Yield_and_features(style="Exp", UseCxx=False, FileSeparation=FileSeparation)
        thisdata.save_to(self.indict["OutFile"], df=thisdata.df)

    def data_expansion(self, keys, features, OutKey_dict, modelname, mname_header, input_fname=None):
        thisfname = self.indict["OutFile"]
        if isinstance(input_fname, str):
            thisfname = input_fname
        thisdata = myData(thisfname, from_DATA=False, Source="INTERNAL")
        if self.style == 1:
            from_ncompon = 3
        else:
            from_ncompon = 2
        thisDExp = DataExpansion(thisdata, keys, from_ncompon, modelname=modelname, mname_header=mname_header,
                                 features=features, Scale_Eform=self.Scale_Eform)
        thisDExp.fill_data(style=self.style,
                           Add_Predict=False, Save_Pred2Data=False, OutKey_dict=OutKey_dict, Save_OrgData=True)

    def compute_sublatt_eform(self):
        all_NDIP_dict = load_all_NDIP_models(keys=["Eform_BOKAS", "Gmix_TC"])

        thisdata = myData(self.indict["OutFile"], from_DATA=False, Source="EXTERNAL")
        compstrs = thisdata.df["Composition"].to_numpy()
        eforms = thisdata.df["Eform_BOKAS"].to_numpy()
        gmixes = thisdata.df["Gmix_TC"].to_numpy()
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
            if self.style4sub == 1:
                eform = eforms[i]
            else:
                eform = gmixes[i]
            LV = Laves(compstr, eform, temp=300.0, style=self.style4sub, isScaled=self.Scale_Eform, Scale_Eform=self.Scale_Eform)
            e1, substr = LV.get_estimation4CCSub()
            e2, substr_2 = LV.get_estimation(all_NDIP_dict=all_NDIP_dict)
            b2l.append(e1)
            b2l2.append(e2)
            lsubstr.append(substr)
            lsubstr2.append(substr_2)
            BL = B2Latt(compstr, eform, temp=300.0, style=self.style4sub, isScaled=self.Scale_Eform, Scale_Eform=self.Scale_Eform)
            e1, substr = BL.get_estimation4CCSub()
            e2, substr_2 = BL.get_estimation(all_NDIP_dict=all_NDIP_dict)
            b2b.append(e1)
            b2b2.append(e2)
            b2substr.append(substr)
            b2substr2.append(substr_2)
            gmix = gmixes[i]
            SubSS = Sub_SS(compstr, gmix, temp=300.0, style=0, isScaled=self.Scale_Eform,
                           Scale_Eform=self.Scale_Eform, fine_tune=self.fine_tune)
            e, substr_2 = SubSS.get_estimation(all_NDIP_dict=all_NDIP_dict)
            subsss.append(SubSS.ediff2)
            subsubstr.append(SubSS.substrs_2)
            subsss2.append(SubSS.ediff_fine)
            subsubstr2.append(SubSS.substrs_fine)
            if i % 1000 == 0:
                print("Finished sublattice calculation of " + str(i) + " compositions!")
        thisdata.df["laves_substrs"] = lsubstr
        thisdata.df["bcc2laves"] = b2l
        thisdata.df["laves_substrs_2"] = lsubstr2
        thisdata.df["bcc2laves_2"] = b2l2
        thisdata.df["b2_substrs"] = b2substr
        thisdata.df["bcc2b2"] = b2b
        thisdata.df["b2_substrs_2"] = b2substr2
        thisdata.df["bcc2b2_2"] = b2b2
        thisdata.df["subss_substrs"] = subsubstr
        thisdata.df["bcc2subss"] = subsss
        thisdata.df["subss_substrs_2"] = subsubstr2
        thisdata.df["bcc2subss_2"] = subsss2
        
        thisdata.save_to(self.indict["OutFile"], df=thisdata.df)

    def get_nonT_models(self):
        keys = ["Vicker"]
        VH_featuress = [["sigma_y0", "Exp_Shear_DISTORT", "Exp_Youngs_ROM", "E_negativity", "volume_DISTORT"]]
        OutKey_dicts = [{"Vicker": "Vicker_M1"}]
        mname_headers = ["Vicker_M1"]
        modelname = "GBR"
        for ifrom in range(0, len(OutKey_dicts)):
            features = VH_featuress[ifrom]
            mname_header = mname_headers[ifrom]
            OutKey_dict = OutKey_dicts[ifrom]
            self.data_expansion(keys, features, OutKey_dict, modelname, mname_header)

        keys = ["Elongation_EXP"]
        #EL_featuress = [["ShearBurger_Surf", "volume_DELTA", "Exp_Shear_ROM", "Exp_Shear_DISTORT", "volume_DISTORT_Max"],
        #                ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"]]
        EL_featuress = [["ShearBurger", "Exp_Surf_ROM", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"],
                        ["E_negativity", "bcc2hcp", "Exp_Shear_DISTORT", "volume_DISTORT", "sigma_y0"]]
        OutKey_dicts = [{"Elongation_EXP": "Compress_M"}]
        mname_headers = ["Compression_MPEA"]
        modelname = "GBR"
        ifrom = 0
        for itype in range(0, len(EL_featuress)):
            features = EL_featuress[itype]
            mname_header = mname_headers[ifrom] + str(itype + 1)
            tmp_dict = OutKey_dicts[ifrom]
            OutKey_dict = {}
            OutKey_dict["Elongation_EXP"] = tmp_dict["Elongation_EXP"] + str(itype + 1)
            self.data_expansion(keys, features, OutKey_dict, modelname, mname_header)

        ### Tensile ductility models ###
        keys = ["Elongation_EXP"]
        #### tensile ductility models ####
        ### Data augmentation: set the RCCAs with Tm > 2600K and D > 5.5 in compressive database to zero.
        ## model 1-3: training database includes the 9 selected refractory metals
        ## model 4: training database inlcudes extended elements
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
            self.data_expansion(keys, features, OutKey_dict, modelname, mname_header)

    def merge_elongations(self, thres=thres4b2h, scale=False):
        thisdata = myData(self.indict["OutFile"], from_DATA=False, Source="INTERNAL")
        thisdata.df["Compress_M1"] = np.select(
            [thisdata.df["Compress_M1"] < 0.0, thisdata.df["Compress_M1"] >= 0.0],
            [0.0, thisdata.df["Compress_M1"]])

        if "Compress_M2" in thisdata.df.columns:
            thisdata.df["Compress_M2"] = np.select(
                [thisdata.df["Compress_M2"] < 0.0, thisdata.df["Compress_M2"] >= 0.0],
                [0.0, thisdata.df["Compress_M2"]])

        if "Compress_M3" in thisdata.df.columns:
            thisdata.df["Compress_M3"] = np.select(
                [thisdata.df["Compress_M3"] < 0.0, thisdata.df["Compress_M3"] >= 0.0],
                [0.0, thisdata.df["Compress_M3"]])

        thisdata.df["Elong_M1"] = np.select(
            [thisdata.df["Elong_M1"] < 0.0, thisdata.df["Elong_M1"] >= 0.0],
            [0.0, thisdata.df["Elong_M1"]])

        if "Elong_M2" in thisdata.df.columns:
            thisdata.df["Elong_M2"] = np.select(
            [thisdata.df["Elong_M2"] < 0.0, thisdata.df["Elong_M2"] >= 0.0],
            [0.0, thisdata.df["Elong_M2"]])

        if "Elong_M3" in thisdata.df.columns:
            thisdata.df["Elong_M3"] = np.select(
            [thisdata.df["Elong_M3"] < 0.0, thisdata.df["Elong_M3"] >= 0.0],
            [0.0, thisdata.df["Elong_M3"]])

        if "Elong_M4" in thisdata.df.columns:
            thisdata.df["Elong_M4"] = np.select(
            [thisdata.df["Elong_M4"] < 0.0, thisdata.df["Elong_M4"] >= 0.0],
            [0.0, thisdata.df["Elong_M4"]])

        if scale:
            if "Compress_M2" in thisdata.df.columns:
                xs = thisdata.df[b2h4ml].to_numpy()
                thisdata.df["Compress_M2"] = np.select(
                                           [xs < thres, xs >= thres],
                                           [thisdata.df["Compress_M2"], thisdata.df["Compress_M2"] / 10.0])
        thisdata.save_to(self.indict["OutFile"], df=thisdata.df)

    @staticmethod
    def normalization_YS(fname, ykey, thres=1.0):
        thisdata = myData(fname, from_DATA=False, Source="INTERNAL")
        ys = thisdata.df[ykey].to_numpy()
        xs = thisdata.df["temp_ratio"].to_numpy()
        thisdata.df[ykey] = np.select([xs < thres, xs >= thres], [ys, 0.0])
        thisdata.save_to(fname, df=thisdata.df)

    @staticmethod
    def compute_Tabor_yield(fname, ykey):
        thisdata = myData(fname, from_DATA=False, Source="INTERNAL")
        ys = thisdata.df[ykey].to_numpy()
        outkey = ykey.split("_")
        outkey = "Tabor_" + outkey[-1]
        thisdata.df[outkey] = ys * 3.0 / 1000.0
        thisdata.save_to(fname, df=thisdata.df)

    def get_T_models(self, FileSeparation=False):
        self.compute_temp_features(FileSeparation=FileSeparation)
        YS_featuress = [["temp_ratio", "delta_Eb0", "sigma_y0", "E_negativity",
                         "volume_DISTORT", "Exp_Shear_DISTORT", "Exp_Youngs_ROM"]]
        OutKey_dicts = [{"Yield_EXP": "Yield_M"}]
        mname_headers = ["Yield_MPEA"]
        keys = ["Yield_EXP"]
        modelname = "GBR"

        if FileSeparation:
            for i in range(len(self.indict["Temp_Level"])):
                t = self.indict["Temp_Level"][i]
                thisfname = self.indict["OutFile"].replace(".csv", "")
                thisfname += "_T" + str(int(t)) + ".csv"
                for ifrom in range(0, len(mname_headers)):
                    for itype in range(0, len(YS_featuress)):
                        features = YS_featuress[itype]
                        mname_header = mname_headers[ifrom] + str(itype + 1)
                        tmp_dict = OutKey_dicts[ifrom]
                        OutKey_dict = {}
                        OutKey_dict["Yield_EXP"] = tmp_dict["Yield_EXP"] + str(itype + 1)
                        self.data_expansion(keys, features, OutKey_dict, modelname, mname_header, input_fname=thisfname)
                        ExtendPredictor.normalization_YS(thisfname, OutKey_dict["Yield_EXP"], thres=1.0)
                        ExtendPredictor.compute_Tabor_yield(thisfname, OutKey_dict["Yield_EXP"])
        else:
            for ifrom in range(0, len(mname_headers)):
                for itype in range(0, len(YS_featuress)):
                    features = YS_featuress[itype]
                    mname_header = mname_headers[ifrom] + str(itype + 1)
                    tmp_dict = OutKey_dicts[ifrom]
                    OutKey_dict = {}
                    OutKey_dict["Yield_EXP"] = tmp_dict["Yield_EXP"] + str(itype + 1)
                    self.data_expansion(keys, features, OutKey_dict, modelname, mname_header,
                                        input_fname=self.indict["OutFile"])
                    ExtendPredictor.normalization_YS(self.indict["OutFile"], OutKey_dict["Yield_EXP"], thres=1.0)
                    ExtendPredictor.compute_Tabor_yield(self.indict["OutFile"], OutKey_dict["Yield_EXP"])

    def get_nonT_all(self, fill_transform=True):
        self.get_basic_properties(fill_transform=fill_transform)
        self.compute_extend_features()
        self.compute_sublatt_eform()
        self.get_nonT_models()
        self.merge_elongations(scale=False)

    def get_all(self, FileSeparation=False, fill_transform=True):
        self.get_nonT_all(fill_transform=fill_transform)
        self.get_T_models(FileSeparation=FileSeparation)


class Screener:
    def __init__(self, fname, from_database=False):
        self.fname = fname
        if from_database:
            self.df = pd.read_csv(os.path.join(DATA_PATH,fname))
        else:
            self.df = pd.read_csv(fname)

    def screen_elements(self, elements, style="EXCLUDE"):
        ys = self.df["Composition"].to_numpy()
        inds = np.arange(len(ys), dtype=int)
        bads = []
        for i in range(len(ys)):
            compstr = ys[i]
            comp = Composition(compstr)
            eles = []
            for el in comp.elements:
                eles.append(el.symbol)

            isValid = True
            if style[0:3].upper() == "EXA":
                if len(eles) != len(elements):
                    isValid = False
            if isValid:
                for sym in elements:
                    if sym in eles:
                        if style[0:3].upper() == "EXC":
                            isValid = False
                            break
                    else:
                        if style[0:3].upper() == "INC" or style[0:3].upper() == "EXA":
                            isValid = False
                            break

            if not isValid:
                bads.append(i)
        bads = np.array(bads).astype(int)
        inds = np.delete(inds, bads)
        self.df = self.df.iloc[inds]

    def screen_to_elements_basic(self):
        self.screen_elements(Element_Additional, style="EXCLUDE")

    def screen_compstrs(self, compstrs):
        rcs = []
        for compstr in compstrs:
            comp = Composition(compstr)
            rcs.append(comp.reduced_formula)
        rcs = np.array(rcs)
        self.df = self.df.set_index("ReducedFormula")
        self.df = self.df.loc[rcs]

    def screen_ncompon(self, ncompons):
        ys = self.df["Composition"].to_numpy()
        inds = np.arange(len(ys), dtype=int)
        bads = []
        for i in range(len(ys)):
            compstr = ys[i]
            isValid = True
            comp = Composition(compstr)
            thisn = len(comp.elements)
            if thisn in ncompons:
                pass
            else:
                isValid = False
            if not isValid:
                bads.append(i)
        bads = np.array(bads).astype(int)
        inds = np.delete(inds, bads)
        self.df = self.df.iloc[inds]

    def ratio_screener(self, key, rmin, rmax):
        ys = self.df[key].to_numpy()
        inds = np.arange(len(ys))
        maxv = np.max(ys)
        if isinstance(rmin, float) or isinstance(rmin, int):
            inds = np.compress(ys > rmin * maxv, inds)
            ys = ys[inds]
        if isinstance(rmax, float) or isinstance(rmax, int):
            inds = np.compress(ys < rmax * maxv, inds)
        self.df = self.df.iloc[inds]

    def value_screener(self, key, vmin, vmax):
        ys = self.df[key].to_numpy()
        inds = np.arange(len(ys))
        if isinstance(vmin, float) or isinstance(vmin, int):
            inds = np.compress(ys >= vmin, inds)
            ys = ys[inds]
        if isinstance(vmax, float) or isinstance(vmax, int):
            inds = np.compress(ys <= vmax, inds)
        self.df = self.df.iloc[inds]

    def percentage_screener(self, key, vmin, vmax):
        ys = self.df[key].to_numpy()
        maxv = np.max(ys)
        minv = np.min(ys)
        span = maxv - minv
        inds = np.arange(len(ys))
        if isinstance(vmin, float) or isinstance(vmin, int):
            inds = np.compress(ys >= vmin * span + minv, inds)
            ys = ys[inds]
        if isinstance(vmax, float) or isinstance(vmax, int):
            inds = np.compress(ys <= vmax * span + minv, inds)
        self.df = self.df.iloc[inds]

    def display_screener(self, sort_by=["Yield_M1"], DisplayCols=None, Index="Composition"):
        display_df(self.df, sort_by=sort_by, DisplayCols=DisplayCols, Index=Index)

    def save_screener(self, outfile=None, ShortCols=None):
        if outfile is None:
            outfile = self.fname
            outfile = outfile.split("/")
            outfile = outfile[-1]
            outfile = outfile.replace(".csv", "")
            outfile1 = outfile + "_Screen.csv"
            outfile2 = outfile + "_Screen_short.csv"
        else:
            outfile1 = outfile
            outfile2 = outfile.replace(".csv", "")
            outfile2 = outfile2 + "_short.csv"
        self.df.to_csv(outfile1, index=False, float_format=float_format)
        if ShortCols is None:
            self.df[shortcols].rename(columns=rename_dict).to_csv(outfile2, index=False, float_format=float_format)
        else:
            self.df[ShortCols].rename(columns=rename_dict).to_csv(outfile2, index=False, float_format=float_format)
