import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.core.composition import Composition
from myml.myglobal import Element_Basic, Element_Extend, Element_Add_Ext, config_vars, Constants
from myml.myglobal import compstr2concs, get_sorted_compspace, compstr2sconf
from myml.myelements.myelements import thermo_scale_keys
from myml.myelements.myelements import eos_keys, hcp_eos_keys, transformkey, elastic_keys, TC_keys, BOKAS_key
from myml.data.data import myData, DATA_PATH
from myml.data.data_util import compstr2eform_omegas, compstr2ROM
from myml.models.models import get_default_outfile
from myml.models.models import NDInterpolate, MLRegressor
from myml.plot.plot import plot_linear_xys

MODEL_PATH = config_vars["MODEL_PATH"]
float_format = Constants["float_format"]

def scaling_factor(n, from_ncompon=3):
    return (1.0 + 2.0 * (n - 1) / float(n) - 2.0 * (from_ncompon - 1) / float(from_ncompon))

def scale_eform(v, itype=0, reverse=False):
    ps = np.array([[-0.05517239, 0.41543859], [-0.0459761, 0.43046792]])
    p = ps[itype, :]
    if reverse:
        v = (v - p[0]) / p[1]
    else:
        v = p[0] + p[1] * v
    return v

def compstr2NDIP_keys(compstr, keys, style=1,
                      Scale_Bokas=True, Scale_Eform=False,
                      Exception_1="ROM", Exception_2="ZERO"):
    comp = Composition(compstr)
    ncompon = len(comp.elements)
    if ncompon == 1:
        return np.zeros(len(keys))
    else:
        rcompstr = comp.reduced_formula
        comp = Composition(rcompstr)
        compspace = []
        isExt = False
        for el in comp.elements:
            compspace.append(el.symbol)
            if style > 1:
                if el.symbol in Element_Extend: isExt = True
            else:
                if el.symbol in Element_Add_Ext: isExt = True
        if style > 1:
            from_ncompon = 2
        else:
            if isExt:
                from_ncompon = 2
            else:
                from_ncompon = min(3, ncompon)

        concs = compstr2concs(rcompstr, reduced=True)
        inds = np.arange(ncompon, dtype=int)
        combs = combinations(inds, from_ncompon)

        thisvlist = np.zeros(len(keys))
        thisctot = 0.0
        for comb in list(combs):
            thiscs = []
            thisc = 0.0
            thisij = 1.0
            thisc4eval = []
            substr = ""
            for i in range(from_ncompon):
                ii = comb[i]
                thiscs.append(compspace[ii])
                thisc4eval.append(concs[ii])
                thisc += concs[ii]
                thisij *= concs[ii]
                substr += compspace[ii] + str(int(round(concs[ii] * 100, 0)))
            thisc4eval = np.array(thisc4eval) / thisc
            thisctot += thisc
            if style > 1:
                fmodeldata = "bccC2_TC_300_IP.csv"
            else:
                if from_ncompon == 2:
                    fmodeldata = "bccC2_TC_300_IP.csv"
                else:
                    fmodeldata = "bccC3_TC_300_IP.csv"
            tmpdata = myData(fmodeldata, from_DATA=True, Source="EXTERNAL", Recompute=False)
            isCont = True
            for ikey in range(len(keys)):
                key = keys[ikey]
                thisIP = NDInterpolate(tmpdata, thiscs)
                isValid = thisIP.load_model(key)
                if not isValid:
                    out = 0.0
                    if Exception_1.upper() == "NAN" or Exception_1.upper() == "ZERO":
                        isCont = False
                        break
                    else:
                        if "Eform" in key or "Hmix_" in key or "Gmix_" in key:
                            try:
                                out = compstr2eform_omegas(substr, crystal="bcc", style=1)
                                if Scale_Bokas:
                                    if "BOKAS" in key:
                                        pass
                                    elif "Eform" in key:
                                        out = scale_eform(out, itype=0, reverse=True)
                            except:
                                print(f"CANNOT find formation database for compstr: {substr}!")
                                out = 0.0
                                if Exception_2.upper() == "NAN" or Exception_2.upper() == "ZERO":
                                    isCont = False
                                    break
                        else:
                            try:
                                out = compstr2ROM(substr, key)
                            except:
                                print(f"CANNOT find ROM {key} database for compstr: {substr}!")
                                out = 0.0
                                if Exception_2.upper() == "NAN" or Exception_2.upper() == "ZERO":
                                    isCont = False
                                    break
                else:
                    out = thisIP.Interpolate_data(thisc4eval, key)
                    out = out[0]

                if Scale_Eform:
                    if key in thermo_scale_keys:
                        out *= scaling_factor(ncompon, from_ncompon=from_ncompon)
                thisvlist[ikey] += out * thisc

            if not isCont:
                thisvlist = np.zeros(len(keys))
                break

        thisvlist /= (thisctot + 1.0e-20)
        return thisvlist


def load_all_NDIP_models(keys=None):
    all_NDIP_dict = {}
    SAVE_PATH = "NDIP_Models"
    for fname in os.listdir(os.path.join(MODEL_PATH, SAVE_PATH)):
        if "bcc" in fname:
            if keys is None:
                with open(os.path.join(MODEL_PATH, SAVE_PATH, fname), 'rb') as f:
                    all_NDIP_dict[fname] = pickle.load(f)
            else:
                for key in keys:
                    if key in fname:
                        with open(os.path.join(MODEL_PATH, SAVE_PATH, fname), 'rb') as f:
                            all_NDIP_dict[fname] = pickle.load(f)
    return all_NDIP_dict


def compstr2all_NDIP_keys(compstr, keys, all_NDIP_dict,
                          style=1, Scale_Bokas=True, Scale_Eform=False,
                          Exception_1="ROM", Exception_2="ZERO"):
    comp = Composition(compstr)
    ncompon = len(comp.elements)
    if ncompon == 1:
        return np.zeros(len(keys))
    else:
        rcompstr = comp.reduced_formula
        comp = Composition(rcompstr)
        compspace = []
        isExt = False
        for el in comp.elements:
            compspace.append(el.symbol)
            if style > 1:
                if el.symbol in Element_Extend: isExt = True
            else:
                if el.symbol in Element_Add_Ext: isExt = True
        if style > 1:
            from_ncompon = 2
        else:
            if isExt:
                from_ncompon = 2
            else:
                from_ncompon = min(3, ncompon)

        concs = compstr2concs(rcompstr, reduced=True)
        inds = np.arange(ncompon, dtype=int)
        combs = combinations(inds, from_ncompon)

        thisvlist = np.zeros(len(keys))
        thisctot = 0.0
        for comb in list(combs):
            thiscs = []
            thisc = 0.0
            thisij = 1.0
            thisc4eval = []
            substr = ""
            for i in range(from_ncompon):
                ii = comb[i]
                thiscs.append(compspace[ii])
                thisc4eval.append(concs[ii])
                thisc += concs[ii]
                thisij *= concs[ii]
                substr += compspace[ii] + str(int(round(concs[ii] * 100, 0)))
            thisc4eval = np.array(thisc4eval) / thisc
            thisctot += thisc

            thiscs = get_sorted_compspace(thiscs)
            csstr = ""
            for sym in thiscs:
                csstr += sym
            mname_header = "bcc" + "_" + csstr + "_"

            isCont = True
            for ikey in range(len(keys)):
                key = keys[ikey]
                modelkey = mname_header + key
                isValid = True
                if modelkey in all_NDIP_dict:
                    if len(thisc4eval.shape) == 1: thisc4eval = np.array([list(thisc4eval)])
                    thisxs = thisc4eval[:, 1:from_ncompon]
                    if from_ncompon == 2:
                        out = all_NDIP_dict[modelkey](thisxs.flatten())
                    else:
                        out = all_NDIP_dict[modelkey](thisxs)
                    out = out[0]
                else:
                    isValid = False

                if not isValid:
                    out = 0.0
                    if Exception_1.upper() == "NAN" or Exception_1.upper() == "ZERO":
                        isCont = False
                        break
                    else:
                        if "Eform" in key or "Hmix_" in key or "Gmix_" in key:
                            try:
                                out = compstr2eform_omegas(substr, crystal="bcc", style=1)
                                if Scale_Bokas:
                                    if "BOKAS" in key:
                                        pass
                                    elif "Eform" in key:
                                        out = scale_eform(out, itype=0, reverse=True)
                            except:
                                print(f"CANNOT find formation database for compstr: {substr}!")
                                out = 0.0
                                if Exception_2.upper() == "NAN" or Exception_2.upper() == "ZERO":
                                    isCont = False
                                    break
                        else:
                            try:
                                out = compstr2ROM(substr, key)
                            except:
                                print(f"CANNOT find ROM {key} database for compstr: {substr}!")
                                out = 0.0
                                if Exception_2.upper() == "NAN" or Exception_2.upper() == "ZERO":
                                    isCont = False
                                    break
                if Scale_Eform:
                    if key in thermo_scale_keys:
                        out *= scaling_factor(ncompon, from_ncompon=from_ncompon)

                thisvlist[ikey] += out * thisc

            if not isCont:
                thisvlist = np.zeros(len(keys))
                break
        thisvlist /= (thisctot + 1.0e-20)
        return thisvlist


class DataExpansion:
    def __init__(self, data, keys, from_ncompon,
                 modelname="NDIP", mname_header=None,
                 features=None, elements=None, normalization=False, style4Eform="NDIP",
                 Scale_Eform=False):
        self.SAVE_PATH = "DATA_EXP"

        self.data = data
        self.keys = keys
        self.columns = data.df.columns.tolist()
        compstrs = data.df["Composition"].tolist()
        self.ncompstr = len(compstrs)
        comp = Composition(compstrs[0])
        self.compstrs = compstrs

        self.latname = self.data.df.iloc[0]["Crystal"]
        self.from_ncompon = from_ncompon

        if "HGBR" in modelname:
            modelname = "HGBR"
        elif "GBR" in modelname:
            modelname = "GBR"
        elif "MLPR" in modelname:
            modelname = "MLPR"
        else:
            modelname = "NDIP"
        self.modelname = modelname
        self.mname_header = mname_header

        self.features = features  #for vickers and yield strength u need input features

        if elements is None:
            if features is None:
                elements = Element_Basic[0:len(Element_Basic)]
            else:
                elements = data.elements[0:len(data.elements)]
        self.elements = elements

        self.normalization = normalization
        self.style4Eform = style4Eform
        self.Scale_Eform = Scale_Eform

        outfile = get_default_outfile(self.data.fname)
        if self.modelname == "NDIP":
            outfile = outfile + "_IP.csv"
        else:
            outfile = outfile + "_" + self.modelname + ".csv"
        self.default_outfile = outfile

    def fill_dataframe_MLR(self, Add_Predict=True, OutKey_dict=None):
        thisMLR = MLRegressor(self.data, self.keys, modelname=self.modelname,
                              mname_header=self.mname_header, features=self.features,
                              elements=self.elements, normalization=self.normalization, SHAP_Plot=False)
        X_test_this = thisMLR.generate_X()
        for key in self.keys:
            if isinstance(OutKey_dict, dict) and key in OutKey_dict:
                outkey = OutKey_dict[key]
            else:
                outkey = key
                if Add_Predict: outkey = "Predicted_" + outkey
            mname = key + thisMLR.keyapps[0]
            isValid = thisMLR.load_model(key)
            errmsg = f"MLR model for {key} is not existed."
            if not isValid: raise ValueError(errmsg)
            outs = thisMLR.get_predictions(key, X_test_this)
            self.data.df[outkey] = outs

    def fill_data(self, style=1, Add_Predict=True, Save_Pred2Data=True, OutKey_dict=None, Save_OrgData=True):
        if self.modelname == "NDIP":
            ifill = 0
            outkeys = self.keys[0:len(self.keys)]
            for ikey in range(len(self.keys)):
                key = self.keys[ikey]
                if isinstance(OutKey_dict, dict) and key in OutKey_dict:
                    outkeys[ikey] = OutKey_dict[key]
                else:
                    if Add_Predict: outkeys[ikey] = "Predicted_" + outkeys[ikey]
            outlist = []
            for istr in range(len(self.data.df)):
                compstr = self.data.df.iloc[istr]["Composition"]
                thisvlist = compstr2NDIP_keys(compstr, self.keys, style=style,
                                              Scale_Bokas=True, Scale_Eform=self.Scale_Eform,
                                              Exception_1="ROM", Exception_2="ZERO")
                outlist.append(thisvlist)

                if ifill % 1000 == 0:
                    print(f"Finished {ifill} compstrs!")
                ifill += 1

            outlist = np.array(outlist)
            outlist = outlist.T
            for ikey in range(len(outkeys)):
                key = outkeys[ikey]
                self.data.df[key] = outlist[ikey]
        else:
            self.fill_dataframe_MLR(Add_Predict=Add_Predict, OutKey_dict=OutKey_dict)

        if Save_Pred2Data:
            if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
                os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))

            outfile = os.path.join(self.SAVE_PATH, self.default_outfile)
            self.data.df.to_csv(outfile, index=True, float_format=float_format)
        if Save_OrgData:
            self.data.df.to_csv(self.data.fname, index=True, float_format=float_format)


def Initialization_df_from_compstrs(columns, compstrs, latname, subID=0, Temp=300.0):
    columns = list(columns)
    if "CompID" in columns:
        pass
    else:
        columns = ["CompID"] + columns
    ncol = len(columns)
    nrow = len(compstrs)
    tdf = pd.DataFrame(np.zeros([nrow, ncol]), columns=columns)

    tdf["Composition"] = compstrs
    tdf["Crystal"] = [latname] * nrow
    compids = []
    sconfs = []
    rfs = []
    for i in range(nrow):
        compstr = compstrs[i]
        comp = Composition(compstr)
        rcompstr = comp.reduced_formula
        comp = Composition(rcompstr)
        compid = i
        compids.append(compid)
        sconf = compstr2sconf(rcompstr)
        sconfs.append(sconf)
        rfs.append(rcompstr)

    tdf["CompID"] = compids
    tdf["Sconf"] = sconfs
    tdf["ReducedFormula"] = rfs
    return tdf


class CSExpansion:
    def __init__(self, compstrs, data, modelname="NDIP", mode="eos", keys=None, elements=None,
                 normalization=False, fill_transform=False, subID=0, Temp=300, style4Eform="NDIP",
                 Scale_Eform=False):
        self.SAVE_PATH = "CS_EXPANSION"

        self.data = data
        self.columns = data.df.columns.tolist()
        self.compstrs = compstrs
        self.ncompstr = len(compstrs)
        comp = Composition(compstrs[0])

        self.ncompon = len(comp.elements)
        self.latname = self.data.df.iloc[0]["Crystal"]
        if "HGBR" in modelname:
            modelname = "HGBR"
        elif "GBR" in modelname:
            modelname = "GBR"
        elif "MLPR" in modelname:
            modelname = "MLPR"
        else:
            modelname = "NDIP"
        self.modelname = modelname
        if mode[0:3] == "ela":
            self.mode = "elastic"
        else:
            self.mode = "eos"
        if self.mode == "eos":
            self.undotype = 1
        else:
            self.undotype = 3

        self.fill_transform = fill_transform
        self.all_keys = eos_keys[0:len(eos_keys)]
        if "Hmix_TC" in data.df.columns.tolist():
            self.all_keys += TC_keys
        if "Eform_BOKAS" in data.df.columns.tolist():
            self.all_keys += [BOKAS_key]

        if self.latname == "hcp":
            self.all_keys += hcp_eos_keys
        elif self.latname == "bcc":
            if self.fill_transform: self.all_keys += [transformkey]
        if self.mode == "elastic":
            self.all_keys += elastic_keys

        if keys is None:
            self.keys = self.all_keys[0:len(self.all_keys)]
        elif isinstance(keys, str):
            if keys in self.all_keys:
                self.keys = [keys]
            else:
                raise ValueError(keys + " is not valid!")
        elif isinstance(keys, list) or isinstance(keys, np.ndarray):
            self.keys = []
            for key in keys:
                if key in self.all_keys: self.keys.append(key)
        self.style4Eform = style4Eform
        self.Scale_Eform = Scale_Eform
        if elements is None:
            elements = data.elements[0:len(data.elements)]
        self.elements = elements
        self.normalization = normalization
        compstr = self.data.df.iloc[0]["Composition"]
        comp = Composition(compstr)
        self.from_ncompon = len(comp.elements)
        self.default_outfile = self.latname + "C" + str(self.ncompon) + "_from_" + self.modelname + ".csv"
        self.default_Xlabel = "From CS Expansion"
        self.default_Ylabel = "Test Data"
        self.default_Title = "Evaluation of CS Expansion_"
        self.outfig_header = self.latname + "C" + str(self.ncompon) + "_from_" + self.modelname + "_"

        tdf = Initialization_df_from_compstrs(self.columns, self.compstrs, self.latname, subID=subID, Temp=Temp)
        tdf.to_csv(self.default_outfile, index=False, float_format=float_format)
        self.outdata = myData(self.default_outfile, from_DATA=False, Source="INTERNAL")
        os.remove(self.default_outfile)

    def fill_dataframe_MLR(self, loadmodel=False, savemodel=False):
        X_test_this = self.outdata.get_concs(self.outdata.df)

        thisMLR = MLRegressor(self.data, self.keys, modelname=self.modelname,
                              mname_header=None, features=None, elements=self.elements,
                              normalization=self.normalization, SHAP_Plot=False)
        for key in self.keys:
            mname = key + thisMLR.keyapps[0]
            if loadmodel:
                isValid = thisMLR.load_model(key)
            else:
                isValid = False
            if not isValid:
                X_train, X_test, y_train, y_test, X, y = thisMLR.generate_data(key, random_state=None)
                thisMLR.get_regression_model(X_train, y_train, key, savemodel=savemodel)
            #thismodel = thisMLR.models[mname]
            outs = thisMLR.get_predictions(key, X_test_this)
            self.outdata.df[key] = outs

    def fill_data(self, style=1, fname4update_trans=None, loadmodel=False, savemodel=False,
                  savefile=False, outfile=None, Load_all_NDIP=False):
        if self.modelname == "NDIP":
            ifill = 0
            cols = self.outdata.df.columns.tolist()
            if Load_all_NDIP: all_NDIP_dict = load_all_NDIP_models()
            outlist = []
            for istr in range(len(self.outdata.df)):
                compstr = self.outdata.df.iloc[istr]["Composition"]
                if Load_all_NDIP:
                    thisvlist = compstr2all_NDIP_keys(compstr, self.keys, all_NDIP_dict,
                                                      style=style, Scale_Bokas=True, Scale_Eform=self.Scale_Eform,
                                                      Exception_1="ROM", Exception_2="ZERO")
                else:
                    thisvlist = compstr2NDIP_keys(compstr, self.keys, style=style,
                                                  Scale_Bokas=True, Scale_Eform=self.Scale_Eform,
                                                  Exception_1="ROM", Exception_2="ZERO")
                outlist.append(thisvlist)
                if ifill % 1000 == 0:
                    print(f"Finished {ifill} compstrs!")
                ifill += 1
            outlist = np.array(outlist)
            outlist = outlist.T

            for ikey in range(len(self.keys)):
                self.outdata.df[self.keys[ikey]] = outlist[ikey]
        else:
            self.fill_dataframe_MLR(loadmodel=loadmodel, savemodel=savemodel)

        self.outdata.df = self.outdata.compute_lattice_parameter(savefile=False)
        #self.outdata.df = self.outdata.update_gmix_cpa(Temp=300.0, savefile=False)

        if not self.fill_transform:
            if self.latname == "bcc":
                if fname4update_trans is None:
                    fname4update_trans = self.default_outfile.replace("bcc", "hcp")
                    fname4update_trans = os.path.join(DATA_PATH, fname4update_trans)
                self.outdata.update_transform_from(fname4update_trans, from_DATA=False, savefile=False)
        if self.mode == "elastic":
            if self.latname == "bcc":
                self.outdata.df = self.outdata.compute_elastic_props(savefile=False)

        if savefile:
            if outfile is None:
                if not os.path.isdir(os.path.join(os.getcwd(), self.SAVE_PATH)):
                    os.mkdir(os.path.join(os.getcwd(), self.SAVE_PATH))
                outfile = os.path.join(self.SAVE_PATH, self.default_outfile)
            if self.outdata.df.index.name == "CompID":
                self.outdata.df.to_csv(outfile, index=True, float_format=float_format)
            else:
                self.outdata.df.to_csv(outfile, index=False, float_format=float_format)
            self.outdata = myData(outfile, from_DATA=False, Source="INTERNAL", Recompute=True)
            self.outdata.compute_Hardness_features(style="Exp")
            self.outdata.save_to(outfile, df=self.outdata.df)

    def evaluation(self, df2, keys, savefig=False, outfig_header=None):
        outfig_header = os.path.join(self.SAVE_PATH, self.outfig_header)
        xdf = self.outdata.df.copy()
        ydf = df2.copy()
        nx = len(xdf)
        ny = len(df2)
        xdf = xdf.reset_index().set_index("Composition")
        ydf = ydf.reset_index().set_index("Composition")
        if nx > ny:
            inds = ydf.index.to_numpy()
            xdf = xdf.loc[inds]
        else:
            inds = xdf.index.to_numpy()
            ydf = ydf.loc[inds]

        for key in keys:
            x = xdf[key].to_numpy()
            y = ydf[key].to_numpy()
            fig = plot_linear_xys(x, y, colors=None, style="scatter",
                                  Xlabel=self.default_Xlabel, Ylabel=self.default_Ylabel,
                                  Title=self.default_Title + "_" + key,
                                  compute_R2=True, Label_R2=True, plot_xx=True, ShowColorbar=False)
            if savefig:
                if outfig_header is None: outfile = "Comparison_"
                outfile = outfig_header + key + ".png"
                plt.savefig(outfile, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
        xdf = None
        ydf = None
