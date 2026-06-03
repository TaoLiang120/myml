import copy
import os
from itertools import combinations

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

from myml.myglobal import Element_external_crystal, Element_external
from myml.myglobal import config_vars, Constants
from myml.myglobal import Basic_ROM_Keys, Add_ROM_Keys, DELTA_Keys, DISTORT_Keys, SCALED_Keys
from myml.myelements.myelements import ndip_all_keys

from myml.data.data_util import compstr2ROM, compstr2DELTA, compstr2DISTORT, compstr2YS_Params

from myml.data.data_util import compute_YS_Params_from_2pts, compute_YS_from_2params
from myml.data.data_util import compute_df_reducedformula, compute_df_concs, compute_df_sconfig, compute_df_radius_sws
from myml.data.data_util import compute_df_volume, compute_df_density, compute_df_composite_keys
from myml.data.data_util import compute_df_ElasticEnergy, compute_df_SH_ratio, compute_df_temp_ratio
from myml.data.data_util import compute_df_YS_Params, compute_df_YieldStrength, get_df_pretty_formula

# from pymatgen.util.string import latexify, htmlify
# from pymatgen.analysis.phase_diagram import PDEntry
DATA_PATH = config_vars["DATA_PATH"]
KJ2eVA = Constants["KJ2eVA"]
float_format = Constants["float_format"]

rename_dict_Born = {}
rename_dict_Born["FORMULA"] = "Composition"
rename_dict_Born["PROPERTY: Test temperature ($^\circ$C)"] = "temperature"
rename_dict_Born["PROPERTY: Type of test"] = "Tension0_Compression1"
rename_dict_Born["PROPERTY: Microstructure"] = "Crystal"
rename_dict_Born["PROPERTY: Processing method"] = "Processing"
rename_dict_Born["PROPERTY: Elongation (%)"] = "Elongation_EXP"
rename_dict_Born["PROPERTY: Elongation plastic (%)"] = "Elongation_plastic_EXP"
rename_dict_Born["PROPERTY: YS (MPa)"] = "Yield_EXP"
rename_dict_Born["PROPERTY: UTS (MPa)"] = "UTS_EXP"
rename_dict_Born["PROPERTY: Exp. Young modulus (GPa)"] = "Youngs_EXP"
rename_dict_Born["PROPERTY: HV"] = "Vicker"
rename_dict_Born["PROPERTY: grain size ($\mu$m)"] = "GrainSize"
rename_dict_Born["REFERENCE: doi"] = "DOI"
rename_dict_Born["REFERENCE: year"] = "YEAR"
rename_dict_Born["REFERENCE: title"] = "TITLE"
rename_dict_Born["PROPERTY: Calculated Young modulus (GPa)"]= "Youngs_ROM_ORG"


rename_dict_JT = {}
rename_dict_JT["FORMULA"] = "Composition"
rename_dict_JT["Test temperature (C)"] = "temperature"
rename_dict_JT["Type of test"] = "Tension0_Compression1"
rename_dict_JT["Microstructure"] = "Crystal"
rename_dict_JT["Processing method"] = "Processing"
rename_dict_JT["Elongation (%)"] = "Elongation_EXP"
rename_dict_JT["Plastic Elongation (%)"] = "Elongation_plastic_EXP"
rename_dict_JT["YS (MPa)"] = "Yield_EXP"
rename_dict_JT["UTS (MPa)"] = "UTS_EXP"
rename_dict_JT["HV"] = "Vicker"
rename_dict_JT["grain size"] = "GrainSize"
rename_dict_JT["doi"] = "DOI"
rename_dict_JT["year"] = "YEAR"
rename_dict_JT["title"] = "TITLE"
rename_dict_JT["journal"] = "JOURNAL"
rename_dict_JT["comments"] = "COMMENTS"

STRING_KEYS = ["Composition", "Crystal", "Processing", "DOI", "YEAR", "TITLE"]

def compute_df_NDIP_keys(df, keys, outkeys=None, style=1, Exception_1="ROM", Exception_2="ZERO",
                         Scale_Eform=False, recompute=True):
    from myml.expansion.expansion import load_all_NDIP_models, compstr2all_NDIP_keys
    if outkeys is None:
        outkeys = copy.deepcopy(keys)

    compstrs = df["Composition"].to_numpy()
    all_NDIP_dict = load_all_NDIP_models()
    vlist = []
    for i in range(len(compstrs)):
        compstr = compstrs[i]
        vs = compstr2all_NDIP_keys(compstr, keys, all_NDIP_dict,
                                   style=style, Scale_Bokas=True, Scale_Eform=Scale_Eform,
                                   Exception_1=Exception_1, Exception_2=Exception_2)
        vlist.append(vs)
    vlist = np.array(vlist)
    vlist = vlist.T
    for ikey in range(len(keys)):
        outkey = outkeys[ikey]
        if recompute:
            df[outkey] = vlist[ikey, :]
        else:
            if outkey in df.columns:
                pass
            else:
                df[outkey] = vlist[ikey, :]
    return df

class DataReorganizer:
    def __init__(self, fname, Scale_Eform=False, normalization=True):
        SAVE_PATH = "EXTERNAL_DATABASE"
        if not os.path.isdir(os.path.join(os.getcwd(), SAVE_PATH)):
            os.mkdir(os.path.join(os.getcwd(), SAVE_PATH))
        self.SAVE_PATH = SAVE_PATH
        self.fname = fname
        self.Scale_Eform = Scale_Eform
        self.normalization = normalization
        self.df = pd.read_csv(self.fname, engine='python')
        self.filehead = self.fname.replace(".csv", "")
        self.outfile = self.filehead + "_EXT.csv"
        self.orgfile = self.filehead +"_EXT_ORG.csv"

    def save_to(self, outfile=None):
        if not isinstance(outfile, str):
            outfile = self.outfile
        fname = os.path.join(self.SAVE_PATH, outfile)
        self.df.to_csv(fname, index=False, float_format=float_format)

    def df_drop_columns(self, rename_dict):
        cols = self.df.columns.tolist()
        keepcols = []
        for key in rename_dict:
            keepcols.append(rename_dict[key])
        dropcols = []
        for key in cols:
            if key not in keepcols: dropcols.append(key)
        self.df = self.df.drop(dropcols, axis=1)

    def delete_duplicate(self, df, df_multi=None, key="Yield_EXP"):
        '''
        # delete cuplication within one file
        Args:
            df: dataframe
            df_multi: None, df_multi means one composition have multiple measurements (T-dep YS)
                      in this case, df often refers to the composition with single measurement.
            key: key
        Returns:
        '''
        rcompstrs = df["ReducedFormula"].to_numpy()
        if df_multi is None:
            pass
        else:
            rcs = df_multi["ReducedFormula"].tolist()
            rcs = list(set(rcs))
            goods = []
            bads = []
            for i in range(len(rcompstrs)):
                c = rcompstrs[i]
                if c in rcs:
                    bads.append(i)
                else:
                    goods.append(i)
            bads = np.array(bads)
            goods = np.array(goods)
            df_bad = df.iloc[bads]
            df = df.iloc[goods]

        nbefore = len(df)
        rcompstrs = df["ReducedFormula"].to_numpy()
        a, inds, rinds = np.unique(rcompstrs, return_index=True, return_inverse=True)
        ys = df[key].to_numpy()
        cols = df.columns.tolist()
        icol = cols.index(key)
        print("=== Duplication in one file ===")
        for i in range(len(inds)):
            ind = inds[i]
            rind = np.where(rinds == i)
            rind = rind[0]
            n = len(rind)
            if n > 0:
                avg = 0.0
                for j in range(n):
                    jj = rind[j]
                    avg += ys[jj]
                    if n > 1:
                        print(f"composition:{rcompstrs[jj]} nentries:{n} thisentry:{j} values:{ys[jj]}")
                avg /= n
                df.iloc[ind, icol] = avg
                if n > 1:
                    print(f"=== end of composition:{rcompstrs[jj]} final values:{avg} ===")
        print("=== End of duplication in one file ===")
        df = df.iloc[inds]
        print(f"Length before remove duplicate:{nbefore} after:{len(df)}")
        return df

    def YieldStrength_Augment(self, df, keys, YSkey="Yield_EXP", a=0.55, strain_ratio=1e7, multipler=1):
        def get_augment_parameters(T2, Y2, Tm, a=0.55, strain_ratio=1e7):
            T4screen = 800.0
            Tmin = 800.0
            Tmax = 2000.0
            n = len(T2)
            isValid = True
            inds = np.arange(n, dtype=int)
            TC = np.array(T2[0:n])
            YC = np.array(Y2[0:n])
            inds = np.compress(TC > T4screen, inds)
            TC = TC[inds]
            YC = YC[inds]
            nc = len(TC)
            if nc < 2:
                isValid = False
            else:
                if TC[nc - 1] - TC[0] < 100.0:
                    isValid = False
                else:
                    slope = (YC[0] - YC[nc - 1]) / (TC[nc - 1] - TC[0])
                    if slope < 0.1: isValid = False
            nT = 5
            incr = (Tmax - Tmin) / (nT - 1)
            Temps = np.linspace(Tmin, Tmax, nT)
            if isValid:
                T2max = TC[nc - 1]
                istart = int(round((T2max - Tmin) / incr))
                if istart < 0: istart = 0
                if istart >= len(Temps): istart = len(Temps)
                iend = int(round((Tm - Tmin) / incr))
                if iend >= len(Temps): iend = len(Temps)
                if iend <= istart: iend = istart
                tl = Temps[istart:iend]
            else:
                tl = np.array([])
            if len(tl) > 0:
                T1 = TC[0]
                TF = TC[nc - 1]
                Y1 = YC[0]
                YF = YC[nc - 1]
                y0, Eb0 = compute_YS_Params_from_2pts(T1, TF, Y1, YF, a=a, strain_ratio=strain_ratio)
            else:
                y0 = 0.0
                Eb0 = 0.0
                isValid = False
            return y0, Eb0, tl, isValid

        def expand_df(df, indict, T, Y, multipler, Tmin=800, style=1, YSkey="Yield_EXP"):
            Ts = np.array(T)
            Ys = np.array(Y)
            if style == 1:
                inds = np.arange(len(Ts), dtype=int)
                inds = np.compress(Ts >= Tmin, inds)
                Ts = Ts[inds]
                Ys = Ys[inds]
                multipler -= 1

            for jj in range(multipler):
                for ii in range(len(Ys)):
                    thisdict = {}
                    for key in indict:
                        thisdict[key] = indict[key]
                    thisdict["temperature"] = Ts[ii]
                    thisdict[YSkey] = Ys[ii]
                    df.loc[len(df)] = thisdict
            return df

        #############################################
        temps = df["temperature"].to_numpy()
        ys = df[YSkey].to_numpy()
        compstrs = df["Composition"].to_numpy()
        tms = df["Tm_ROM"].to_numpy()
        vlast = [""] * len(keys)
        nbefore = len(df)

        inds = np.arange(len(df), dtype=int)
        df = df.reset_index()
        df["tmpid"] = inds
        df = df.set_index("tmpid")

        T2 = []
        Y2 = []
        NCOMP = 0
        for i in range(len(ys)):
            isIden = True
            for j in range(len(keys)):
                v = df.at[i, keys[j]]
                if v != vlast[j]:
                    isIden = False
                    break
            if isIden:
                T2.append(temps[i])
                Y2.append(ys[i])
            else:
                counts = len(T2)
                tm = tms[i - 1]
                if counts >= 2:
                    y0, Eb0, Ts, isValid = get_augment_parameters(T2, Y2, tm, a=a, strain_ratio=strain_ratio)
                    if isValid:
                        indict = df.loc[i - 1].to_dict()
                        cys = compute_YS_from_2params(Ts, y0, Eb0, style=1, a=a, strain_ratio=strain_ratio)
                        cys = np.select([Ts < 0.9 * tm, Ts >= 0.9 * tm], [cys, 1.0])
                        acys = np.compress(Ts < 0.9 * tm, cys)
                        if len(acys) > 0: NCOMP += 1
                        df = expand_df(df, indict, T2, Y2, multipler, Tmin=800, style=1, YSkey=YSkey)
                        df = expand_df(df, indict, Ts, cys, multipler, Tmin=800, style=2, YSkey=YSkey)

                for j in range(len(keys)):
                    vlast[j] = df.at[i, keys[j]]
                T2 = []
                Y2 = []
                T2.append(temps[i])
                Y2.append(ys[i])

        if len(T2) >= 2:
            tm = tms[i]
            y0, Eb0, Ts, isValid = get_augment_parameters(T2, Y2, tm, a=a, strain_ratio=strain_ratio)
            if isValid:
                indict = df.loc[i].to_dict()
                cys = compute_YS_from_2params(Ts, y0, Eb0, style=1, a=a, strain_ratio=strain_ratio)
                cys = np.select([Ts < 0.9 * tm, Ts >= 0.9 * tm], [cys, 1.0])
                acys = np.compress(Ts < 0.9 * tm, cys)
                if len(acys) > 0: NCOMP += 1
                df = expand_df(df, indict, T2, Y2, multipler, Tmin=800, style=1, YSkey=YSkey)
                df = expand_df(df, indict, Ts, cys, multipler, Tmin=800, style=2, YSkey=YSkey)

        df = df.sort_values(keys, ignore_index=True)
        print(f"Number of compositions: {NCOMP} Length before YS temp augment:{nbefore} after:{len(df)}")
        return df

    def get_length_unique_system(self, df):
        rcompstrs = df["ReducedFormula"].tolist()
        rcompstrs = list(set(rcompstrs))
        n = len(rcompstrs)
        return n

    def reorganizer1(self):
        dropcols = ["No.", "material", "best", "y_cs", "y_gap", "y_den", "y_csbg", "y_csden", "y_bgden"]
        self.df = self.df.drop(dropcols, axis=1)
        rename_dict = {}
        rename_dict["Formula"] = "Composition"
        rename_dict["CrystalSystem"] = "Crystal"
        rename_dict["bulk"] = "Bulk_EXP"
        rename_dict["shear"] = "Shear_EXP"
        rename_dict["young"] = "Youngs_EXP"
        rename_dict["poisson"] = "Poisson_EXP"
        rename_dict["Hexp"] = "Vicker"
        self.df = self.df.rename(columns=rename_dict)
        compstrs = self.df["Composition"].tolist()

        inds = np.arange(len(self.df), dtype=int)
        bads = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            if "(" in compstr:
                bads.append(i)
            else:
                try:
                    comp = Composition(compstr)
                    if not comp.valid: bads.append(i)
                except:
                    bads.append(i)
        bads = np.array(bads)
        goods = np.delete(inds, bads)
        self.df = self.df.iloc[goods]
        self.df = compute_df_reducedformula(self.df)

    def compute_standardard_features(self, thisdf, style=1, Exception_1="ROM", Exception_2="ZERO",
                                     outkeys=None, recompute=True):
        thisdf = compute_df_reducedformula(thisdf)
        thisdf = compute_df_concs(thisdf, elements=None)
        if "temperature" not in thisdf.columns: thisdf["temperature"] = [300.0] * len(thisdf)

        thisdf = compute_df_sconfig(thisdf)
        compstrs = thisdf["Composition"].tolist()
        keys = ndip_all_keys
        thisdf = compute_df_NDIP_keys(thisdf, keys, outkeys=outkeys, style=style,
                                      Exception_1=Exception_1, Exception_2=Exception_2,
                                      Scale_Eform=self.Scale_Eform, recompute=recompute)


        if "radius" not in thisdf.columns: thisdf = compute_df_radius_sws(thisdf)
        if "Sconf" not in thisdf.columns: thisdf = compute_df_sconfig(thisdf)

        for ikey in range(len(Basic_ROM_Keys)):
            key = Basic_ROM_Keys[ikey]
            outkey = key
            if outkey not in thisdf.columns:
                vlist = []
                for i in range(len(thisdf)):
                    compstr = compstrs[i]
                    v = compstr2ROM(compstr, key)
                    vlist.append(v)
                thisdf[outkey] = vlist

        if "volume" not in thisdf.columns: thisdf = compute_df_volume(thisdf, latname="mix")

        for ikey in range(len(Add_ROM_Keys)):
            key = Add_ROM_Keys[ikey]
            outkey = key + "_ROM"
            if outkey not in thisdf.columns:
                vlist = []
                for i in range(len(thisdf)):
                    compstr = compstrs[i]
                    v = compstr2ROM(compstr, key)
                    vlist.append(v)
                thisdf[outkey] = vlist

        thisdf = compute_df_density(thisdf)

        for ikey in range(len(DELTA_Keys)):
            key = DELTA_Keys[ikey]
            outkey = key + "_DELTA"
            if outkey not in thisdf.columns:
                vlist = []
                for i in range(len(thisdf)):
                    compstr = compstrs[i]
                    refv = thisdf.iloc[i][key]
                    v = compstr2DELTA(compstr, key, refv=refv, style=1)
                    vlist.append(v)
                thisdf[outkey] = vlist

        for ikey in range(len(DISTORT_Keys)):
            key = DISTORT_Keys[ikey]
            outkey1 = key + "_DISTORT"
            outkey2 = key + "_DISTORT_Max"
            if outkey1 not in thisdf.columns:
                v1list = []
                v2list = []
                for i in range(len(thisdf)):
                    compstr = compstrs[i]
                    v1, v2 = compstr2DISTORT(compstr, key)
                    v1list.append(v1)
                    v2list.append(v2)
                thisdf[outkey1] = v1list
                thisdf[outkey2] = v2list

        thisdf = compute_df_composite_keys(thisdf, compute_vdiff=False)

        thisdf = compute_df_ElasticEnergy(thisdf, style="Exp")

        thisdf = compute_df_SH_ratio(thisdf)
        #thisdf = compute_df_gmix(thisdf, temp=300.0, style="Eform")
        #thisdf = compute_df_gmix(thisdf, temp=300.0, style="SCALE")
        #thisdf = compute_df_VH4(thisdf, style="ROM")

        thisdf = compute_df_temp_ratio(thisdf)
        thisdf = compute_df_YS_Params(thisdf, style="Exp", UseCxx=False)
        thisdf = compute_df_YieldStrength(thisdf, a=0.55, style=1, strain_ratio=1.0e7)
        return thisdf

    def reorganizer2(self):
        rename_dict = {}
        rename_dict["alloy_name"] = "Composition"
        rename_dict["phases"] = "Crystal"
        rename_dict["VHN"] = "Vicker"
        rename_dict["H_chem"] = "Eform_ORG"

        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)

        compstrs = self.df["Composition"].tolist()
        cs = self.df["Crystal"].to_numpy()
        inds = np.arange(len(self.df), dtype=int)
        bads = []
        rcompstrs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break
            if isValid:
                c = str(cs[i])
                if "BCC" not in c.upper(): isValid = False
            if isValid:
                rcompstrs.append(rcompstr)
            else:
                bads.append(i)

        bads = np.array(bads)
        goods = np.delete(inds, bads)
        self.df = self.df.iloc[goods]
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0
        self.df["Vicker"] = self.df["Vicker"].to_numpy()
        self.df["ReducedFormula"] = rcompstrs
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Vicker")
        self.df = self.compute_standardard_features(self.df, style=1)
        print(len(self.df))

    def reorganizer3(self, DataAugment=False):
        rename_dict = {}
        rename_dict["Composition  (mole fraction)"] = "Composition"
        rename_dict["TestTempC"] = "temperature"
        rename_dict["Tension0/Compression1"] = "Tension0_Compression1"
        rename_dict["Experimental DeltaE_b0"] = "delta_Eb0_EXP"
        rename_dict["YieldStr(MPa)"] = "Yield_EXP"
        rename_dict["Poisson Ratio"] = "Poisson_ORG"
        rename_dict["Bulk Modulus"] = "Bulk_ORG"
        rename_dict["Shear Modulus"] = "Shear_ORG"
        '''
        rename_dict["Calculated sigma_y0"] = "sigma_y0_ORG"
        rename_dict["Calculated DeltaE_b0"] = "delta_Eb0_ORG"
        rename_dict["Lattice Constant"] = "LattPara_ORG"
        rename_dict["Varvenne Yield Stress (MPa)"] = "Yield_ORG"
        rename_dict["Yang Hmix"] = "Eform_ORG"
        '''

        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        tcs = self.df["Tension0_Compression1"].tolist()
        rcs = []
        inds = np.arange(len(self.df), dtype=int)
        bads = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            comp = Composition(compstr)
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                tc = str(tcs[i])
                tc = tc.strip()
                if "0" == tc[0].upper(): isValid = False
            if not isValid:
                bads.append(i)
            else:
                rcs.append(comp.reduced_formula)

        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        #self.df["delta_Eb0_ORG"] = self.df["delta_Eb0_ORG"] * KJ2eVA
        self.df["delta_Eb0_EXP"] = self.df["delta_Eb0_EXP"] * KJ2eVA
        self.df["Crystal"] = ["bcc"] * len(self.df)
        self.df["ReducedFormula"] = rcs

        compstrs = self.df["Composition"].tolist()
        rs = []
        rtcs = []
        ts = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            v = compstr2ROM(compstr, "radius")
            rs.append(v)
            v = compstr2ROM(compstr, "radius_TC")
            rtcs.append(v)
            v = compstr2ROM(compstr, "Tm")
            ts.append(v)
        rs = np.array(rs)
        self.df["radius"] = rs
        self.df["radius_TC"] = rtcs
        self.df["Tm_ROM"] = ts
        vs = 4.0 / 3.0 * np.pi * np.power(rtcs, 3)
        self.df["volume"] = vs
        ls = 2 * vs
        ls = np.power(ls, 1.0 / 3.0)
        self.df["LattPara"] = ls

        keys = ["ReducedFormula", "temperature"]
        self.df.sort_values(keys, inplace=True, ignore_index=True)

        inds = np.arange(len(self.df), dtype=int)
        self.df["tmpid"] = inds
        self.df = self.df.reset_index().set_index("tmpid")
        self.df = self.df.drop(columns=["index"])

        keys = ["ReducedFormula"]
        yss = self.df["Yield_EXP"].to_numpy()
        bads = []
        counts = 0
        vlast = [""] * len(keys)
        tlast = -1000.0
        ntcount = 1.0

        for i in range(len(self.df)):
            t = self.df.at[i, "temperature"]
            isIden = True

            for ii in range(len(keys)):
                v = self.df.at[i, keys[ii]]
                if v != vlast[ii]:
                    isIden = False
                    break
            if isIden:
                if abs(t - tlast) < 10.0:
                    bads.append(i - 1)
                    self.df.at[i, "Yield_EXP"] = (ntcount * self.df.at[i - 1, "Yield_EXP"] + yss[i]) / (ntcount + 1)
                    ntcount += 1.0
                else:
                    counts += 1
                    tlast = t
                    ntcount = 1.0
            else:
                if counts == 1: bads.append(i - 1)
                counts = 1
                ntcount = 1
                for ii in range(len(keys)):
                    vlast[ii] = self.df.at[i, keys[ii]]
                tlast = self.df.at[i, "temperature"]
        if counts == 1: bads.append(i)

        bads = np.array(bads).astype(int)
        goods = np.delete(inds, bads)
        self.baddf = self.df.iloc[bads]
        self.df = self.df.iloc[goods]

        self.baddf = self.delete_duplicate(self.baddf, df_multi=self.df, key="Yield_EXP")


        none = self.get_length_unique_system(self.baddf)
        nmore = self.get_length_unique_system(self.df)
        print(f"Length of compstrs with one measurement: {none} Length of compstrs with multiple measurement: {nmore}")
        print(f"Length for one measurement: {len(self.baddf)} Length for temp augment: {len(self.df)}")

        '''
        keys = ["ReducedFormula"]
        self.df = self.YieldStrength_Augment(self.df, keys, \
                   YSkey="Yield_EXP", a=0.55, strain_ratio=1e7)
        '''
        self.df = pd.concat([self.df, self.baddf])
        print(f"Length after temp augment: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        ss = []
        ebs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            s, eb = compstr2YS_Params(compstr, i, self.df, style="ORG", UseCxx=False)
            ss.append(s)
            ebs.append(eb)
        self.df["sigma_y0_ORG_CAL"] = ss
        self.df["delta_Eb0_ORG_CAL"] = ebs
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1="ROM", Exception_2="ZERO")


    def data_augment_outside(self, cols, fname, istension=False):
        def get_index(indss, ind, ndim=1):
            thisindex = []
            for i in range(len(indss)):
                thisindex.append(indss[i][ind])
            if ndim == 1:
                thisindex = thisindex[0]
            else:
                thisindex = tuple(thisindex)
            return thisindex
        ndim = len(cols)
        df1 = pd.read_csv(fname)
        df2 = self.df.copy()
        dict_tmp = df2.iloc[len(df2) - 1].to_dict()
        for key in dict_tmp:
            if key in STRING_KEYS:
                dict_tmp[key] = "UNKNOWN"
            elif key == "Tension0_Compression1":
                if istension:
                    dict_tmp[key] = 0.0
                else:
                    dict_tmp[key] = 1.0
            else:
                dict_tmp[key] = 0.0

        df1 = df1.sort_values(cols, ignore_index=True)
        df2 = df2.sort_values(cols, ignore_index=True)
        df1 = df1.reset_index().set_index(cols)
        df2 = df2.reset_index().set_index(cols)
        index1 = df1.index.tolist()
        index2 = df2.index.tolist()
        index1 = set(index1)
        index2 = set(index2)
        all_index = list(index2)
        add_index = []
        for i, thisindex in enumerate(index1):
            if thisindex in all_index:
                pass
            else:
                all_index.append(thisindex)
                add_index.append(thisindex)
        df2 = None

        df1 = pd.read_csv(fname)
        indss = []
        for i in range(len(cols)):
            key = cols[i]
            indss.append(df1[key].tolist())
        df_add = pd.DataFrame(columns=self.df.columns)
        for i in range(len(df1)):
            thisindex = get_index(indss, i, ndim=ndim)
            if thisindex in add_index:
                tmpdict = df1.iloc[i].to_dict()
                thisdict = copy.deepcopy(dict_tmp)
                for key in tmpdict:
                    if key in thisdict:
                        thisdict[key] = tmpdict[key]
                df_add.loc[len(df_add)] = thisdict

        df1 = None
        df2 = self.df.copy()
        if len(df_add) > 0:
            df2 = pd.concat([df2, df_add])

        print(f"=== results of combining two files  === ")
        print(f"length of orginal:{len(self.df)} added:{len(df_add)} final:{len(df2)}")
        print(f"=== end of combining two files ===")
        return df2

    def reorganizer4(self, DataAugment=True, fname="Giles_YS_EXT.csv", temp_augment=True):
        rename_dict = copy.deepcopy(rename_dict_Born)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        ys = self.df["Yield_EXP"].to_numpy()
        cs = self.df["Crystal"].to_numpy()
        ts = self.df["temperature"].to_numpy()
        tcs = self.df["Tension0_Compression1"].to_numpy()
        inds = np.arange(len(self.df), dtype=int)
        bads = []
        newcompstrs = []
        newcs = []
        rcompstrs = []
        newtcs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                y = str(ys[i])
                if y.upper() == "NAN":
                    isValid = False
                elif len(y) == 0:
                    isValid = False

            if isValid:
                c = str(cs[i])
                if "BCC" not in c.upper(): isValid = False

            if isValid:
                tc = str(tcs[i])
                tc = tc.strip()
                if "T" == tc[0].upper(): isValid = False

            if isValid:
                t = str(ts[i])
                if t.upper() == "NAN":
                    isValid = False
                elif len(t) == 0:
                    isValid = False

            if not isValid:
                bads.append(i)
            else:
                newcompstrs.append(compstr)
                c = str(cs[i])
                if len(c) == 0: c = "BCC"
                newcs.append(c)
                rcompstrs.append(rcompstr)
                newtcs.append(1)

        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]
            self.df["Composition"] = newcompstrs
            self.df["Crystal"] = newcs
            self.df["ReducedFormula"] = rcompstrs
            self.df["Tension0_Compression1"] = newtcs

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0

        if DataAugment:
            indexcols = ["ReducedFormula", "Tension0_Compression1"]
            self.df = self.data_augment_outside(indexcols, fname, istension=False)
            print(f"Length after data augment: {len(self.df)}")
        compstrs = self.df["Composition"].tolist()
        rs = []
        rtcs = []
        ts = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            v = compstr2ROM(compstr, "radius")
            rs.append(v)
            v = compstr2ROM(compstr, "radius_TC")
            rtcs.append(v)
            v = compstr2ROM(compstr, "Tm")
            ts.append(v)
        rs = np.array(rs)
        self.df["radius"] = rs
        self.df["radius_TC"] = rtcs
        self.df["Tm_ROM"] = ts
        vs = 4.0 / 3.0 * np.pi * np.power(rtcs, 3)
        self.df["volume"] = vs
        ls = 2 * vs
        ls = np.power(ls, 1.0 / 3.0)
        self.df["LattPara"] = ls

        keys = ["ReducedFormula", "Tension0_Compression1", "temperature"]
        self.df.sort_values(keys, inplace=True, ignore_index=True)

        inds = np.arange(len(self.df), dtype=int)
        self.df["tmpid"] = inds
        self.df = self.df.reset_index().set_index("tmpid")
        self.df = self.df.drop(columns=["index"])

        keys = ["ReducedFormula"]
        yss = self.df["Yield_EXP"].to_numpy()
        bads = []
        counts = 0
        vlast = [""] * len(keys)
        tlast = -1000.0
        ntcount = 1.0

        for i in range(len(self.df)):
            t = self.df.at[i, "temperature"]
            isIden = True
            for ii in range(len(keys)):
                v = self.df.at[i, keys[ii]]
                if v != vlast[ii]:
                    isIden = False
                    break
            if isIden:
                if abs(t - tlast) < 10.0:
                    self.df.at[i, "Yield_EXP"] = (ntcount * self.df.at[i - 1, "Yield_EXP"] + yss[i]) / (ntcount + 1)
                    ntcount += 1.0
                    bads.append(i - 1)
                else:
                    counts += 1
                    ntcount = 1.0
                    tlast = t
            else:
                if counts == 1: bads.append(i - 1)
                counts = 1
                ntcount = 1
                for ii in range(len(keys)):
                    vlast[ii] = self.df.at[i, keys[ii]]
                tlast = self.df.at[i, "temperature"]

        if counts == 1: bads.append(i)

        bads = np.array(bads).astype(int)
        goods = np.delete(inds, bads)
        self.baddf = self.df.iloc[bads]
        self.df = self.df.iloc[goods]

        self.baddf = self.delete_duplicate(self.baddf, df_multi=self.df, key="Yield_EXP")
        rcss = self.baddf["ReducedFormula"].tolist()

        none = self.get_length_unique_system(self.baddf)
        nmore = self.get_length_unique_system(self.df)
        tthres = 1500
        ts = self.df["temperature"].to_numpy()
        tmax = np.max(ts)
        ts = np.compress(ts > tthres, ts)
        print(f"Length T> {tthres}: {len(ts)} tmax:{tmax} ts:{ts}")
        print(f"Length of compstrs with one measurement: {none} Length of compstrs with multiple measurement: {nmore}")
        print(f"length one measurement:{len(self.baddf)} Length for temp-augment: {len(self.df)}")

        keys = ["ReducedFormula"]
        if temp_augment:
            self.df = self.YieldStrength_Augment(self.df, keys, YSkey="Yield_EXP", a=0.55, strain_ratio=1e7)

        self.df = self.df[self.df["Yield_EXP"] >= 5.0]
        self.df.to_csv("YS_Temp_Augment.csv", index=False)

        self.df = pd.concat([self.df, self.baddf])

        self.df = self.df[self.df["Yield_EXP"] <= 3000.0]
        self.df = self.df[self.df["Yield_EXP"] >= 5.0]

        self.df = self.df.reset_index()
        dropcols = []
        if "tmpid" in self.df.columns: dropcols.append("tmpid")
        if "index" in self.df.columns: dropcols.append("index")
        if "level_0" in self.df.columns: dropcols.append("level_0")
        if len(dropcols) > 0: self.df = self.df.drop(columns=dropcols)
        ts = self.df["temperature"].to_numpy()
        ts_low = np.compress(ts < 800, ts)
        ts_high = np.compress(ts >= 800, ts)
        print(f"Length df: {len(self.df)} lows:{len(ts_low)} highs:{len(ts_high)}")
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1="ROM", Exception_2="ZERO")


    def reorganizer5(self, Phases="BCC", Exception_1="ROM"):
        rename_dict = copy.deepcopy(rename_dict_Born)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        ys = self.df["Elongation_EXP"].to_numpy()
        ys2 = self.df["Elongation_plastic_EXP"].to_numpy()
        cs = self.df["Crystal"].to_numpy()
        ts = self.df["temperature"].to_numpy()
        tcs = self.df["Tension0_Compression1"].to_numpy()

        yieldss = self.df["Yield_EXP"].to_numpy()
        utss = self.df["UTS_EXP"].to_numpy()
        youngs = self.df["Youngs_EXP"].to_numpy()
        cyoungs = self.df["Youngs_ROM_ORG"].to_numpy()
        inds = np.arange(len(self.df), dtype=int)
        bads = []
        newcompstrs = []
        newcs = []
        rcompstrs = []
        newts = []
        newtcs = []
        newelongs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                tc = str(tcs[i])
                tc = tc.strip()
                if "T" == tc[0].upper(): isValid = False

            if isValid:
                t = str(ts[i])
                if t.upper() == "NAN":
                    isValid = False
                elif len(t) == 0:
                    isValid = False
                else:
                    t = float(t)
                    if t < 25.0 - 50.0 or t > 25.0 + 50.0:
                        isValid = False

            if isValid:
                y = str(ys[i])
                y2 = str(ys2[i])
                if y.upper() == "NAN" and y2.upper() == "NAN":
                    isValid = False
                elif y.upper() == "NAN":
                    y = float(y2)
                    young = str(youngs[i])
                    cyoung = str(cyoungs[i])
                    yields = str(yieldss[i])
                    uts = str(utss[i])
                    #print(f"i:{i} ybefore:{y} ys:{young} cys:{cyoung} yield:{yields} uts:{uts}")
                    if young.upper() == "NAN" and cyoung.upper() == "NAN":
                        isValid = False
                    else:
                        if young.upper() == "NAN":
                            young = float(cyoung) * 1000
                        else:
                            young = float(young) * 1000
                        if yields.upper() == "NAN" and uts.upper() == "NAN":
                            isValid = False
                        elif uts.upper() == "NAN":
                            y += float(yields) / young * 100.0
                        else:
                            y += float(uts) / young * 100.0
                else:
                    y = float(y)
                if isValid and y < 0.001:
                    print(f"compstr:{compstr} elong/compr:{y} plastic:{y2}")
                    isValid = False

            if not isValid:
                bads.append(i)
            else:
                c = str(cs[i])
                if len(c) == 0:
                    c = "BCC"
                elif "BCC" in c.upper() or "FCC" not in c.upper():
                    if "BCC" in c.upper() and "FCC" in c.upper():
                        c = "MIX"
                    elif "BCC" in c.upper():
                        c = "BCC"
                    else:
                        c = "FCC"
                else:
                    c = "AMO"
                if Phases.upper() == "ALL":
                    pass
                elif Phases.upper() == "MIX":
                    if c == "AMO": isValid = False
                elif Phases.upper() == "BCC":
                    if c == "AMO" or c == "FCC": isValid = False
                elif Phases.upper() == "FCC":
                    if c == "AMO" or c == "BCC": isValid = False
                elif Phases.upper() == "AMO":
                    if c != "AMO": isValid = False
                elif Phases.upper() == "BCCONLY":
                    if c != "BCC": isValid = False
                elif Phases.upper() == "FCCONLY":
                    if c != "FCC": isValid = False

                if not isValid:
                    bads.append(i)
                else:
                    newcompstrs.append(compstr)
                    newcs.append(c)
                    rcompstrs.append(rcompstr)
                    newts.append(t)
                    newtcs.append(1)
                    newelongs.append(y)

        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]
            self.df["Composition"] = newcompstrs
            self.df["Crystal"] = newcs
            self.df["ReducedFormula"] = rcompstrs
            self.df["temperature"] = newts
            self.df["Tension0_Compression1"] = newtcs
            self.df["Elongation_EXP"] = newelongs

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Elongation_EXP")
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")

    def reorganizer6(self, Phases="ALL", Exception_1="ROM",app_b2h=True):
        rename_dict = copy.deepcopy(rename_dict_Born)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        ys = self.df["Elongation_EXP"].to_numpy()
        ys2 = self.df["Elongation_plastic_EXP"].to_numpy()
        cs = self.df["Crystal"].to_numpy()
        ts = self.df["temperature"].to_numpy()
        tcs = self.df["Tension0_Compression1"].to_numpy()

        yieldss = self.df["Yield_EXP"].to_numpy()
        utss = self.df["UTS_EXP"].to_numpy()
        youngs = self.df["Youngs_EXP"].to_numpy()
        cyoungs = self.df["Youngs_ROM_ORG"].to_numpy()

        inds = np.arange(len(self.df), dtype=int)
        bads = []
        newcompstrs = []
        newcs = []
        rcompstrs = []
        newts = []
        newtcs = []
        newelongs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                tc = str(tcs[i])
                tc = tc.strip()
                if "C" == tc[0].upper(): isValid = False

            if isValid:
                t = str(ts[i])
                if t.upper() == "NAN":
                    isValid = False
                elif len(t) == 0:
                    isValid = False
                else:
                    t = float(t)
                    if t < 25.0 - 50.0 or t > 25.0 + 50.0:
                        isValid = False

            if isValid:
                y = str(ys[i])
                y2 = str(ys2[i])
                if y.upper() == "NAN" and y2.upper() == "NAN":
                    isValid = False
                elif y.upper() == "NAN":
                    y = float(y2)
                    young = str(youngs[i])
                    cyoung = str(cyoungs[i])
                    yields = str(yieldss[i])
                    uts = str(utss[i])
                    if young.upper() == "NAN" and cyoung.upper() == "NAN":
                        isValid = False
                    else:
                        if young.upper() == "NAN":
                            young = float(cyoung) * 1000
                        else:
                            young = float(young) * 1000
                        if yields.upper() == "NAN" and uts.upper() == "NAN":
                            isValid = False
                        elif uts.upper() == "NAN":
                            y += float(yields) / young * 100.0
                        else:
                            y += float(uts) / young * 100.0
                else:
                    y = float(y)
                if isValid and y < 0.001:
                    print(f"compstr:{compstr} elong/compr:{y} plastic:{y2}")
                    isValid = False

            if not isValid:
                bads.append(i)
            else:
                c = str(cs[i])
                c = c.upper()
                c = c.replace(" ", "")
                c = c.replace("+", "")
                c = c.replace("；", "")
                c = c.replace("，", "")
                if len(c) == 0:
                    c = "BCC"
                elif "BCC" in c:
                    if "BCC1" in c:
                        c = c.replace("BCC1", "")
                    if "BCC2" in c:
                        c = c.replace("BCC2", "")
                    if "BCC" in c:
                        c = c.replace("BCC", "")
                    if len(c) == 0:
                        c = "BCC"
                    elif "HCP" in c:
                        c = "BCC+HCP"
                    else:
                        c = "MIX"
                else:
                    c = "OTHER"
                if Phases.upper() == "ALL":
                    pass
                elif Phases.upper() == "MIX":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC+HCP":
                    if c == "OTHER" or c == "MIX": isValid = False
                elif Phases.upper() == "BCCONLY":
                    if c != "BCC": isValid = False

                if not isValid:
                    bads.append(i)
                else:
                    newcompstrs.append(compstr)
                    newcs.append(c)
                    rcompstrs.append(rcompstr)
                    newts.append(t)
                    newtcs.append(0)
                    newelongs.append(y)

        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]
            self.df["Composition"] = newcompstrs
            self.df["Crystal"] = newcs
            self.df["ReducedFormula"] = rcompstrs
            self.df["temperature"] = newts
            self.df["Tension0_Compression1"] = newtcs
            self.df["Elongation_EXP"] = newelongs

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Elongation_EXP")
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")
        if app_b2h:
            inds = np.arange(len(self.df), dtype=int)
            ys = self.df["b2h4phase"].to_numpy()
            nbefore = len(self.df)
            inds = np.compress(ys>0.0, inds)
            self.df = self.df.iloc[inds]
            print(f"Length before: {nbefore} after applying b2h: {len(self.df)}")

    def reorganizer7(self, DataAugment=True, fname="Beniwal_Hardness_EXT.csv"):
        rename_dict = copy.deepcopy(rename_dict_Born)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        cs = self.df["Crystal"].to_numpy()
        ts = self.df["temperature"].to_numpy()
        tcs = self.df["Tension0_Compression1"].to_numpy()
        vhs = self.df["Vicker"].to_numpy()
        inds = np.arange(len(self.df), dtype=int)
        bads = []
        newcompstrs = []
        newcs = []
        rcompstrs = []
        newts = []
        newtcs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                vh = str(vhs[i])
                if vh.upper() == "NAN":
                    isValid = False
                elif len(vh) == 0:
                    isValid = False

            if isValid:
                c = str(cs[i])
                if "BCC" not in c.upper(): isValid = False

            if isValid:
                t = str(ts[i])
                if t.upper() == "NAN":
                    t = 25.0
                elif len(t) == 0:
                    t = 25.0
                else:
                    t = float(t)
                    if t > 25.0 + 50.0 or t < 25 - 50.0: isValid = False

            if not isValid:
                bads.append(i)
            else:
                newcompstrs.append(compstr)
                c = str(cs[i])
                if len(c) == 0: c = "BCC"
                newcs.append(c)
                rcompstrs.append(rcompstr)
                newts.append(t)
                if tcs[i] == "T":
                    newtcs.append(0)
                else:
                    newtcs.append(0)
        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]
            self.df["Composition"] = newcompstrs
            self.df["Crystal"] = newcs
            self.df["ReducedFormula"] = rcompstrs
            self.df["temperature"] = newts
            self.df["Tension0_Compression1"] = newtcs

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0

        self.df = self.delete_duplicate(self.df, df_multi=None, key="Vicker")

        if DataAugment:
            indexcols = ["ReducedFormula"]
            self.df = self.data_augment_outside(indexcols, fname, istension=False)
            print(f"Length after data augment: {len(self.df)}")
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Vicker")
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1="ROM", Exception_2="ZERO")



    def reorganizer8(self, Phases="BCC", Exception_1="ROM", fname="Elongation_EXT.csv", app_b2h=True):
        rename_dict = copy.deepcopy(rename_dict_JT)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        ys = self.df["Elongation_EXP"].to_numpy()
        ys2 = self.df["Elongation_plastic_EXP"].to_numpy()
        cs = self.df["Crystal"].to_numpy()
        ts = self.df["temperature"].to_numpy()
        tcs = self.df["Tension0_Compression1"].to_numpy()

        yieldss = self.df["Yield_EXP"].to_numpy()
        utss = self.df["UTS_EXP"].to_numpy()
        #youngs = self.df["Youngs_EXP"].to_numpy()

        inds = np.arange(len(self.df), dtype=int)
        bads = []
        newcompstrs = []
        newcs = []
        rcompstrs = []
        newts = []
        newtcs = []
        newelongs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                tc = str(tcs[i])
                tc = tc.strip()
                if "C" == tc[0].upper(): isValid = False

            if isValid:
                t = str(ts[i])
                if t.upper() == "NAN":
                    isValid = False
                elif len(t) == 0:
                    isValid = False
                else:
                    t = float(t)
                    if t < 25.0 - 50.0 or t > 25.0 + 50.0:
                        isValid = False

            if isValid:
                y = str(ys[i])
                y2 = str(ys2[i])
                if y.upper() == "NAN":
                    isValid = False
                else:
                    y = float(y)
                if isValid and y < 0.001:
                    print(f"compstr:{compstr} elong/compr:{y} plastic:{y2}")
                    isValid = False

            if not isValid:
                bads.append(i)
            else:
                c = str(cs[i])
                c = c.upper()
                c = c.replace(" ", "")
                c = c.replace("+", "")
                c = c.replace("；", "")
                c = c.replace("，", "")
                if len(c) == 0:
                    c = "BCC"
                elif "BCC" in c:
                    if "BCC1" in c:
                        c = c.replace("BCC1", "")
                    if "BCC2" in c:
                        c = c.replace("BCC2", "")
                    if "BCC" in c:
                        c = c.replace("BCC", "")
                    if len(c) == 0:
                        c = "BCC"
                    elif "HCP" in c:
                        c = "BCC+HCP"
                    else:
                        c = "MIX"
                else:
                    c = "OTHER"
                if Phases.upper() == "ALL":
                    pass
                elif Phases.upper() == "MIX":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC+HCP":
                    if c == "OTHER" or c== "MIX": isValid = False
                elif Phases.upper() == "BCCONLY":
                    if c != "BCC": isValid = False

                if not isValid:
                    bads.append(i)
                else:
                    newcompstrs.append(compstr)
                    newcs.append(c)
                    rcompstrs.append(rcompstr)
                    newts.append(t)
                    newtcs.append(0)
                    newelongs.append(y)

        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]
            self.df["Composition"] = newcompstrs
            self.df["Crystal"] = newcs
            self.df["ReducedFormula"] = rcompstrs
            self.df["temperature"] = newts
            self.df["Tension0_Compression1"] = newtcs
            self.df["Elongation_EXP"] = newelongs

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0

        nbefore = len(self.df)
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Elongation_EXP")

        df_org = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")
        df_org.to_csv(os.path.join(self.SAVE_PATH, self.orgfile), index=False, float_format=float_format)

        indexcols = ["ReducedFormula"]
        self.df = self.data_augment_outside(indexcols, fname, istension=True)
        print(f"Length after data augment: {len(self.df)}")
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Elongation_EXP")
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")
        if app_b2h:
            inds = np.arange(len(self.df), dtype=int)
            ys = self.df["bcc2hcp_Gmix_TC2CPA"].to_numpy()
            nbefore = len(self.df)
            inds = np.compress(ys>0.0, inds)
            self.df = self.df.iloc[inds]
            print(f"Length before: {nbefore} after applying b2h: {len(self.df)}")

    def reorganizer9(self, Phases="BCC", Exception_1="ROM", fname="Compression_EXT.csv", app_b2h=True):
        rename_dict = copy.deepcopy(rename_dict_JT)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        ys = self.df["Elongation_EXP"].to_numpy()
        ys2 = self.df["Elongation_plastic_EXP"].to_numpy()
        cs = self.df["Crystal"].to_numpy()
        ts = self.df["temperature"].to_numpy()
        tcs = self.df["Tension0_Compression1"].to_numpy()

        yieldss = self.df["Yield_EXP"].to_numpy()
        utss = self.df["UTS_EXP"].to_numpy()
        #youngs = self.df["Youngs_EXP"].to_numpy()

        inds = np.arange(len(self.df), dtype=int)
        bads = []
        newcompstrs = []
        newcs = []
        rcompstrs = []
        newts = []
        newtcs = []
        newelongs = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                tc = str(tcs[i])
                tc = tc.strip()
                if "T" == tc[0].upper(): isValid = False

            if isValid:
                t = str(ts[i])
                if t.upper() == "NAN":
                    isValid = False
                elif len(t) == 0:
                    isValid = False
                else:
                    t = float(t)
                    if t < 25.0 - 50.0 or t > 25.0 + 50.0:
                        isValid = False

            if isValid:
                y = str(ys[i])
                y2 = str(ys2[i])
                if y.upper() == "NAN":
                    isValid = False
                else:
                    y = float(y)
                if isValid and y < 0.001:
                    print(f"compstr:{compstr} elong/compr:{y} plastic:{y2}")
                    isValid = False

            if not isValid:
                bads.append(i)
            else:
                c = str(cs[i])
                c = c.upper()
                c = c.replace(" ", "")
                c = c.replace("+", "")
                c = c.replace("；", "")
                c = c.replace("，", "")
                if len(c) == 0:
                    c = "BCC"
                elif "BCC" in c:
                    if "BCC1" in c:
                        c = c.replace("BCC1", "")
                    if "BCC2" in c:
                        c = c.replace("BCC2", "")
                    if "BCC" in c:
                        c = c.replace("BCC", "")
                    if len(c) == 0:
                        c = "BCC"
                    elif "HCP" in c:
                        c = "BCC+HCP"
                    else:
                        c = "MIX"
                else:
                    c = "OTHER"
                if Phases.upper() == "ALL":
                    pass
                elif Phases.upper() == "MIX":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC+HCP":
                    if c == "OTHER" or c== "MIX": isValid = False
                elif Phases.upper() == "BCCONLY":
                    if c != "BCC": isValid = False

                if not isValid:
                    bads.append(i)
                else:
                    newcompstrs.append(compstr)
                    newcs.append(c)
                    rcompstrs.append(rcompstr)
                    newts.append(t)
                    newtcs.append(1)
                    newelongs.append(y)

        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]
            self.df["Composition"] = newcompstrs
            self.df["Crystal"] = newcs
            self.df["ReducedFormula"] = rcompstrs
            self.df["temperature"] = newts
            self.df["Tension0_Compression1"] = newtcs
            self.df["Elongation_EXP"] = newelongs

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0

        nbefore = len(self.df)
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Elongation_EXP")

        df_org = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")
        df_org.to_csv(os.path.join(self.SAVE_PATH, self.orgfile), index=False, float_format=float_format)

        indexcols = ["ReducedFormula"]
        self.df = self.data_augment_outside(indexcols, fname, istension=False)
        print(f"Length after data augment: {len(self.df)}")
        self.df = self.delete_duplicate(self.df, df_multi=None, key="Elongation_EXP")
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")
        if app_b2h:
            inds = np.arange(len(self.df), dtype=int)
            ys = self.df["bcc2hcp_Gmix_TC2CPA"].to_numpy()
            nbefore = len(self.df)
            inds = np.compress(ys>0.0, inds)
            self.df = self.df.iloc[inds]
            print(f"Length before: {nbefore} after applying b2h: {len(self.df)}")

    def reorganizer10(self, Exception_1="ROM"):
        rename_dict = copy.deepcopy(rename_dict_JT)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        ys = self.df["Elongation_EXP"].to_numpy()
        yieldss = self.df["Yield_EXP"].to_numpy()
        rcompstrs = []
        newelongs = []
        newyieldss = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            rcompstrs.append(rcompstr)
            try:
                y = float(ys[i])
            except:
                y = 0
            newelongs.append(y)
            try:
                y = float(yieldss[i])
            except:
                y = 0
            newyieldss.append(y)

        self.df["ReducedFormula"] = rcompstrs
        self.df["Elongation_EXP"] = newelongs
        self.df["Yield_EXP"] = newyieldss

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0

        self.df = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")

    def reorganizer11(self, Phases="BCC", Exception_1="ROM"):
        rename_dict = copy.deepcopy(rename_dict_JT)
        self.df = self.df.rename(columns=rename_dict)
        self.df_drop_columns(rename_dict)
        self.df = get_df_pretty_formula(self.df, normalization=self.normalization)
        print(f"Original length: {len(self.df)}")

        compstrs = self.df["Composition"].tolist()
        cs = self.df["Crystal"].to_numpy()
        ts = self.df["temperature"].to_numpy()
        Vickers = self.df["Vicker"].to_numpy()

        inds = np.arange(len(self.df), dtype=int)
        bads = []
        rcompstrs = []
        newcompstrs = []
        newcs = []
        newts = []
        newVickers = []
        for i in range(len(self.df)):
            compstr = compstrs[i]
            compstr = compstr.replace(" ", "")
            comp = Composition(compstr)
            rcompstr = comp.reduced_formula
            isValid = True
            for el in comp.elements:
                sym = el.symbol
                if sym not in Element_external:
                    isValid = False
                    break

            if isValid:
                if len(comp.elements) <= 2:
                    isValid = False

            if isValid:
                t = str(ts[i])
                if t.upper() == "NAN":
                    isValid = False
                elif len(t) == 0:
                    isValid = False
                else:
                    t = float(t)
                    if t < 25.0 - 50.0 or t > 25.0 + 50.0:
                        isValid = False
            if isValid:
                y = str(Vickers[i])
                if y.upper() == "NAN":
                    isValid = False
                else:
                    y = float(y)

            if not isValid:
                bads.append(i)
            else:
                c = str(cs[i])
                c = c.upper()
                c = c.replace(" ", "")
                c = c.replace("+", "")
                c = c.replace("；", "")
                c = c.replace("，", "")
                if len(c) == 0:
                    c = "BCC"
                elif "BCC" in c:
                    if "BCC1" in c:
                        c = c.replace("BCC1", "")
                    if "BCC2" in c:
                        c = c.replace("BCC2", "")
                    if "BCC" in c:
                        c = c.replace("BCC", "")
                    if len(c) == 0:
                        c = "BCC"
                    elif "HCP" in c:
                        c = "BCC+HCP"
                    else:
                        c = "MIX"
                else:
                    c = "OTHER"
                if Phases.upper() == "ALL":
                    pass
                elif Phases.upper() == "MIX":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC":
                    if c == "OTHER": isValid = False
                elif Phases.upper() == "BCC+HCP":
                    if c == "OTHER" or c== "MIX": isValid = False
                elif Phases.upper() == "BCCONLY":
                    if c != "BCC": isValid = False

                if not isValid:
                    bads.append(i)
                else:
                    newcompstrs.append(compstr)
                    newcs.append(c)
                    rcompstrs.append(rcompstr)
                    newts.append(t)
                    newVickers.append(y)

        if len(bads) > 0:
            bads = np.array(bads)
            goods = np.delete(inds, bads)
            self.df = self.df.iloc[goods]
            self.df["Composition"] = newcompstrs
            self.df["Crystal"] = newcs
            self.df["ReducedFormula"] = rcompstrs
            self.df["temperature"] = newts
            self.df["Vicker"] = newVickers

        print(f"Length after removal: {len(self.df)}")
        self.df["temperature"] = self.df["temperature"] + 273.15
        self.df["Vicker"] = self.df["Vicker"] * 9.8 / 1000.0
        self.df = self.compute_standardard_features(self.df, style=1, Exception_1=Exception_1, Exception_2="ZERO")


