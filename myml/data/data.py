import copy
import os
from itertools import combinations

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

from myml.myglobal import VERY_SMALL_VALUE, Element_external_crystal, Element_external
from myml.myglobal import Element_negativity, nele_default
from myml.myglobal import compstr2PMGROM, compstr2compid, compstr2entropy
from myml.myglobal import compstr2concs, compute_sequential_concs, get_sorted_compspace
from myml.myglobal import config_vars, Constants, TEMPERATURE_LEVEL, key2elekey, b2h4phase, b2h4ml, thres4b2h
from myml.myglobal import Basic_ROM_Keys, Add_ROM_Keys, DELTA_Keys, DISTORT_Keys, SCALED_Keys
from myml.myelements.myelements import DF_bcc, DF_hcp, DF_basic, DF_ground, DF_external
from myml.myelements.myelements import ndip_all_keys
from myml.myelements.myelements import Element_extreme_keys, additional_keys
from myml.data.data_util import value_normalization
from myml.data.data_util import compstr2ROM, compstr2StrainRoy
#from myml.data.data_util import compute_YS_Params_from_2pts, compute_YS_from_2params
from myml.data.data_util import compute_df_reducedformula, compute_df_concs, compute_df_sconfig, compute_df_radius_sws
from myml.data.data_util import compute_df_volume, compute_df_density, compute_df_composite_keys
from myml.data.data_util import compute_df_ElasticEnergy, compute_df_SH_ratio
from myml.data.data_util import compute_df_YS_Params, compute_df_YieldStrength
from myml.data.data_util import get_df_concs, compute_df_gmix, expand_df_templevels, get_element_value

# from pymatgen.util.string import latexify, htmlify
# from pymatgen.analysis.phase_diagram import PDEntry

DATA_PATH = config_vars["DATA_PATH"]
kb = Constants["kb"]
bohr2angstrom = Constants["bohr2angstrom"]
density2gcm = Constants["density2gcm"]
eVA2GPa = Constants["eVA2GPa"]
GPa2eVA = Constants["GPa2eVA"]
KJ2eVA = Constants["KJ2eVA"]
float_format = Constants["float_format"]

class myData:
    def __init__(self, fname, from_DATA=False, Source="INTERNAL",
                 Recompute=False, Temp_Level=TEMPERATURE_LEVEL, elements=None):

        self.IndKey = "CompID"
        self.fname = fname
        if ".csv" in fname:
            if from_DATA:
                self.df = pd.read_csv(os.path.join(DATA_PATH, fname))
            else:
                self.df = pd.read_csv(fname)
        else:
            raise ValueError("Data must be in csv format")
        try:
            self.df = self.df.set_index(self.IndKey, inplace=False)
        except:
            self.df[self.IndKey] = np.arange(len(self.df), dtype=int)
            self.df = self.df.set_index(self.IndKey, inplace=False)
        self.ntot = len(self.df)
        if Source[0:3].upper() == "INT":
            self.Source = "INTERNAL"
        else:
            self.Source = "EXTERNAL"

        if elements is None:
            if Source[0:3].upper() == "INT":
                elements = Element_negativity[0:nele_default]
            else:
                elements = Element_external[0:len(Element_external)]
        self.elements = elements

        self.Recompute = Recompute
        self.Temp_Level = Temp_Level
        if "bcc" in self.fname:
            self.latname = "bcc"
        elif "hcp" in self.fname:
            self.latname = "hcp"
        elif "laves" in self.fname:
            self.latname = "laves"
        elif "b2" in self.fname:
            self.latname = "b2"
        elif "sfs" in self.fname:
            self.latname = "sfs"
        else:
            self.latname = "mix"

        if self.Source == "INTERNAL":
            self.initialize_df()
        self.gooddf = self.df.copy(deep=True)
        self.baddf = pd.DataFrame(columns=[self.IndKey] + self.df.columns.tolist())
        self.baddf = self.baddf.set_index(self.IndKey, inplace=False)
        self.ngood = len(self.gooddf)
        self.compstrs = self.gooddf["Composition"].to_numpy()
        self.undo1df = self.baddf.copy(deep=True)
        self.undo2df = self.baddf.copy(deep=True)
        self.undo3df = self.baddf.copy(deep=True)

    def initialize_additional_dfs(self):
        self.gooddf = self.df.copy(deep=True)
        self.baddf = pd.DataFrame(columns=[self.IndKey] + self.df.columns.tolist())
        self.baddf = self.baddf.set_index(self.IndKey, inplace=False)
        self.set_df_values()
        self.undo1df = self.baddf.copy(deep=True)
        self.undo2df = self.baddf.copy(deep=True)
        self.undo3df = self.baddf.copy(deep=True)

    def initialize_df(self):
        if "Group" in self.df.columns: self.df = self.df.drop(columns=["Group"])
        if self.Recompute:
            self.df = self.compute_reducedformula(self.df)
            self.df = self.compute_concs(self.df)
            self.df = self.compute_radius_sws(self.df)
            self.df = self.compute_volume(self.df)
            self.df = self.compute_sconfig(self.df)
            #self.df = compute_eform_omegas(self.df, OutKey="Eform_BOKAS")
            #self.df = compute_df_eform_scaled(self.df, app="_SCALED", Keys=None)
            #self.df = self.compute_gmix_scaled(self.df, temp=300.0, Recompute=self.Recompute)
        else:
            if "radius" not in self.df.columns: self.df = self.compute_radius_sws(self.df)
            if "volume" not in self.df.columns: self.df = self.compute_volume(self.df)
            if "ReducedFormula" not in self.df.columns: self.df = self.compute_reducedformula(self.df)
            if "CONC0" not in self.df.columns: self.df = self.compute_concs(self.df)
            if "Sconf" not in self.df.columns: self.df = self.compute_sconfig(self.df)
            #if "Eform_BOKAS" not in self.df.columns: self.df = compute_eform_omegas(self.df, OutKey="Eform_BOKAS")
            #if "Eform_SCALED" not in self.df.columns: self.df = compute_df_eform_scaled(self.df, app="_SCALED",
            #                                                                            Keys=None)
            #if "Gmix_CPA_SCALED" not in self.df.columns: self.df = self.compute_gmix_scaled(self.df, temp=300.0,
            #                                                                         Recompute=True)

        keys = []
        for key in Basic_ROM_Keys:
            if self.Recompute:
                keys.append(key)
            else:
                if key not in self.df.columns: keys.append(key)
        self.df = self.compute_ROMs_df(self.df, keys, SameCrystal=False, Add_ROM=False)

        if self.Recompute:
            self.df = self.compute_density(self.df)
        else:
            if "density" not in self.df.columns: self.df = self.compute_density(self.df)

        keys = []
        for key in Add_ROM_Keys:
            if self.Recompute:
                keys.append(key)
            else:
                if key + "_ROM" not in self.df.columns: keys.append(key)
        self.df = self.compute_ROMs_df(self.df, keys, SameCrystal=False, Add_ROM=True)
        self.initialize_additional_dfs()

    def compute_Hardness_features(self, style="Exp"):
        if self.Source == "INTERNAL":
            self.compute_DELTAs_df(self.df, DELTA_Keys, style=1, Add_DELTA=True, Recompute=self.Recompute)
            self.compute_DISTORTs_df(self.df, DISTORT_Keys, Add_DISTORT=True, Recompute=self.Recompute)
            self.compute_ElasticEnergy(self.df, style=style, Recompute=self.Recompute)
            self.compute_SH_ratio(self.df, Recompute=self.Recompute)
            self.compute_YS_Params(self.df, style=style, Recompute=self.Recompute)

            if "Exp_Poisson_Comp" not in self.df.columns:
                self.df = compute_df_composite_keys(self.df, compute_vdiff=False)
            else:
                if self.Recompute:
                    self.df = compute_df_composite_keys(self.df, compute_vdiff=False)

            self.initialize_additional_dfs()

    def compute_Yield_and_features(self, style="Exp", UseCxx=False, a=0.55, YSstyle=1, strain_ratio=1.0e7,
                                   FileSeparation=False):
        if self.Source == "INTERNAL":
            self.compute_Hardness_features(style=style)
            self.df = self.compute_YieldStrength(self.df, style=style, UseCxx=UseCxx,
                                                 a=a, YSstyle=YSstyle, strain_ratio=strain_ratio,
                                                 Recompute=self.Recompute)
            if FileSeparation:
                inds = np.arange(len(self.df), dtype=int)
                for i in range(len(self.Temp_Level)):
                    t = self.Temp_Level[i]
                    thisfname = self.fname.replace(".csv", "")
                    thisfname += "_T" + str(int(t)) + ".csv"
                    thisinds = inds[i * self.ntot:(i + 1) * self.ntot]
                    thisdf = self.df.iloc[thisinds]
                    self.save_to(thisfname, df=thisdf)
            self.initialize_additional_dfs()

    def set_df_values(self):
        self.ngood = len(self.gooddf)
        self.compstrs = self.gooddf["Composition"].to_numpy()

    def save_to(self, fname, df=None):
        if df is None: df = self.df.copy()
        df.to_csv(fname, index=True, float_format=float_format)

    def display_dataframe(self, complist, keys=None):
        from myml.myelements.myelements import eos_keys
        if keys is None:
            keys = eos_keys[0:len(eos_keys)]
            if "bcc" in self.fname: eos_keys += ["bcc2hcp"]
        idlist = []
        for compstr in complist:
            idlist.append(compstr2compid(compstr, self.latname))
        idlist = np.array(idlist)
        outdf = self.df.loc[idlist]
        outdf = outdf[keys]
        return outdf

    def normalization(self, df, keys=None, extended=True, ele_ref="BAS", savefile=False, outfile=None, Add_norm=False):
        columns = df.columns.tolist()
        if keys is None:
            thiskeys = columns[0:len(columns)]
        else:
            if isinstance(keys, str): keys = [keys]
            thiskeys = []
            for key in keys:
                if key in df.columns:
                    thiskeys.append(key)

        if ele_ref[0:3] == "BAS":
            DF_ele = DF_basic.copy(deep=True)
        elif ele_ref[0:3] == "ADD":
            DF_ele = DF_ground.copy(deep=True)
        else:
            DF_ele = DF_external.copy(deep=True)

        for ikey in range(len(thiskeys)):
            thiskey = thiskeys[ikey]
            thisnorm = True
            thisvals = df[thiskey].to_numpy()
            vmin = np.min(thisvals)
            vmax = np.max(thisvals)
            if extended:
                thiselekey = key2elekey(thiskey)
                if "DISTORT" in thiskey or "DELTA" in thiskey:
                    pass
                else:
                    if thiselekey in DF_ele.columns:
                        vmin = DF_ele[thiselekey].min()
                        vmax = DF_ele[thiselekey].max()
            if vmax == vmin: thisnorm = False
            if thisnorm:
                normvals = value_normalization(thisvals, vmin, vmax)
            else:
                normvals = np.ones(len(df))
            if Add_norm: df[thiskey + "_norm"] = normvals

        if savefile:
            if outfile is None:
                outfile = self.fname
                if "_normed" not in outfile:
                    outfile = outfile.replace(".csv", "")
                    outfile = outfile + "_normed.csv"
            self.save_to(outfile, df=df)
        return df

    def get_concs(self, df):
        return get_df_concs(df)

    def compute_radius_sws(self, df):
        return compute_df_radius_sws(df)

    def compute_volume(self, df):
        return compute_df_volume(df, latname=self.latname)

    def compute_reducedformula(self, df):
        return compute_df_reducedformula(df)

    def compute_concs(self, df):
        return compute_df_concs(df, elements=self.elements)

    def compute_sconfig(self, df):
        return compute_df_sconfig(df)

    def compute_ROM_compstrs(self, df, key):
        compstrs = df["Composition"].to_numpy()
        roms = []
        for i in range(len(compstrs)):
            compstr = compstrs[i]
            comp = Composition(compstr)
            compstr = comp.reduced_formula
            roms.append(compstr2ROM(compstr, key))
            if i % 1000 == 0: print(f"finished {i} structures - ROM_compstrs!")
        return np.array(roms)

    def compute_ROM_concs(self, concs, key, crystal, SameCrystal=True):
        nele = concs.shape[1]
        if SameCrystal:
            if crystal == "hcp":
                vals = DF_hcp[key].to_numpy()
            else:
                vals = DF_bcc[key].to_numpy()
        else:
            vals = DF_ground[key].to_numpy()
        vals = vals[0:nele]
        roms = np.zeros(concs.shape[0])
        for iele in range(nele):
            roms += concs[:, iele] * vals[iele]
        return roms

    def compute_ROMs_df(self, df, keys, SameCrystal=True, Add_ROM=True):
        concs = self.get_concs(df)
        crystal = df.iloc[0]["Crystal"]
        for ikey in range(len(keys)):
            key = keys[ikey]
            if self.Source[0:3] == "INT":
                ys = self.compute_ROM_concs(concs, key, crystal, SameCrystal=SameCrystal)
            else:
                ys = self.compute_ROM_compstrs(df, key)
            if Add_ROM:
                newkey = key + "_ROM"
            else:
                newkey = key
            df[newkey] = ys
        return df

    def compute_DELTA_concs(self, concs, key, df, style=1):
        refvs = df[key].to_numpy()
        elekey = key2elekey(key)
        nele = concs.shape[1]
        vals = DF_ground[elekey].to_numpy()
        vals = vals[0:nele]
        deltas = np.zeros(concs.shape[0])
        for iele in range(nele):
            if style == 1:
                deltas += concs[:, iele] * np.power((1 - vals[iele] / (refvs + VERY_SMALL_VALUE)), 2)
            elif style == 2:
                deltas += concs[:, iele] * np.power(vals[iele] - refvs, 2)
        deltas = np.sqrt(deltas)
        return deltas

    def compute_DELTAs_df(self, df, keys, style=1, Add_DELTA=True, Recompute=False):
        concs = self.get_concs(df)
        for ikey in range(len(keys)):
            key = keys[ikey]
            if Add_DELTA:
                newkey = key + "_DELTA"
            else:
                newkey = key
            if newkey in df.columns:
                isCompute = False
                if Recompute: isCompute = True
            else:
                isCompute = True
            if isCompute:
                ys = self.compute_DELTA_concs(concs, key, df, style=style)
                df[newkey] = ys
        return df

    def compute_DISTORT_concs(self, concs, key):
        elekey = key2elekey(key)
        nele = concs.shape[1]
        vals = DF_ground[elekey].to_numpy()
        vals = vals[0:nele]
        distorts = np.zeros(concs.shape[0])
        maxdistorts = np.zeros(concs.shape[0])
        for iele in range(nele):
            idistorts = np.zeros(concs.shape[0])
            for jele in range(nele):
                if iele != jele:
                    idistorts += concs[:, jele] * np.power((vals[iele] - vals[jele]) / (vals[iele] + vals[jele]), 2)
            idistorts = np.sqrt(idistorts)
            idistorts = concs[:, iele] * idistorts * 9.0 / 8.0
            maxdistorts = np.select([idistorts<=maxdistorts, idistorts>maxdistorts], [maxdistorts, idistorts])
            distorts += idistorts

        return distorts, maxdistorts

    def compute_DISTORTs_df(self, df, keys, Add_DISTORT=True, Recompute=False):
        concs = self.get_concs(df)
        for ikey in range(len(keys)):
            key = keys[ikey]
            if Add_DISTORT:
                newkey1 = key + "_DISTORT"
                newkey2 = key + "_DISTORT_Max"
            else:
                newkey1 = key
                newkey2 = key + "_Max"
            if newkey1 in df.columns:
                isCompute = False
                if Recompute: isCompute = True
            else:
                isCompute = True
            if isCompute:
                ys1, ys2 = self.compute_DISTORT_concs(concs, key)
                df[newkey1] = ys1
                df[newkey2] = ys2
        return df

    def compute_density(self, df):
        return compute_df_density(df)

    def compute_StrainRoy(self, df, key="radius_TC", Recompute=False):
        outkey = "StrainRoy"
        if outkey in df.columns:
            isCompute = False
            if Recompute: isCompute = True
        else:
            isCompute = True
        if isCompute:
            compstrs = df["Composition"].to_numpy()
            radii = df[key].to_numpy()
            StrainRoys = []
            for icomp in range(len(compstrs)):
                compstr = compstrs[icomp]
                v = compstr2StrainRoy(compstr, key=key, refv=radii[icomp])
                StrainRoys.append(v)
                if icomp % 1000 == 0: print(f"finished {icomp} structures - StrainRoy!")
            df[outkey] = StrainRoys
        return df

    def compute_ElasticEnergy(self, df, style="Exp", Recompute=False):
        outkey = "ElasticEnergy"
        if outkey in df.columns:
            isCompute = False
            if Recompute: isCompute = True
        else:
            isCompute = True
        if isCompute:
            df = compute_df_ElasticEnergy(df, style=style)
        return df

    def compute_SH_ratio(self, df, Recompute=False):
        outkey = "SH_ratio"
        if outkey in df.columns:
            isCompute = False
            if Recompute: isCompute = True
        else:
            isCompute = True
        if isCompute:
            df = compute_df_SH_ratio(df)
        return df

    def compute_gmix_scaled(self, df, temp=300.0, Recompute=False):
        outkey = "Gmix_CPA_SCALED"
        if outkey in df.columns:
            isCompute = False
            if Recompute: isCompute = True
        else:
            isCompute = True
        if isCompute:
            df = compute_df_gmix(df, temp=temp, style="SCALED")
        return df

    def compute_YS_Params(self, df, style="Exp", UseCxx=False, Recompute=False):
        outkey = "sigma_y0"
        if outkey in df.columns:
            isCompute = False
            if Recompute: isCompute = True
        else:
            isCompute = True
        if isCompute:
            df = compute_df_YS_Params(df, style=style, UseCxx=UseCxx)
        return df

    def expand_templevels(self, df, style="Exp", UseCxx=False):
        df = expand_df_templevels(df, style=style, UseCxx=UseCxx, Temp_Level=self.Temp_Level)
        return df

    def compute_YieldStrength(self, df, style="Exp", UseCxx=False, a=0.55,
                              YSstyle=1, strain_ratio=1.0e7, Recompute=False):
        outkey = "Yield_MC"
        if outkey in df.columns:
            isCompute = False
            if Recompute: isCompute = True
        else:
            isCompute = True
        if isCompute:
            df = self.expand_templevels(df, style=style, UseCxx=UseCxx)
            df = compute_df_YieldStrength(df, a=a, style=YSstyle, strain_ratio=strain_ratio)
        return df

    ############################################################
    def select_by_CS(self, df, compspace, condition, ncompon=None, style=0, set2gooddf=False):
        inds = np.arange(len(df), dtype=int)
        compstrs = df['Composition'].to_numpy()
        bads = []
        for icomp in range(len(df)):
            compstr = compstrs[icomp]
            comp = Composition(compstr)
            thissyms = []
            for isym in range(len(comp.elements)):
                thissyms.append(comp.elements[isym].symbol)
            if condition[0:3].upper() == "ALL":
                isValid = True
            else:
                if isinstance(ncompon, int):
                    if len(comp.elements) == ncompon:
                        isValid = True
                    else:
                        isValid = False
                else:
                    isValid = True
                if condition[0:3].upper() == "EXA":
                    if len(comp.elements) != len(compspace):
                        isValid = False

                if isValid:
                    if style == 0:
                        for sym in compspace:
                            if condition[0:3].upper() == "EXC":
                                if sym in thissyms:
                                    isValid = False
                            else:
                                if sym not in thissyms:
                                    isValid = False
                        if not isValid:
                            bads.append(icomp)
                    else:
                        for sym in thissyms:
                            if condition[0:3].upper() == "EXC":
                                if sym in compspace:
                                    isValid = False
                            else:
                                if sym not in compspace:
                                    isValid = False
                        if not isValid:
                            bads.append(icomp)
            if icomp % 10000 == 0: print(f"finished {icomp} structures - select_by_CS!")

        bads = np.array(bads, dtype=int)
        goods = np.delete(inds, bads)
        if set2gooddf:
            self.baddf = df.iloc[bads]
            self.gooddf = df.iloc[goods]
            self.ngood = len(self.gooddf)
            self.compstrs = self.gooddf["Composition"].to_numpy()
        return df.iloc[goods], df.iloc[bads]

    def select_by_ncompons(self, df, ncompons, set2gooddf=False):
        if isinstance(ncompons, int):
            ncompons = [ncompons]
        else:
            ncompons = list(ncompons)
        inds = np.arange(len(df), dtype=int)
        compstrs = df['Composition'].to_numpy()
        bads = []
        for icomp in range(len(df)):
            compstr = compstrs[icomp]
            comp = Composition(compstr)
            thisncompon = len(comp.elements)
            if thisncompon in ncompons:
                pass
            else:
                bads.append(icomp)
        bads = np.array(bads, dtype=int)
        goods = np.delete(inds, bads)
        if set2gooddf:
            self.baddf = df.iloc[bads]
            self.gooddf = df.iloc[goods]
            self.ngood = len(self.gooddf)
            self.compstrs = self.gooddf["Composition"].to_numpy()
        return df.iloc[goods], df.iloc[bads]

    def select_by_DF(self, df, key, condition, set2gooddf=False):
        if key in df.columns:
            values = df[key].to_numpy()
            inds = np.arange(len(df), dtype=int)
            goods = np.compress(eval(condition), inds)
            bads = np.delete(inds, goods)
            if set2gooddf:
                self.baddf = df.iloc[bads]
                self.gooddf = df.iloc[goods]
                self.ngood = len(self.gooddf)
                self.compstrs = self.gooddf["Composition"].to_numpy()
            return df.iloc[goods], df.iloc[bads]
        else:
            raise ValueError("KeyError: key is not found")

    def validate_dataframe(self, df=None, undotype=1, set2data=False):
        if df is None: df = self.df.copy(deep=True)

        key = "sws"
        condition = "values>1.0"
        gooddf, undo1df = self.select_by_DF(df, key, condition, set2gooddf=set2data)
        key = "Bulk"
        condition = "values<500.0"
        gooddf, tmp = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
        undo1df = pd.concat([undo1df, tmp])
        key = "Bulk"
        condition = "values>90.0"
        gooddf, tmp = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
        undo1df = pd.concat([undo1df, tmp])

        key = "Eform"
        condition = "np.absolute(values)<5.0"
        gooddf, tmp = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
        undo1df = pd.concat([undo1df, tmp])

        if "Hmix_TC" in  gooddf.columns:
            key = "Hmix_TC"
            condition = "np.absolute(values)<5.0"
            gooddf, tmp = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
            undo1df = pd.concat([undo1df, tmp])

        undo2df = pd.DataFrame(columns=[self.IndKey] + self.df.columns.tolist())
        undo2df = undo2df.set_index(self.IndKey, inplace=False)

        undo3df = pd.DataFrame(columns=[self.IndKey] + self.df.columns.tolist())
        undo3df = undo3df.set_index(self.IndKey, inplace=False)

        if undotype == 2:
            key = "bcc2hcp"
            condition = "np.absolute(values)<10.0"
            try:
                gooddf, undo2df = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
            except:
                pass
            if "bcc2hcp_Gmix_TC2CPA" in gooddf.columns:
                key = "bcc2hcp_Gmix_TC2CPA"
                condition = "np.absolute(values)<10.0"
                try:
                    gooddf, tmp = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
                    undo2df = pd.concat([undo2df, tmp])
                except:
                    pass
            if "bcc2hcp_Gmix_TC" in gooddf.columns:
                key = "bcc2hcp_Gmix_TC"
                condition = "np.absolute(values)<10.0"
                try:
                    gooddf, tmp = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
                    undo2df = pd.concat([undo2df, tmp])
                except:
                    pass

        if undotype == 3:
            key = "Youngs"
            condition = "np.absolute(values)>1.0"
            try:
                gooddf, undo3df = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
                condition = "np.absolute(values)<750.0"
                gooddf, tmp = self.select_by_DF(gooddf, key, condition, set2gooddf=set2data)
                undo3df = pd.concat([undo3df, tmp])
            except:
                pass

        if set2data:
            self.undo1df = undo1df.copy()
            self.undo2df = undo2df.copy()
            self.undo3df = undo3df.copy()
            self.gooddf = gooddf.copy()
        return undo1df, undo2df, undo3df

    def modified_value_by_compids(self, df, key, compids, values):
        for i in range(len(compids)):
            compid = compids[i]
            df.loc[compid, key] = values[i]
        return df

    def get_terminals(self, compspace, key):
        isEndPointsElement = True
        cs_serial = []
        endpoints = []
        endpoints_value = []
        for i in range(len(compspace)):
            if isinstance(compspace[i], str):
                cs_serial.append(compspace[i])
                sym = compspace[i]
                thisvalue = get_element_value(sym, key, Source=self.Source)
                endpoints_value.append(thisvalue)
                endpoints.append(Composition(sym))
            else:
                a = list(compspace[i])
                s = ""
                y = 0.0
                n = len(a)
                for j in range(n):
                    cs_serial.append(a[j])
                    sym = a[j]
                    thisvalue = get_element_value(sym, key, Source=self.Source)
                    y += thisvalue / n
                    s += sym
                endpoints.append(Composition(s))
                endpoints_value.append(y)
        if len(cs_serial) != len(compspace): isEndPointsElement = False
        return cs_serial, endpoints, endpoints_value, isEndPointsElement

    def get_data4compspace(self, df, compspace, key, condition, key4terminals=None,
                           screen=["inf", "inf"], extended=False, norm=False,
                           validate_data=True, undotype=1, ncompon=None, set2gooddf=False):
        if key4terminals is None:
            key4terminals = key
        compspace = get_sorted_compspace(compspace)

        list_concs = []
        list_ys = []
        list_concs4eval = []
        list_compid4eval = []
        if extended:
            cs_serial, endpoints, endpoints_value, isEndPointsElement = self.get_terminals(compspace, key4terminals)
            if not isEndPointsElement: raise ValueError("the terminals must be elemental")
            for i in range(len(endpoints)):
                thisc = np.zeros(len(endpoints))
                thisc[i] = 1.0
                list_concs.append(thisc)
                list_ys.append(endpoints_value[i])

        df, baddf = self.select_by_CS(df, compspace, condition, ncompon=ncompon, set2gooddf=set2gooddf)
        if validate_data:
            bad1df, bad2df, bad3df = self.validate_dataframe(df=df, undotype=undotype, set2data=False)
            if undotype == 1:
                baddf = bad1df.copy()
            elif undotype == 2:
                baddf = bad2df.copy()
            elif undotype == 3:
                baddf = bad3df.copy()
            undocompids = baddf.index.to_numpy()
        else:
            undocompids = np.array([], dtype=int)

        ys = df[key].to_numpy()
        xs = df["Composition"].to_numpy()
        compids = df.index.to_numpy()

        inds = np.arange(len(ys), dtype=int)
        bads = []
        for i in range(len(xs)):
            compstr = xs[i]
            thisy = ys[i]
            thisconc = compstr2concs(compstr, reduced=True)
            if compids[i] in undocompids:
                isValid = False
                list_concs4eval.append(thisconc)
                list_compid4eval.append(compids[i])
            else:
                isValid = True
                if isinstance(screen[0], float) or isinstance(screen[0], int):
                    if thisy < screen[0]:
                        isValid = False
                        list_concs4eval.append(thisconc)
                        list_compid4eval.append(compids[i])
                        bads.append(i)
                if isinstance(screen[1], float) or isinstance(screen[1], int):
                    if thisy >= screen[1]:
                        isValid = False
                        list_concs4eval.append(thisconc)
                        list_compid4eval.append(compids[i])
                        bads.append(i)
                if isValid:
                    list_concs.append(thisconc)
                    list_ys.append(thisy)

        inds = np.delete(inds, bads)
        xs = xs[inds]
        ys = ys[inds]
        list_concs = np.array(list_concs)
        list_ys = np.array(list_ys)

        list_concs4eval = np.array(list_concs4eval)
        list_compid4eval = np.array(list_compid4eval)

        if norm:
            vmin = np.min(list_ys)
            vmax = np.max(list_ys)
            if vmin == vmax: raise ValueError("Minimum and maximum values are the same.")
            list_ys = value_normalization(list_ys, vmin, vmax)
        return list_concs, list_ys, list_concs4eval, list_compid4eval

    def get_xy4compspace(self, thisdf, compspace, key, key4terminals=None,
                         screen=["inf", "inf"], global_extrema=False):
        if key4terminals is None:
            key4terminals = key
        cs_serial, endpoints, endpoints_value, isEndPointsElement = self.get_terminals(compspace, key4terminals)

        ys = thisdf[key].to_numpy()
        xs = thisdf["Composition"].to_numpy()
        if isinstance(screen[0], float) or isinstance(screen[0], int):
            inds = np.arange(len(ys), dtype=int)
            inds = np.compress(ys > screen[0], inds)
            xs = xs[inds]
            ys = ys[inds]
        if isinstance(screen[1], float) or isinstance(screen[1], int):
            inds = np.arange(len(ys), dtype=int)
            inds = np.compress(ys < screen[1], inds)
            xs = xs[inds]
            ys = ys[inds]

        if global_extrema:
            global_minimum = np.min(ys)
            global_maximum = np.max(ys)
        else:
            global_minimum = None
            global_maximum = None

        inds = np.arange(len(ys), dtype=int)
        bads = []
        for i in range(len(xs)):
            compstr = xs[i]
            thiscomp = Composition(compstr)
            isValid = True
            if len(cs_serial) != len(thiscomp.elements): isValid = False
            if isValid:
                for j in range(len(cs_serial)):
                    if cs_serial[j] not in compstr: isValid = False
            if not isValid: bads.append(i)
        bads = np.array(bads, dtype=int)
        inds = np.delete(inds, bads)
        xs = xs[inds]
        ys = ys[inds]

        outlist = [xs, ys, endpoints, endpoints_value, isEndPointsElement, global_minimum, global_maximum]
        return outlist

    def update_gmix_cpa(self, Temp=300, savefile=False):
        if "Eform" in ndip_all_keys:
            Gmixs = self.df["Eform"].to_numpy() - Temp * self.df["Sconf"].to_numpy()
            self.df["Gmix_CPA"] = Gmixs
            if savefile: self.save_to(self.fname, df=self.df)
        return self.df

    def compute_lattice_parameter(self, savefile=False):
        if "sws" not in self.df.columns:
            return self.df
        else:
            swss = self.df["sws"].to_numpy()
            if "hcp" in self.fname:
                swss = self.df["sws"].to_numpy()
                coas = self.df["COA"].to_numpy()
                V = 4.0 * np.pi / 3.0 * np.power(bohr2angstrom * swss, 3)
                a = 4 * V / np.sqrt(3) / coas
                a = np.power(a, 1.0 / 3.0)
            elif "bcc" in self.fname:
                swss = self.df["sws"].to_numpy()
                V = 4.0 * np.pi / 3.0 * np.power(bohr2angstrom * swss, 3)
                a = np.power(2 * V, 1.0 / 3.0)
            else:
                a = 2 * bohr2angstrom * swss

        self.df["LattPara"] = a
        self.df = self.compute_radius_sws(self.df)
        self.df = self.compute_volume(self.df)
        self.df = self.compute_density(self.df)
        if savefile: self.save_to(self.fname, df=self.df)
        return self.df

    def update_transform_from(self, fname, from_DATA=False, savefile=False, Adjustment=True):
        a0 = -0.03755495
        a1 = 0.93945124
        if "Eform" in ndip_all_keys:
            if "hcp" not in fname: raise ValueError("fname must be hcpCxxxx.csv")
            if "bcc" not in self.fname:
                print("thisdata must be bccCxxxx.csv")
            else:
                hcpdata = myData(fname, from_DATA=from_DATA, Source="INTERNAL")
                self.df = self.df.reset_index().set_index("Composition")
                hcpdf = hcpdata.df.reset_index().set_index("Composition")
                for ind in self.df.index:
                    try:
                        if Adjustment and "from_NDIP" in self.fname:
                            self.df.loc[ind, "bcc2hcp"] = (hcpdf.loc[ind, "Eform"] - self.df.loc[
                                ind, "Eform"]) * a1 + a0
                        else:
                            self.df.loc[ind, "bcc2hcp"] = hcpdf.loc[ind, "Eform"] - self.df.loc[ind, "Eform"]
                    except:
                        pass

                self.df = self.df.reset_index().set_index(self.IndKey)
                if savefile: self.save_to(self.fname, df=None)
        return self.df

    def compute_elastic_props(self, savefile=False):
        if "bcc" not in self.fname: raise ValueError("thisdata must be bccCxxxx.csv")
        if "C11" in ndip_all_keys:
            c11 = self.df["C11"].to_numpy()
            c12 = self.df["C12"].to_numpy()
            c44 = self.df["C44"].to_numpy()

            B = (c11 + 2 * c12) / 3.0

            # Voigt average
            GV = (c11 - c12 + 3 * c44) / 5.0
            EV = 9 * B * GV / (3 * B + GV + 1e-20)
            vV = (3 * B - 2 * GV) / (6 * B + 2 * GV + + 1e-20)
            # Reuss average
            GR = 5 * (c11 - c12) * c44 / (4 * c44 + 3 * (c11 - c12) + 1e-20)
            ER = 9 * B * GR / (3 * B + GR + 1e-20)
            vR = (3 * B - 2 * GR) / (6 * B + 2 * GR + 1e-20)

            # Hill average
            GH = (GV + GR) / 2.0
            EH = 9 * B * GH / (3 * B + GH + 1e-20)
            vH = (3 * B - 2 * GH) / (6 * B + 2 * GH + 1e-20)

            # Elastic anisotropy
            AVR = (GV - GR) / (GV + GR + 1e-20)
            AU = 5 * GV / (GR + 1e-20) + 1.0 - 6.0
            AZ = 2.0 * c44 / (c11 - c12 + 1e-20)

            self.df["Shear"] = GH
            self.df["Youngs"] = EH
            self.df["Poisson"] = vH
            self.df["Au"] = AU
            self.df["Az"] = AZ
            self.df["BOG"] = B / (GH + 1.0e-20)
            if savefile: self.save_to(self.fname, df=None)

        return self.df
