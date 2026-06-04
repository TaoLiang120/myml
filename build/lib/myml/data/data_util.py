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
from myml.myelements.myelements import DF_bcc, DF_hcp, DF_ground, DF_external

from myml.myelements.myelements import omega_binary

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

def value_normalization(thisvals, vmin, vmax):
    v = (thisvals - vmin) / (vmax - vmin)
    v = np.select([v < 0.0, v < 1.0, v >= 1.0], [0.0, v, 1.0])
    return v

def extract_paratheses(s):
    l = len(s)
    matches = []
    start = 0
    end = 0
    while True:
        relative_s = s[start:].find("(")
        relative_e = s[end:].find(")")
        if relative_s == -1 or relative_e == -1:
            break
        matches.append(s[start+relative_s+1:end+relative_e])
        start += 1+relative_s
        end += 1+relative_e
    return matches

def compstr2frac_formula(compstr, decimals=4):
    comp = Composition(compstr)
    newstr = ""
    for iele in range(len(comp.elements)):
        el = comp.elements[iele]
        sym = el.symbol
        frac = comp.get_atomic_fraction(el)
        frac = round(frac, decimals)
        newstr += sym + str(frac)
    return newstr

def normalize_composition(compstr):
    matches = extract_paratheses(compstr)
    if len(matches) == 0:
        compstr = compstr2frac_formula(compstr)
    else:
        for i in range(len(matches)):
            c = matches[i]
            newc = compstr2frac_formula(c)
            compstr = compstr.replace(c, newc)
    return compstr

def get_pretty_formula(compstr, multiplier=1000.0):
    comp = Composition(compstr)
    newstr = ""
    for iele in range(len(comp.elements)):
        el = comp.elements[iele]
        sym = el.symbol
        pct = comp.get_atomic_fraction(el) * multiplier
        pct = int(pct)
        newstr += sym + str(pct)
    comp = Composition(newstr)
    return comp.reduced_formula

def compute_YS_Params_from_2pts(T1, T2, Y1, Y2, a=0.55, strain_ratio=1e7):
    logratio = np.log(strain_ratio)
    r = Y1 / Y2
    delta_Eb0 = kb * logratio / a / np.log(r) * (T2 - T1)
    sigma_y0 = np.exp(-kb * T1 * logratio / a / delta_Eb0)
    sigma_y0 = Y1 / sigma_y0
    return sigma_y0, delta_Eb0


def compute_YS_from_2params(Ts, sigma_y0, delta_Eb0, style=1, a=0.55, strain_ratio=1e7):
    Ts = np.array(Ts)
    logratio = np.log(strain_ratio)
    if style == 1:
        y1 = sigma_y0 * (1.0 - np.power(kb * logratio * Ts / delta_Eb0, 2.0 / 3.0))
        y2 = sigma_y0 * np.exp(-kb * logratio * Ts / a / delta_Eb0)
        ys = np.select([y1 / sigma_y0 < 0.5, y1 / sigma_y0 >= 0.5], [y2, y1])
    else:
        ys = sigma_y0 * np.exp(-kb * logratio * Ts / a / delta_Eb0)
    return ys


def get_element_df():
    df = DF_external.copy(deep=True)
    elements = copy.deepcopy(Element_external)
    return df, elements


def rescale(vals, vmin=-0.20, vmax=0.75):
    vals = np.array(vals)
    thismin = np.min(vals)
    thismax = np.max(vals)
    r = (vals - thismin) / (thismax - thismin)
    v = r * (vmax - vmin) + vmin
    return v


def get_element_value(sym, key, Source="INTERNAL"):
    iele = Element_external.index(sym)
    lat = Element_external_crystal[iele]
    if Source[0:3].upper() == "INT":
        if lat == "hcp":
            df = DF_hcp.copy(deep=True)
        else:
            df = DF_bcc.copy(deep=True)
    else:
        df = DF_external.copy(deep=True)
    elekey = key2elekey(key)
    try:
        v = df.iloc[iele][elekey]
    except:
        v = 0.0
    return v


def compstr2ROM(compstr, key):
    elekey = key2elekey(key)
    df, elements = get_element_df()
    comp = Composition(compstr)
    nele = len(comp.elements)
    thisv = 0.0
    for i in range(nele):
        iele = comp.elements[i]
        ifrac = comp.get_atomic_fraction(iele)
        iind = elements.index(iele.symbol)
        ival = df.iloc[iind][elekey]
        thisv += ival * ifrac
    return thisv


def compstr2DELTA(compstr, key, refv=None, style=1):
    elekey = key2elekey(key)
    if refv is None: refv = compstr2ROM(compstr, key)
    df, elements = get_element_df()
    comp = Composition(compstr)
    nele = len(comp.elements)
    thisv = 0.0
    for i in range(nele):
        iele = comp.elements[i]
        ifrac = comp.get_atomic_fraction(iele)
        iind = elements.index(iele.symbol)
        ival = df.at[iind, elekey]
        if style == 2:
            v = ifrac * np.power(ival - refv, 2)
        else:
            v = ifrac * np.power(1.0 - ival / (refv + 1e-20), 2)
        thisv += v
    thisv = np.sqrt(thisv)
    return thisv


def compstr2DISTORT(compstr, key):
    elekey = key2elekey(key)
    df, elements = get_element_df()
    comp = Composition(compstr)
    nele = len(comp.elements)
    thisv = 0.0
    maxval = 0.0
    for i in range(nele):
        ival = 0.0
        iele = comp.elements[i]
        ifrac = comp.get_atomic_fraction(iele)
        iind = elements.index(iele.symbol)
        iv = df.at[iind, elekey]
        for j in range(nele):
            jele = comp.elements[j]
            jfrac = comp.get_atomic_fraction(jele)
            jind = elements.index(jele.symbol)
            jv = df.at[jind, elekey]
            v = jfrac * np.power((iv - jv) / (iv + jv), 2)
            ival += v
        ival = np.sqrt(ival)
        ival = ival * ifrac * 9.0 / 8.0
        thisv += ival
        if ival > maxval:
            maxval = ival
    return thisv, maxval


def compstr2StrainRoy(compstr, key="radius_TC", refv=None):
    def compute_extreme(v, refv):
        v1 = np.power(v + refv, 2)
        v2 = np.power(refv, 2)
        v3 = 1.0 - np.sqrt((v1 - v2) / v1)
        return v3

    if refv is None: refv = compstr2ROM(compstr, key)
    df, elements = get_element_df()
    comp = Composition(compstr)
    nele = len(comp.elements)
    rads = []
    for i in range(nele):
        iele = comp.elements[i]
        iind = elements.index(iele.symbol)
        rads.append(df.at[iind, key])
    rads = np.array(rads)
    rmin = np.min(rads)
    rmax = np.max(rads)
    vmin = compute_extreme(rmin, refv)
    vmax = compute_extreme(rmax, refv)
    thisv = (1 - vmin) / (1 - vmax)
    return thisv


def compstr2ElasticEnergy(compstr, style="Exp"):
    df, elements = get_element_df()
    def get_EE(iind, jind, iele, jele, ifrac, jfrac, style="Exp"):
        if "EXP" in style.upper():
            header = "Exp_"
        else:
            header = ""
        ibulk = df.at[iind, header + "Bulk"]
        jbulk = df.at[jind, header + "Bulk"]
        ishear = df.at[iind, header + "Shear"]
        jshear = df.at[jind, header + "Shear"]
        ivol = df.at[iind, "volume"]
        jvol = df.at[jind, "volume"]
        thisctot = (ifrac + jfrac)
        ih = 2.0 * ibulk * jshear * np.power(ivol - jvol, 2) / (4.0 * jshear * ivol + 3.0 * ibulk * jvol)
        ih = ih * jfrac / thisctot
        jh = 2.0 * jbulk * ishear * np.power(jvol - ivol, 2) / (4.0 * ishear * jvol + 3.0 * jbulk * ivol)
        jh = jh * ifrac / thisctot
        return ih + jh

    comp = Composition(compstr)
    nele = len(comp.elements)
    thisv = 0.0
    if nele >= 2:
        combs = combinations(np.arange(nele), 2)
        totc = 0.0
        for comb in list(combs):
            i = comb[0]
            j = comb[1]
            iele = comp.elements[i]
            iind = elements.index(iele.symbol)
            ifrac = comp.get_atomic_fraction(iele)
            jele = comp.elements[j]
            jind = elements.index(jele.symbol)
            jfrac = comp.get_atomic_fraction(jele)
            h = get_EE(iind, jind, iele, jele, ifrac, jfrac, style=style)
            thisctot = ifrac + jfrac
            h = h * thisctot
            thisv += h
            totc += thisctot
        thisv /= totc
    return thisv * GPa2eVA


def compstr2YS_Params(compstr, ind, df, style="Exp", UseCxx=False):
    A_sigma = 0.04
    a = 1.0 / 12.0
    AE = 2.0
    df_ele, elements = get_element_df()
    if "EXP" in style.upper():
        pre = "Exp_"
        app = "_ROM"
        esapp = app
    elif "ORG" in style.upper():
        pre = ""
        app = "_ORG"
        esapp = app
    else:
        pre = ""
        app = ""
        esapp = app

    if UseCxx:
        c11 = df.iloc[ind][pre + "C11" + esapp]
        c12 = df.iloc[ind][pre + "C12" + esapp]
        c44 = df.iloc[ind][pre + "C44" + esapp]
        shear = np.sqrt(c44 * (c11 - c12) / 2.0)
        bulk = (c11 + 2.0 * c12) / 3.0
        poisson = (3 * bulk - 2 * shear) / 2 / (3 * bulk + shear)
    else:
        shear = df.iloc[ind][pre + "Shear" + esapp]
        poisson = df.iloc[ind][pre + "Poisson" + app]

    volume = df.iloc[ind]["volume"] + VERY_SMALL_VALUE
    burger = np.sqrt(3) / 2.0 * np.power(2 * volume, 1.0 / 3.0)

    sigma_y0 = 3.067 * A_sigma * np.power(a, -1.0 / 3.0) * shear
    sigma_y0 *= np.power((1 + poisson) / (1 - poisson), 4.0 / 3.0)

    delta_Eb0 = AE * np.power(a, 1.0 / 3.0) * shear * np.power(burger, 3)
    delta_Eb0 *= np.power((1 + poisson) / (1 - poisson), 2.0 / 3.0)

    comp = Composition(compstr)
    nele = len(comp.elements)
    volume_misfit = 0.0
    for i in range(nele):
        iele = comp.elements[i]
        ifrac = comp.get_atomic_fraction(iele)
        iind = elements.index(iele.symbol)
        ival = df_ele.at[iind, "volume"]
        v = ifrac * np.power(ival - volume, 2) / np.power(burger, 6)
        volume_misfit += v

    sigma_y0 *= 1000.0 * np.power(volume_misfit, 2.0 / 3.0)
    delta_Eb0 *= GPa2eVA * np.power(volume_misfit, 1.0 / 3.0)
    return sigma_y0, delta_Eb0


def compstr2eform_omegas(compstr, crystal="BCC", style=1):
    if "BCC" in crystal:
        strkey = "BCC"
    elif "FCC" in crystal:
        strkey = "FCC"
    else:
        strkey = "BCC"
    comp = Composition(compstr)
    nele = len(comp.elements)
    combs = combinations(np.arange(nele), 2)
    v = 0.0
    ctot = 0.0
    #ctot3 = 0.0
    for comb in list(combs):
        ii = comb[0]
        jj = comb[1]
        isym = comp.elements[ii].symbol
        jsym = comp.elements[jj].symbol
        ifrac = comp.get_atomic_fraction(comp.elements[ii])
        jfrac = comp.get_atomic_fraction(comp.elements[jj])
        cijtot = (ifrac + jfrac)
        ctot += cijtot
        a = [isym, jsym]
        a.sort()
        key = a[0] + "-" + a[1]
        try:
            vij = omega_binary["omegas"][strkey][key]
        except:
            vij = 0.0
        if style == 1:
            v += ifrac * jfrac * vij
        else:
            v += cijtot * vij / 4.0
    if style == 2: v /= ctot
    return v


def compstr2conc_delta(compstr):
    comp = Composition(compstr)
    nele = len(comp.elements)
    if nele == 1:
        v = 1
    else:
        combs = combinations(np.arange(nele), 2)
        v = 0.0
        ctot = 0.0
        for comb in list(combs):
            ii = comb[0]
            jj = comb[1]
            ifrac = comp.get_atomic_fraction(comp.elements[ii])
            jfrac = comp.get_atomic_fraction(comp.elements[jj])
            cijtot = (ifrac + jfrac)
            ctot += cijtot
            vij = 2.0 * np.absolute(ifrac / cijtot - 0.5)
            v += cijtot * vij
        v /= ctot
    return v


def compute_eform_omegas(df, OutKey="Eform_BOKAS", style=1):
    compstrs = df["Composition"].to_numpy()
    if "Crystal" in df.columns:
        crystals = df["Crystal"].to_numpy()
    else:
        crystals = ["BCC"] * len(df)

    eforms = []
    for i in range(len(compstrs)):
        compstr = compstrs[i]
        cry = str(crystals[i]).upper()
        v = compstr2eform_omegas(compstr, crystal=cry, style=style)
        eforms.append(v)
        if i % 1000 == 0: print(f"finished {i} structures - eform omegas!")
    df[OutKey] = eforms
    return df


def compute_df_eform_scaled(df, app="_SCALED", Keys=None):
    KEYS = ["Eform", "Eform_str"]
    ps = np.array([[-0.05517239, 0.41543859], [-0.0459761, 0.43046792]])
    if Keys is None:
        Keys = KEYS[0:len(KEYS)]
    elif isinstance(Keys, str):
        Keys = [Keys]
    for key in Keys:
        ikey = KEYS.index(key)
        thisk = KEYS[ikey]
        x = df[thisk].to_numpy()
        p = ps[ikey, :]
        y = p[0] + p[1] * x
        outkey = thisk + app
        df[outkey] = y
    return df


def normalize_df_composition(df):
    compstrs = df['Composition'].to_numpy()
    df['Composition'] = [normalize_composition(s) for s in compstrs]
    return df


def get_df_pretty_formula(df, normalization=True):
    '''
    Args:
        df:
        normalization: if True, normalize the compositions inside paratheses to 1.0
        In pymatgen reduced formulas, it should be set to False
    Returns:
    '''
    if normalization:
        df = normalize_df_composition(df)
    compstrs = df['Composition'].to_numpy()
    df['Composition'] = [get_pretty_formula(s) for s in compstrs]
    return df

def compute_df_PMGROM(df, key):
    compstrs = df["Composition"].tolist()
    roms = []
    for icomp in range(len(compstrs)):
        compstr = compstrs[icomp]
        comp = Composition(compstr)
        thisv = compstr2PMGROM(compstr, key)
        roms.append(thisv)
        if icomp % 1000 == 0: print(f"finished {icomp} structures - PMGROM!")
    return np.array(roms)


def compute_df_radius_sws(df):
    if "sws" in df.columns:
        swss = df["sws"].to_numpy()
        radii = swss * bohr2angstrom
    else:
        radii = compute_df_PMGROM(df, "radius")
    df["radius"] = radii
    return df


def compute_df_volume(df, latname="bcc"):
    if "radius_TC" in df.columns:
        r = df["radius_TC"].to_numpy()
        vols = 4.0 * np.pi * np.power(r, 3) / 3.0
    else:
        r = df["radius"].to_numpy()
        vols = 4.0 * np.pi * np.power(r, 3) / 3.0
    df["volume"] = vols
    return df


def compute_df_reducedformula(df):
    compstrs = df["Composition"].to_numpy()
    rcompstrs = []
    for i in range(len(df)):
        compstr = compstrs[i]
        comp = Composition(compstr)
        rcompstr = comp.reduced_formula
        rcompstrs.append(rcompstr)
        if i % 1000 == 0: print(f"finished {i} structures - reduced_formula!")
    df["ReducedFormula"] = rcompstrs
    return df


def compute_df_conc_delta(df):
    compstrs = df["Composition"].to_numpy()
    vs = []
    for i in range(len(df)):
        compstr = compstrs[i]
        vs.append(compstr2conc_delta(compstr))
        if i % 1000 == 0: print(f"finished {i} structures - conc_delta!")
    df["c_DELTA"] = vs
    return df


def compute_df_elasticscale(df, maxscale=0.15):
    if "c_DELTA" not in df.columns:
        df = compute_df_conc_delta(df)
    c = df["c_DELTA"].to_numpy()
    vs = -maxscale * np.power(c, 2) + maxscale
    df["Elastic_Scale"] = 1.0 - vs
    return df


def compute_df_scaled_elastic(df, keys):
    if "Elastic_Scale" not in df.columns:
        df = compute_df_elasticscale(df)
    scales = df["Elastic_Scale"].to_numpy()
    for key in keys:
        if key in df.columns:
            vals = df[key].to_numpy()
            vals = vals * scales
            outkey = key + "_SCALED"
            df[outkey] = vals
    return df


def compute_df_concs(df, elements=Element_negativity):
    if "ReducedFormula" in df.columns:
        compstrs = df["ReducedFormula"].to_numpy()
    else:
        compstrs = df["Composition"].to_numpy()

    list_concs = []
    for i in range(len(df)):
        compstr = compstrs[i]
        concs = compute_sequential_concs(compstr, elements=elements)
        list_concs.append(concs)
        if i % 1000 == 0: print(f"finished {i} structures - df concs!")

    list_concs = np.array(list_concs)
    for iele in range(nele_default):
        thiskey = "CONC" + str(iele)
        df[thiskey] = list_concs[:, iele]
    return df


def get_df_concs(df):
    keys = []
    for key in df.columns:
        if "CONC" == key[0:4]: keys.append(key)
    for ikey in range(len(keys)):
        key = keys[ikey]
        if ikey == 0:
            concs = df[key].to_numpy()
        else:
            concs = np.vstack([concs, df[key].to_numpy()])
    concs = concs.T
    return concs


def compute_df_sconfig(df):
    concs = get_df_concs(df)
    nele = concs.shape[1]
    sconfs = np.zeros(concs.shape[0])
    for iele in range(nele):
        sconfs -= kb * concs[:, iele] * np.log(concs[:, iele] + 1.0e-20)
    df["Sconf"] = sconfs
    return df


def compute_df_temp_ratio(df, cap=True):
    temps = df["temperature"]
    Tms = df["Tm_ROM"].to_numpy()
    tr = temps / Tms
    if cap: tr = np.select([tr < 1.0, tr >= 1.0], [tr, 1.0])
    df["temp_ratio"] = tr
    return df


def compute_df_density(df):
    amasses = df["amass"].to_numpy()
    vols = df["volume"].to_numpy() + 1.0e-20
    dens = density2gcm * amasses / vols
    df["density"] = dens
    return df


def compute_df_SH_ratio(df):
    s = df["Sconf"].to_numpy()
    t = df["Tm_ROM"].to_numpy()
    h = df["Hmix_TC"].to_numpy()
    v = s * t / (np.absolute(h) + VERY_SMALL_VALUE)
    df["SH_ratio"] = v
    return df


def compute_df_gmix(df, temp=300.0, style="SCALE"):
    s = df["Sconf"].to_numpy()
    if "SCALE" in style.upper():
        e = df["Eform_SCALED"].to_numpy()
        outkey = "Gmix_CPA_SCALED"
    else:
        e = df["Eform"].to_numpy()
        outkey = "Gmix_CPA"
    y = e - s * temp
    df[outkey] = y
    return df


def compute_df_ElasticEnergy(df, style="Exp"):
    compstrs = df["Composition"].to_numpy()
    ees = []
    for icomp in range(len(compstrs)):
        compstr = compstrs[icomp]
        v = compstr2ElasticEnergy(compstr, style=style)
        ees.append(v)
        if icomp % 1000 == 0:
            print("Finished " + str(icomp) + " compositions")
    df["ElasticEnergy"] = ees
    return df


def compute_df_composite_ductility(df):
    if "Exp_Shear_ROM" in df.columns:
        Gs = df["Exp_Shear_ROM"].to_numpy()
    else:
        Gs = df["Exp_Shear"].to_numpy()

    if "Exp_Surf_ROM" in df.columns:
        Ss = df["Exp_Surf_ROM"].to_numpy()
    else:
        Ss = df["Exp_Surf"].to_numpy()

    if "Cal_Surf_ROM" in df.columns:
        cSs = df["Cal_Surf_ROM"].to_numpy()
    else:
        cSs = df["Cal_Surf"].to_numpy()

    if "D_Surf_GBR_ROM" in df.columns:
        dSGBs = df["D_Surf_GBR_ROM"].to_numpy()
    else:
        dSGBs = df["D_Surf_GBR"].to_numpy()

    if "E_USF_ROM" in df.columns:
        USFs = df["E_USF_ROM"].to_numpy()
    else:
        USFs = df["E_USF"].to_numpy()
    if "volume_TC" in df.columns:
        bs = 2.0 * df["volume_TC"].to_numpy()
    else:
        bs = 2.0 * df["volume"].to_numpy()
    bs = np.power(bs, 1.0/3.0)
    bs = np.sqrt(3) * bs/2.0

    df["ShearBurger"] = 0.1 * Gs * bs
    df["ShearBurger_Surf"] = 0.1 * Gs * bs / Ss
    df["ShearBurger_D_Surf_GBR"] = 0.1 * Gs * bs / dSGBs
    df["Surf_USF"] = cSs / USFs

    if "DOS_EF_ROM" in df.columns:
        DOS_EFs = df["DOS_EF_ROM"].to_numpy()
    else:
        DOS_EFs = df["DOS_EF"].to_numpy()
    if "VEC" in df.columns:
        VECs = df["VEC"].to_numpy()
    else:
        VECs = np.ones(len(df))
    df["DOS_VEC"] = DOS_EFs/VECs
    return df


def compute_df_composite_Poisson(df, style="Exp"):
    p = df["Exp_Poisson_ROM"].to_numpy()
    v = (1 + p) / (1 - p)
    df["Exp_Poisson_Comp"] = v
    return df


def compute_df_composite_Vdiff(df, style=1):
    if style == 1:
        vs = df["volume_DELTA"].to_numpy()
        df["volume_DELTA_Comp"] = 1.0 + vs
    elif style == 2:
        vs = df["volume_DISTORT"].to_numpy()
        df["volume_DISTORT_Comp"] = 1.0 + vs
    return df


def compute_df_composite_keys(df, compute_vdiff=False):
    df = compute_df_composite_ductility(df)
    df = compute_df_composite_Poisson(df)
    if compute_vdiff:
        df = compute_df_composite_Vdiff(df, style=1)
        df = compute_df_composite_Vdiff(df, style=2)
    return df


def compute_df_YS_Params(df, style="Exp", UseCxx=False):
    compstrs = df["Composition"].to_numpy()
    sigma_y0s = []
    delta_Eb0s = []
    for ind in range(len(df)):
        compstr = compstrs[ind]
        sigma_y0, delta_Eb0 = compstr2YS_Params(compstr, ind, df, style=style, UseCxx=UseCxx)
        sigma_y0s.append(sigma_y0)
        delta_Eb0s.append(delta_Eb0)
        if ind % 1000 == 0: print(f"finished {ind} structures - YS_Params!")
    df["sigma_y0"] = sigma_y0s
    df["delta_Eb0"] = delta_Eb0s
    return df


def expand_df_templevels(df, style="Exp", UseCxx=False, Temp_Level=TEMPERATURE_LEVEL):
    if "sigma_y0" not in df.columns:
        compute_df_YS_Params(df, style=style, UseCxx=UseCxx)
    norg = len(df)
    dforg = df.copy(deep=True)
    temps = []
    for i in range(len(Temp_Level)):
        if i > 0: df = pd.concat([df, dforg])
        temps += [Temp_Level[i]] * norg
    temps = np.array(temps)
    df["temperature"] = temps
    df = compute_df_temp_ratio(df)
    if df.index.name == "CompID":
        df.index = np.arange(len(df), dtype=int)
        df.index.name = "CompID"
    dropcols = []
    if "index" in df.columns: dropcols.append("index")
    if "level_0" in df.columns: dropcols.append("level_0")
    if len(dropcols) > 0: df = df.drop(columns=dropcols)
    return df


def compute_df_YieldStrength(df, a=0.55, style=1, strain_ratio=1.0e7, thres=1.0):
    logratio = np.log(strain_ratio)
    sigma_y0s = df["sigma_y0"].to_numpy()
    delta_Eb0s = df["delta_Eb0"].to_numpy()
    temps = df["temperature"].to_numpy()
    temp_ratios = df["temp_ratio"].to_numpy()
    if style == 1:
        y1 = sigma_y0s * (1 - np.power((kb * temps * logratio / delta_Eb0s), 2.0 / 3.0))
        y2 = sigma_y0s * np.exp(-kb * temps * logratio / (a * delta_Eb0s))
        y = np.select([y1 / sigma_y0s < 0.5, y1 / sigma_y0s >= 0.5], [y2, y1])
    else:
        y = sigma_y0s * np.exp(-kb * temps * logratio / (a * delta_Eb0s))
    y = np.select([temp_ratios < thres, temp_ratios >= thres], [y, 0.0])
    df["Yield_MC"] = y
    return df
