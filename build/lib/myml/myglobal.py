import os, sys
from monty.serialization import loadfn

import numpy as np
import json
from itertools import combinations
from pymatgen.core.composition import Composition

try:
    config_vars = loadfn(os.path.join(os.path.expanduser('~'), 'myml4hea.yaml'))
except:
    sys.exit('No myml4hea.yaml file was found. Please configure the '
             ' myml4hea.yaml and put it in your home directory.')

VERY_SMALL_VALUE = 1.0e-40

Constants = {"kb": 8.6173324E-5, "bohr2angstrom": 0.529177249, "density2gcm": 1.6605402,
             "eVA2GPa": 160.2177, "GPa2eVA": 0.00624150648, "KJ2eVA": 1.0/96.4915666370759,
             "J2eVA": 1.0/96491.5666370759,  "TC2CPA_volume": 1660539.0671738465, "float_format": "%.8f"}

b2h4ml = "bcc2hcp"
b2h4phase = "bcc2hcp_Gmix_TC"
thres4b2h = 0.25 # 0.25 if b2h4ml is "bcc2hcp_Gmix_TC2CPA"

fTMPCSV = "tmp.csv"
ntemperatures = 8
TEMPERATURE_LEVEL = np.linspace(0.0, 2100.0, ntemperatures)
Element_Basic = ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"]
Element_Additional = ["Al", "Co", "Fe", "Ni", "Mn", "Re"]
Element_Extend = ["Sc", "Cu", "Zn", "Y", "Sn", "Pd"]
Element_Add_Ext = Element_Additional + Element_Extend

Element_list = Element_Basic + Element_Additional
Element_negativity = ["Hf", "Zr", "Ta", "Ti", "Mn", "Nb", "Al", "V", "Cr", "Fe", "Co", "Re", "Ni", "Mo", "W"]

Element_crystal = ["hcp", "hcp", "bcc", "hcp", "hcp", "bcc", "hcp", "bcc", "bcc",
                   "bcc", "hcp", "hcp", "hcp", "bcc", "bcc"]
Element_groups = [["Ti", "Zr", "Hf", "Al"], ["V", "Nb", "Ta", "Fe", "Mn", "Co", "Ni"], ["Cr", "Mo", "W", "Re"]]

Element_external = Element_negativity + Element_Extend
Element_external_crystal = Element_crystal + ["hcp", "fcc", "hcp", "hcp", "ct", "fcc"]

Element_sublatt = ['Ni', 'Co', 'Fe', 'Cu', 'Mn', 'Cr', 'V', 'Re', 'Zn', 'Mo', 'W', 'Al', 'Ti', 'Ta', 'Nb', 'Hf', 'Zr']

nele_default = len(Element_negativity)

Basic_ROM_Keys = ["radius", "radius_TC", "amass", "VEC", "E_negativity", "bcc2hcp_Gmix_TC_diff"]
Add_ROM_Keys = ["Tm", "Exp_Bulk", "Exp_Shear", "Exp_Youngs", "Exp_Poisson",
                "Exp_Surf", "Cal_Surf", "E_GBR", "DOS_EF", "E_USF", "Exp_Az", "b2h", "D_Surf_GBR"]
DELTA_Keys = ["radius_TC", "volume", "Exp_Shear_ROM", "Exp_Youngs_ROM"] ##, "Exp_Bulk_ROM"]
DISTORT_Keys = ["radius_TC", "volume", "Exp_Shear", "Exp_Youngs"] ##, "Exp_Bulk"]
SCALED_Keys = ["Exp_Bulk_ROM", "Exp_Shear_ROM", "Exp_Youngs_ROM"]

Key2PMG = {
    "radius": "Metallic radius",
    "E_negativity": "X",
    "amass": "Atomic mass",
    "Bulk": "Bulk modulus",
    "Youngs": "Youngs modulus",
    "Poisson": "Poissons ratio",
    "Tm": "Melting point",
}

def key2elekey(key):
    elekey = key
    elekey = elekey.replace("_ROM", "")
    elekey = elekey.replace("_DELTA1", "")
    elekey = elekey.replace("_DELTA2", "")
    elekey = elekey.replace("_DELTA", "")
    elekey = elekey.replace("_DISTORT1", "")
    elekey = elekey.replace("_DISTORT2", "")
    elekey = elekey.replace("_DISTORT", "")
    elekey = elekey.replace("_ORG", "")
    elekey = elekey.replace("_SCALED", "")
    if elekey == "bcc2hcp_Gmix_TC_diff":
        elekey = "bcc2hcp_Gmix_TC_diff"
    elif "bcc2hcp" in elekey:
        elekey = "bcc2hcp"
    elif "BOKAS" in elekey:
        elekey = "Eform_str"
    elif "radius_TC" in elekey:
        elekey = "radius_TC"
    elif "bcc2hcp_Gmix_TC" in elekey:
        elekey = "bcc2hcp_Gmix_TC"
    elif "bcc2hcp_Gmix_TC_diff" in elekey:
        elekey = "bcc2hcp_Gmix_TC"
    elif "Hmix_str" in elekey:
        elekey = "Eform_str"
    elif "Gmix_str" in elekey:
        elekey = "Eform_str"
    elif "Hmix" in elekey:
        elekey = "Eform"
    elif "Gmix" in elekey:
        elekey = "Eform"
    return elekey

def compstr2PMGROM(compstr, key):
    pmgkey = Key2PMG[key]
    comp = Composition(compstr)
    nele = len(comp.elements)
    thisv = 0.0
    for i in range(nele):
        iele = comp.elements[i]
        ifrac = comp.get_atomic_fraction(iele)
        ival = iele.data[pmgkey]
        thisv += ival * ifrac
    return thisv


def load_complist_from_json(fname):
    with open(fname, "r") as f:
        s = f.read()
    complist = json.loads(s)
    return complist


def compstr2compid(compstr, latname):
    latname2id = {"bcc": 1, "hcp": 2, "b2": 3, "laves": 4, "mix": 5}
    comp = Composition(compstr)
    thisstr = str(latname2id[latname])
    for i, ele in enumerate(comp):
        try:
            iele = Element_external.index(ele.symbol)
        except:
            iele = 0
        ifrac = comp.get_atomic_fraction(ele)
        ifrac = int(round(ifrac * 100, 0))
        if ifrac == 100: ifrac = 0
        thisstr += str(iele + 1) + "{:02}".format(ifrac)
    return int(thisstr)


def compstr2sconf(compstr):
    comp = Composition(compstr)
    s = 0.0
    for i, ele in enumerate(comp):
        ifrac = comp.get_atomic_fraction(ele)
        s -= Constants["kb"] * ifrac * np.log(ifrac + 1.0e-20)
    return s


def compstr2entropy(compstr, Temp=300):
    s = compstr2sconf(compstr)
    return s * Temp


def compstr2concs(compstr, reduced=True):
    comp = Composition(compstr)
    if reduced:
        rcompstr = comp.reduced_formula
        comp = Composition(rcompstr)
    concs = []
    for el in comp.elements:
        concs.append(comp.get_atomic_fraction(el))
    concs = np.array(concs)
    return concs


def compute_sequential_concs(compstr, elements=Element_negativity):
    comp = Composition(compstr)
    if elements is None:
        concs = [0.0] * nele_default
    else:
        concs = [0.0] * len(elements)
    for i, ele in enumerate(comp):
        try:
            iele = elements.index(ele.symbol)
        except:
            iele = i
        frac = comp.get_atomic_fraction(ele)
        concs[iele] = round(frac, 3)
    return concs


def get_sorted_compspace(compspace):
    csstr = ""
    for sym in compspace:
        csstr += sym
    comp = Composition(csstr)
    rstr = comp.reduced_formula
    comp = Composition(rstr)
    compspace = []
    for el in comp.elements:
        compspace.append(el.symbol)
    return compspace


def get_compspaces(ncompon, elements=Element_negativity):
    nele = len(elements)
    inds = np.arange(nele, dtype=int)
    compspaces = []
    combs = combinations(inds, ncompon)
    for comb in list(combs):
        thiscs = []
        for i in range(ncompon):
            ii = comb[i]
            thiscs.append(elements[ii])
        compspaces.append(get_sorted_compspace(thiscs))
    return compspaces
