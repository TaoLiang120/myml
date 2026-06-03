import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
from myml.myglobal import config_vars, Constants
from myml.myglobal import Element_Basic, Element_list, Element_crystal
from myml.plot.plot import plot_linear_xys, get_3d_bar
from myml.expansion.expansion import compstr2NDIP_keys

TC_Elements_list = Element_list + ["Cu", "Zn", "Sn", "Re"]
TC_Elements_crystal = Element_crystal + ["hcp", "hcp", "hcp", "hcp"]
TC_Elements_binary = Element_list + ["Re"]
TC_Elements_additional = ["Sn", "Re"]
TEMP_LEVELS = np.array([300, 1000], dtype=int)

PHASES = ['BCC_A2', 'BCC_B2', 'C14_LAVES', 'C15_LAVES', 'HCP_A3', 'LIQUID']
QUANTITIES = ["VM", "GM", "HM", "SM"]
TC_KEYS = ["Composition", "ReducedFormula", "Temperature"]
for i in range(len(PHASES)):
    p = PHASES[i]
    for j in range(len(QUANTITIES)):
        q = QUANTITIES[j]
        key = p + "_" + q
        TC_KEYS.append(key)
TC_KEYS2 = ["Composition", "ReducedFormula", "Temperature", "Phase", "GM", "HM", "SM"]
fTC_elements = "TC_elements_db.csv"

IMP_VALUE = 999999.0
IMP_VALUE4DFT = 1000.0
INDEXKEY = "rcomp_temp"


def TC_elements_raw2db(elements=TC_Elements_list, temps=300.0):
    if isinstance(temps, float) or isinstance(temps, int):
        temps = [temps]
    temps = list(temps)
    for i in range(len(temps)):
        temps[i] = int(temps[i])
    fname = "TC_elements_raw.csv"
    df = pd.read_csv(fname)

    dfcpa = pd.read_csv("External.csv")
    cpaeles = dfcpa["Composition"].tolist()
    for i in range(len(cpaeles)):
        cpaeles[i] = cpaeles[i].replace("100","")

    outcols = [INDEXKEY, "Composition", "ReducedFormula", "Crystal", "Temperature",
               "VM", "GM", "HM", "SM", "volume_TC", "radius_TC",
               "volume_TC_hcp",  "radius_TC_hcp",
               "GM_hcp", "HM_hcp", "SM_hcp",
               "Gmix_TC", "Hmix_TC", "Smix_TC",
               "bcc2hcp_Gmix_TC", "bcc2hcp_Hmix_TC", "bcc2hcp_Smix_TC",
               "bcc2hcp_cpa", "bcc2hcp_Gmix_TC_diff", "bcc2hcp_Hmix_TC_diff",
               "volume_TC_laves", "radius_TC_laves",
               "bcc2laves_Gmix_TC", "bcc2laves_Hmix_TC", "bcc2laves_Smix_TC"]
    nele = len(elements)
    ntemp = len(temps)
    d = np.zeros([nele * ntemp, len(outcols)])
    outdf = pd.DataFrame(d, columns=outcols)
    outdf[INDEXKEY] = outdf[INDEXKEY].astype(str)
    outdf["Composition"] = outdf["Composition"].astype(str)
    outdf["ReducedFormula"] = outdf["ReducedFormula"].astype(str)
    outdf["Crystal"] = outdf["Crystal"].astype(str)
    for i in range(len(df)):
        ele = df.at[i, "ReducedFormula"]
        thistemp = df.at[i, "Temperature"]
        iele = elements.index(ele)
        itemp = temps.index(int(thistemp))
        ii = itemp * nele + iele
        outdf.at[ii, INDEXKEY] = df.at[i, "ReducedFormula"] + "_" + str(int(thistemp))
        outdf.at[ii, "Composition"] = df.at[i, "Composition"]
        outdf.at[ii, "ReducedFormula"] = df.at[i, "ReducedFormula"]
        outdf.at[ii, "Crystal"] = "bcc"
        outdf.at[ii, "Temperature"] = thistemp
        vmb = df.at[i, "BCC_B2_VM"]
        gmb = df.at[i, "BCC_B2_GM"]
        hmb = df.at[i, "BCC_B2_HM"]
        smb = df.at[i, "BCC_B2_SM"]
        vmh = df.at[i, "HCP_A3_VM"]
        gmh = df.at[i, "HCP_A3_GM"]
        hmh = df.at[i, "HCP_A3_HM"]
        smh = df.at[i, "HCP_A3_SM"]

        outdf.at[ii, "VM"] = vmb
        outdf.at[ii, "GM"] = gmb
        outdf.at[ii, "HM"] = hmb
        outdf.at[ii, "SM"] = smb
        avb = vmb * Constants["TC2CPA_volume"]
        rb = avb * 3.0 / 4.0 / np.pi
        rb = np.power(rb, 1.0/3.0)
        outdf.at[ii, "volume_TC"] = avb
        outdf.at[ii, "radius_TC"] = rb
        if vmb == 0.0:
            print(f"element: {ele} in bcc has no volume output.")
        avh = vmh * Constants["TC2CPA_volume"]
        rh = avh * 3.0 / 4.0 / np.pi
        rh = np.power(rh, 1.0/3.0)
        outdf.at[ii, "volume_TC_hcp"] = avh
        outdf.at[ii, "radius_TC_hcp"] = rh
        if vmh == 0.0:
            print(f"element: {ele} in hcp has no volume output.")


        outdf.at[ii, "GM_hcp"] = gmh
        outdf.at[ii, "HM_hcp"] = hmh
        outdf.at[ii, "SM_hcp"] = smh
        outdf.at[ii, "Gmix_TC"] = gmb * Constants["J2eVA"]
        outdf.at[ii, "Hmix_TC"] = hmb * Constants["J2eVA"]
        outdf.at[ii, "Smix_TC"] = smb * Constants["J2eVA"]
        outdf.at[ii, "bcc2hcp_Gmix_TC"] = (gmh - gmb) * Constants["J2eVA"]
        outdf.at[ii, "bcc2hcp_Hmix_TC"] = (hmh - hmb) * Constants["J2eVA"]
        outdf.at[ii, "bcc2hcp_Smix_TC"] = (smh - smb) * Constants["J2eVA"]

        icpa = cpaeles.index(ele)
        b2h = dfcpa.at[icpa, "bcc2hcp"]
        outdf.at[ii, "bcc2hcp_cpa"] = b2h
        outdf.at[ii, "bcc2hcp_Gmix_TC_diff"] = b2h - (gmh - gmb) * Constants["J2eVA"]
        outdf.at[ii, "bcc2hcp_Hmix_TC_diff"] = b2h - (hmh - hmb) * Constants["J2eVA"]

        vml = df.at[i, "C14_LAVES_VM"]
        gml = df.at[i, "C14_LAVES_GM"]
        hml = df.at[i, "C14_LAVES_HM"]
        sml = df.at[i, "C14_LAVES_SM"]
        avl = vml * Constants["TC2CPA_volume"]
        rl = avl * 3.0 / 4.0 / np.pi
        rl = np.power(rl, 1.0/3.0)
        outdf.at[ii, "volume_TC_laves"] = avl
        outdf.at[ii, "radius_TC_laves"] = rl
        if vml == 0.0:
            print(f"element: {ele} in laves has no volume output.")
            outdf.at[ii, "bcc2laves_Gmix_TC"] = IMP_VALUE4DFT
            outdf.at[ii, "bcc2laves_Hmix_TC"] = IMP_VALUE4DFT
            outdf.at[ii, "bcc2laves_Smix_TC"] = IMP_VALUE4DFT
        else:
            outdf.at[ii, "bcc2laves_Gmix_TC"] = (gml - gmb) * Constants["J2eVA"]
            outdf.at[ii, "bcc2laves_Hmix_TC"] = (hml - hmb) * Constants["J2eVA"]
            outdf.at[ii, "bcc2laves_Smix_TC"] = (sml - smb) * Constants["J2eVA"]

    outdf.to_csv(fTC_elements, index=False, float_format=Constants["float_format"])


if os.path.isfile("TC_elements_db.csv"):
    DF_BCC_TC = pd.read_csv(fTC_elements)
    DF_BCC_TC = DF_BCC_TC.set_index(INDEXKEY)


def get_TC_refs(compstr, temp,
                elements=TC_Elements_list, elements_crystal=TC_Elements_crystal):
    comp = Composition(compstr)
    gm = 0.0
    hm = 0.0
    sm = 0.0
    gm_str = 0.0
    hm_str = 0.0
    sm_str = 0.0
    b2h_dg = 0.0
    b2h_dh = 0.0
    for el in comp.elements:
        thisc = comp.get_atomic_fraction(el)
        sym = el.symbol
        isym = elements.index(sym)
        cry = elements_crystal[isym]
        thisind = sym + "_" + str(int(temp))
        if cry == "bcc":
            gm += DF_BCC_TC.at[thisind, "GM"] * thisc
            hm += DF_BCC_TC.at[thisind, "HM"] * thisc
            sm += DF_BCC_TC.at[thisind, "SM"] * thisc
            gm_str += DF_BCC_TC.at[thisind, "GM"] * thisc
            hm_str += DF_BCC_TC.at[thisind, "HM"] * thisc
            sm_str += DF_BCC_TC.at[thisind, "SM"] * thisc
        else:
            gm += DF_BCC_TC.at[thisind, "GM_hcp"] * thisc
            hm += DF_BCC_TC.at[thisind, "HM_hcp"] * thisc
            sm += DF_BCC_TC.at[thisind, "SM_hcp"] * thisc
            gm_str += DF_BCC_TC.at[thisind, "GM"] * thisc
            hm_str += DF_BCC_TC.at[thisind, "HM"] * thisc
            sm_str += DF_BCC_TC.at[thisind, "SM"] * thisc
        b2h_dg += DF_BCC_TC.at[thisind, "bcc2hcp_Gmix_TC_diff"] * thisc
        b2h_dh += DF_BCC_TC.at[thisind, "bcc2hcp_Hmix_TC_diff"] * thisc
    return [gm, hm, sm, gm_str, hm_str, sm_str, b2h_dg, b2h_dh]


def assign2outdf(i, outdf, gmb, hmb, smb, gmh, hmh, smh, gml, hml, sml, gmr, hmr, smr, gmbr, hmbr, smbr,
                 b2h_dg, b2h_dh):
    outdf.at[i, "Gmix_TC"] = IMP_VALUE
    outdf.at[i, "Hmix_TC"] = IMP_VALUE
    outdf.at[i, "Gmix_str_TC"] = IMP_VALUE
    outdf.at[i, "Hmix_str_TC"] = IMP_VALUE
    outdf.at[i, "bcc2hcp_Gmix_TC"] = IMP_VALUE
    outdf.at[i, "bcc2hcp_Hmix_TC"] = IMP_VALUE
    outdf.at[i, "bcc2laves_Gmix_TC"] = IMP_VALUE
    outdf.at[i, "bcc2laves_Hmix_TC"] = IMP_VALUE
    outdf.at[i, "bcc2b2_Gmix_TC"] = IMP_VALUE
    outdf.at[i, "bcc2b2_Hmix_TC"] = IMP_VALUE
    outdf.at[i, "bcc2hcp_Gmix_TC2CPA"] = IMP_VALUE
    outdf.at[i, "bcc2hcp_Hmix_TC2CPA"] = IMP_VALUE
    if abs(gmb) >= 1.0e-10:
        outdf.at[i, "Gmix_TC"] = (gmb - gmr) * Constants["J2eVA"]
        outdf.at[i, "Hmix_TC"] = (hmb - hmr) * Constants["J2eVA"]
        outdf.at[i, "Gmix_str_TC"] = (gmb - gmbr) * Constants["J2eVA"]
        outdf.at[i, "Hmix_str_TC"] = (hmb - hmbr) * Constants["J2eVA"]
        outdf.at[i, "bcc2b2_Gmix_TC"] = (gmb - gmbr) * Constants["J2eVA"]
        outdf.at[i, "bcc2b2_Hmix_TC"] = (hmb - hmbr) * Constants["J2eVA"]
        if abs(gmh) >= 1.0e-10:
            outdf.at[i, "bcc2hcp_Gmix_TC"] = (gmh - gmb) * Constants["J2eVA"]
            outdf.at[i, "bcc2hcp_Hmix_TC"] = (hmh - hmb) * Constants["J2eVA"]
            outdf.at[i, "bcc2hcp_Gmix_TC2CPA"] = (gmh - gmb) * Constants["J2eVA"] + b2h_dg
            outdf.at[i, "bcc2hcp_Hmix_TC2CPA"] = (hmh - hmb) * Constants["J2eVA"] + b2h_dh
        if abs(gmh) >= 1.0e-10:
            outdf.at[i, "bcc2laves_Gmix_TC"] = (gml - gmbr) * Constants["J2eVA"]
            outdf.at[i, "bcc2laves_Hmix_TC"] = (hml - hmbr) * Constants["J2eVA"]
    return outdf


def TC_raw2mix(fname, outf, elements=TC_Elements_list, elements_crystal=TC_Elements_crystal):
    if "raw2" in fname:
        raise ValueError("Invalid filename.")
    if "raw" not in fname:
        raise ValueError("Invalid filename.")
    df = pd.read_csv(fname)
    outcols = [INDEXKEY, "Composition", "ReducedFormula", "Crystal", "Temperature",
               "VM", "volume_TC", "radius_TC",
               "Gmix_TC", "Hmix_TC",
               "Gmix_str_TC", "Hmix_str_TC",
               "bcc2hcp_Gmix_TC", "bcc2hcp_Hmix_TC",
               "bcc2laves_Gmix_TC", "bcc2laves_Hmix_TC",
               "bcc2b2_Gmix_TC", "bcc2b2_Hmix_TC",
               "bcc2hcp_Gmix_TC2CPA", "bcc2hcp_Hmix_TC2CPA"]
    d = np.zeros([len(df), len(outcols)])
    outdf = pd.DataFrame(d, columns=outcols)
    outdf[INDEXKEY] = outdf[INDEXKEY].astype(str)
    outdf["Composition"] = outdf["Composition"].astype(str)
    outdf["ReducedFormula"] = outdf["ReducedFormula"].astype(str)
    outdf["Crystal"] = outdf["Crystal"].astype(str)
    cs = df["Composition"].to_numpy()
    rcs = df["ReducedFormula"].to_numpy()
    ts = df["Temperature"].to_numpy()
    for i in range(len(cs)):
        c = cs[i]
        rc = rcs[i]
        if "67" in c:
            c = c.replace("33", "")
            c = c.replace("67", "2")
            rc = rc.replace("33", "")
            rc = rc.replace("67", "2")
        temp = ts[i]

        [gm, hm, sm, gm_str, hm_str, sm_str, b2h_dg, b2h_dh] = (
            get_TC_refs(c, temp, elements=elements, elements_crystal=elements_crystal))
        try:
            vmb = df.at[i, "BCC_B2_VM"]
        except:
            vmb = 0.0
        gmb = df.at[i, "BCC_B2_GM"]
        hmb = df.at[i, "BCC_B2_HM"]
        smb = df.at[i, "BCC_B2_SM"]
        gmh = df.at[i, "HCP_A3_GM"]
        hmh = df.at[i, "HCP_A3_HM"]
        smh = df.at[i, "HCP_A3_SM"]
        gml = df.at[i, "C15_LAVES_GM"]
        hml = df.at[i, "C15_LAVES_HM"]
        sml = df.at[i, "C15_LAVES_SM"]
        outdf.at[i, INDEXKEY] = rc + "_" + str(int(temp))
        outdf.at[i, "Composition"] = c
        outdf.at[i, "ReducedFormula"] = rc
        outdf.at[i, "Crystal"] = "bcc"
        outdf.at[i, "Temperature"] = temp
        if vmb != 0.0:
            outdf.at[i, "VM"] = vmb
            outdf.at[i, "volume_TC"] = vmb * Constants["TC2CPA_volume"]
            rb = vmb * Constants["TC2CPA_volume"] * 3.0 / 4.0 / np.pi
            outdf.at[i, "radius_TC"] = np.power(rb, 1.0/3.0)
        else:
            outdf.at[i, "VM"] = 0.0
            outdf.at[i, "volume_TC"] = IMP_VALUE
            outdf.at[i, "radius_TC"] = IMP_VALUE
            print(f"rc: {rc} in {fname} has no volume output.")
        outdf = assign2outdf(i, outdf, gmb, hmb, smb, gmh, hmh, smh, gml, hml, sml, gm, hm, sm, gm_str, hm_str, sm_str,
                             b2h_dg, b2h_dh)
    outdf.to_csv(outf, index=False, float_format=Constants["float_format"])


def compress2Element_basic(compstrs):
    inds = np.array([], dtype=int)
    for i in range(len(compstrs)):
        compstr = compstrs[i]
        comp = Composition(compstr)
        isvalid = True
        for el in comp.elements:
            if el.symbol not in Element_Basic:
                isvalid = False
                break
        if isvalid:
            inds = np.append(inds, [i])
    return inds


def compress2temperature(ts, temp):
    inds = np.array([], dtype=int)
    for i in range(len(ts)):
        t = ts[i]
        if abs(t - temp) < 50.0:
            isvalid = True
        else:
            isvalid = False
        if isvalid:
            inds = np.append(inds, [i])
    return inds


def compress_df2Element_list(df, ind4comp="Composition"):
    if ind4comp == "index":
        compstrs = df.index.to_numpy()
    else:
        compstrs = df[ind4comp].to_numpy()

    inds = df.index.to_numpy()
    vinds = np.array([])
    for i in range(len(compstrs)):
        compstr = compstrs[i]
        comp = Composition(compstr)
        isvalid = True
        for el in comp.elements:
            if el.symbol in TC_Elements_additional:
                isvalid = False
                break
        if isvalid:
            vinds = np.append(vinds, [inds[i]])
    df = df.loc[vinds]
    return df


def plot_TC_DFT_element(ykey, temp, ykeydft, label=True, savefig=False):
    df = pd.read_csv("TC_elements_db.csv")
    ys = df["Temperature"].to_numpy().astype(int)
    temp = int(temp)
    inds = np.arange(len(ys), dtype=int)
    inds = np.compress(ys == temp, inds)
    df = df.loc[inds]
    ytc = df[ykey].to_numpy()
    indx = df["Composition"].to_numpy()

    dfdft = pd.read_csv("External.csv")
    cs = dfdft["Composition"].to_numpy()
    rcs = []
    for ics in range(len(cs)):
        c = cs[ics]
        comp = Composition(c)
        rcs.append(comp.reduced_formula)
    dfdft["tmpindx"] = rcs
    dfdft = dfdft.set_index("tmpindx")

    dfdft = dfdft.loc[indx]
    ydft = dfdft[ykeydft].to_numpy()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fontsize = 14
    nsubplot = 1
    fig = plt.figure()
    ax = plt.subplot(nsubplot, 1, 1)
    ax.scatter(ydft, ytc, s=30, marker="*", c="r")
    if label:
        for i in range(len(ydft)):
            ax.text(ydft[i], ytc[i], indx[i], fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    if savefig:
        outfile = ykey + "_" + ykeydft + "_T" + str(temp) + ".png"
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()


def plot_TC_3d_bar(fname, xkey, zkey, savefig=False):
    df = pd.read_csv(fname)
    xs = df[xkey].to_numpy()
    ys = df["Temperature"].to_numpy().astype(int)
    zs = df[zkey].to_numpy()
    fig = get_3d_bar(xs, ys, zs, yticks=TEMP_LEVELS, isstr=True)
    if savefig:
        outfile = "TC_3d_bar.png"
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()


def plot_binary_temps(fname, key, elements, fref, keyref, temps=TEMP_LEVELS, savefig=False, conc4last=None):
    def get_df(df, cs, elements, conc4last=None, remove_equivalence=False):
        nele = len(elements)
        indexes = df.index.to_numpy()
        inds = np.array([], dtype=int)
        concs = [[] for _ in range(nele)]
        csref = []
        for icomp in range(len(cs)):
            c = cs[icomp]
            comp = Composition(c)
            c = comp.reduced_formula
            comp = Composition(c)
            isvalid = True
            if len(comp.elements) == len(elements):
                for el in comp.elements:
                    if el.symbol not in elements:
                        isvalid = False
                        break
            else:
                isvalid = False
            if isinstance(conc4last, float) and isvalid:
                thisx = comp.get_atomic_fraction(comp.elements[-1])
                if abs(thisx - conc4last) > 0.05:
                    isvalid = False
            if isvalid and remove_equivalence:
                if c in csref:
                    isvalid = False
            if isvalid:
                for i in range(len(comp.elements)):
                    concs[i].append(comp.get_atomic_fraction(comp.elements[i]))
                inds = np.append(inds, [icomp])
                csref.append(c)
        indexes = indexes[inds]
        df = df.loc[indexes]
        for i in range(nele):
            thiskey = "Conc" + str(i)
            df[thiskey] = concs[i][0:len(df)]
        return df

    nele = len(elements)
    xss = []
    yss = []
    dfref = pd.read_csv(fref)
    if "ReducedFormula" not in dfref.columns:
        cs = dfref["Compstr"].to_numpy()
    else:
        cs = dfref["ReducedFormula"].to_numpy()

    dfref = get_df(dfref, cs, elements, conc4last=conc4last, remove_equivalence=True)
    xs = dfref["Conc" + str(nele - 1)].to_numpy()
    ys = dfref[keyref].to_numpy()
    xss.append(xs)
    yss.append(ys)

    dftc = pd.read_csv(fname)
    dftc = compress_df2Element_list(dftc)
    cs = dftc["ReducedFormula"].to_numpy()
    dftc = get_df(dftc, cs, elements, conc4last=conc4last, remove_equivalence=False)
    for itemp in range(len(temps)):
        temp = temps[itemp]
        thisdf = dftc.copy(deep=True)
        ts = thisdf["Temperature"].to_numpy()
        inds = compress2temperature(ts, temp)
        thisdf = thisdf.iloc[inds]
        xs = thisdf["Conc" + str(nele - 1)].to_numpy()
        ys = thisdf[key].to_numpy()
        print("===")
    xss = np.array(xss)
    yss = np.array(yss)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({'figure.autolayout': True})
    fontsize = 14
    markers = ["s", "o", "d", "x", "*", "<", ">"]
    colors = ["k", "b", "c", "g", "m", "y", "r"]
    nsubplot = 1
    fig = plt.figure()
    axs = [None] * nsubplot
    iplot = 0
    nline = len(xss)
    lines = [None] * nline
    labels = ["CPA"]
    for temp in temps:
        labels.append(str(int(temp)))
    ndata = xss.shape[1]
    for iline in range(nline):
        x = xss[iline, :]
        y = yss[iline, :]
        thismarker = markers[iline]
        thiscolor = colors[iline]
        axs[iplot] = plt.subplot(nsubplot, 1, iplot + 1)
        lines[iline] = axs[iplot].scatter(x, y, s=30, marker=thismarker, c=thiscolor)
        plt.setp(axs[iplot].get_yticklabels(), fontsize=fontsize)
        plt.setp(axs[iplot].get_xticklabels(), fontsize=fontsize)
    if ndata > 1:
        ncol = 4
    else:
        ncol = 1
    plt.legend(lines, labels, scatterpoints=1, ncol=ncol, fontsize=fontsize - 4)
    if savefig:
        outfile = ""
        for sym in elements:
            outfile += sym
        outfile += "_" + keyref + "_" + key + ".png"
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def to_TC_1ele_IMs(df, keys, keysTC, temp=300, thres=1.0):
    compstrs = df["Composition"].to_numpy()
    for ikey in range(len(keys)):
        key = keys[ikey]
        for i in range(len(compstrs)):
            c = compstrs[i]
            comp = Composition(c)
            rc = comp.reduced_formula
            comp = Composition(rc)
            if len(comp.elements) == 1:
                thisind = comp.elements[0].symbol
                thisind = thisind + "_" + str(temp)
                try:
                    tcval = DF_BCC_TC.at[thisind, keysTC[ikey]]
                    if tcval < thres:
                        df.at[i, key] = tcval
                except:
                    pass
    return df

def DFT_TC_regression(fdft, ftc, keypairs, temp,
                      ind4dft="ReducedFormula", ind4tc="ReducedFormula", thres4tc=1.5,
                      compute_str=False, Label_Comp=False, compress2basic=False,
                      setdft2imp=False, reverse_fill=False, thres4reverse=5.0, savefig=False):
    fhead = fdft.replace(".csv", "")
    fhead = fhead.replace("DFT_", "")
    fhead = fhead.replace(".org", "")
    if "B2Latt" in fhead:
        isB2 = True
    else:
        isB2 = False
    if "bccC2" in fhead:
        isbinary = True
    else:
        isbinary = False
    if isinstance(setdft2imp, bool):
        setdft2imp = [setdft2imp] * len(keypairs)
    if isinstance(thres4reverse, float):
        thres4reverse = [thres4reverse] * len(keypairs)
    if "bccC2_" in fhead:
        ncompon = 2
        ninvalid = 3
    else:
        ncompon = 3
        ninvalid = 10
    keypairs = np.array(keypairs)
    npairs = len(keypairs)
    df_dft = pd.read_csv(fdft)
    df_dft = df_dft.set_index(ind4dft)

    df_tc = pd.read_csv(ftc)
    ts = df_tc["Temperature"].to_numpy()
    inds = compress2temperature(ts, temp)
    df_tc = df_tc.iloc[inds]
    df_tc = df_tc.set_index(ind4tc)
    if ind4tc == "Composition":
        ind4comp = "index"
    else:
        ind4comp = "Composition"
    df_tc = compress_df2Element_list(df_tc, ind4comp="index")

    cols = ["fname", "key", "p0_fit", "p1_fit"]
    paramdf = pd.DataFrame(columns=cols)
    keysinTC = []
    for ipair in range(npairs):
        thissetdft2imp = setdft2imp[ipair]
        keydft = keypairs[ipair, 0]
        keytc = keypairs[ipair, 1]
        keysinTC.append(keytc)
        thiskey = keydft + "_" + keytc

        df2 = df_tc.copy(deep=True)
        ys2 = df2[keytc].to_numpy()
        ts2 = df2["Temperature"].to_numpy()
        x2 = df2.index.to_numpy()
        x2 = np.compress(ys2 != IMP_VALUE, x2)
        ts2 = np.compress(ys2 != IMP_VALUE, ts2)
        ys2 = np.compress(ys2 != IMP_VALUE, ys2)

        if isB2:
            for i in range(len(x2)):
                thisx = x2[i]
                comp = Composition(thisx)
                thisx = comp.elements[1].symbol + comp.elements[0].symbol
                x2 = np.append(x2, [thisx])
                ys2 = np.append(ys2, [ys2[i]])
                ts2 = np.append(ts2, [ts2[0]])

        ## all dfdft that in dftc
        df_dft_tc = df_dft.copy(deep=True)
        df_dft_tc = df_dft_tc.loc[x2]
        df_dft_tc[keytc] = ys2
        x2tc = np.copy(x2)

        if isinstance(thres4tc, float):
            x2 = np.compress(ys2 < thres4tc, x2)
            ts2 = np.compress(ys2 < thres4tc, ts2)
            ys2 = np.compress(ys2 < thres4tc, ys2)

        if compress2basic:
            inds = compress2Element_basic(x2)
            x2 = x2[inds]
            ts2 = ts2[inds]
            ys2 = ys2[inds]

        df1_2 = df_dft.copy(deep=True)
        df1_2 = df1_2.loc[x2]
        #linear regression fitting, ys1_2, ys2_2
        x1_2 = df1_2.index.to_numpy()
        ys1_2 = df1_2[keydft].to_numpy()
        n = len(x1_2)
        print(f"length of data for fitting: {n}  ")
        if Label_Comp:
            Comp_Labels = np.copy(x1_2)
        else:
            Comp_Labels = None
        fig, df_tmp = plot_linear_xys(ys1_2, ys2, curvefit=True, Label_R2=True,
                                      Comp_Labels=Comp_Labels, Label_Comp=Label_Comp)

        if savefig:
            ffig = fhead + "_" + thiskey + "_T" + str(int(temp)) + ".png"
            plt.savefig(ffig, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        p0 = df_tmp.at[0, "p0_fit"]
        p1 = df_tmp.at[0, "p1_fit"]

        df_dft_not = df_dft.copy(deep=True)
        inds = df_dft_not.index.to_numpy()
        xnot = np.array([], dtype=int)
        for thisind in inds:
            if thisind not in x2tc:
                xnot = np.append(xnot, [thisind])
        df_dft_not = df_dft_not.loc[xnot]
        xnew = df_dft_not[keydft].to_numpy()
        ynew = p0 + xnew * p1
        if thissetdft2imp:
            cspaces=[]
            icount=[]
            isid = []
            for i in range(len(xnot)):
                cs = xnot[i]
                comp = Composition(cs)
                comp = Composition(comp.reduced_formula)
                thiscs = ""
                for el in comp.elements:
                    thiscs += el.symbol
                if thiscs in cspaces:
                    thisid = cspaces.index(thiscs)
                    icount[thisid] += 1
                    isid[thisid].append(i)
                else:
                    cspaces.append(thiscs)
                    icount.append(1)
                    thisid = [i]
                    isid.append(thisid)

            thisinds = np.array([], dtype=int)
            for i in range(len(cspaces)):
                if icount[i] >= ninvalid:
                    thisinds = np.append(thisinds, isid[i])
                    print(f"cs:{cspaces[i]} icount:{icount[i]} isid:{isid[i]}")

            ytmp = [IMP_VALUE4DFT]*len(df_dft_not)
            #for i in thisinds:
            #    ytmp[i] = ynew[i]
            df_dft_not[keytc] = ytmp
        else:
            df_dft_not[keytc] = ynew
        print(df_dft_not[keytc].tail())
        print(f"length of data obtained from curve fit: {len(df_dft_not)}.")

        ## compute back
        if reverse_fill:
            ynew = df_dft_tc[keytc]
            xnew = (ynew - p0) / p1
            xdft = df_dft_tc[keydft].to_numpy()
            for i in range(len(xnew)):
                if abs(xdft[i]) > thres4reverse[ipair]:
                    xdft[i] = xnew[i]
            df_dft_tc[keydft] = xdft
        df_dft = pd.concat([df_dft_tc, df_dft_not])

        if isB2:
            inds = df_dft.index.to_numpy()
            for ind in inds:
                comp = Composition(ind)
                if len(comp.elements) == 1:
                    df_dft.loc[ind, keytc] = 0.0
        thisdict = {}
        thisdict["fname"] = fhead
        thisdict["key"] = thiskey
        thisdict["p0_fit"] = p0
        thisdict["p1_fit"] = p1
        paramdf.loc[len(paramdf)] = thisdict

        displaycols = [keydft, keytc]
        print(df_dft[displaycols].head())
        print(df_dft[displaycols].tail())
        print(f"finished fname: {fhead} key: {thiskey}")
        print("==========")

    if compute_str:
        ys_str = df_dft["Eform"] - df_dft["Eform_str"]
        if "Hmix_TC" in keysinTC:
            hs = df_dft["Hmix_TC"]
            hs_str = hs - ys_str
            hs_str = np.select([hs_str<=5.0, hs_str>5.0], [hs_str, IMP_VALUE4DFT])
            df_dft["Hmix_str_TC"] = hs_str
        if "Gmix_TC" in keysinTC:
            gs = df_dft["Gmix_TC"]
            gs_str = gs - ys_str
            gs_str = np.select([gs_str <= 5.0, gs_str > 5.0], [gs_str, IMP_VALUE4DFT])
            df_dft["Gmix_str_TC"] = gs_str
    print(f"finished fname: {fhead}")
    print("========================")
    if "LAVES" in fhead:
        laveskeys = []
        for key in keysinTC:
            if "bcc2laves" in key:
                laveskeys.append(key)
        df_dft = to_TC_1ele_IMs(df_dft, laveskeys, laveskeys, temp=temp, thres=1.0)
    return df_dft, paramdf

def DFT_TC_ternary_dataexpansion(fdft, ftc, keypairs, temp,
                      ind4dft="ReducedFormula", ind4tc="ReducedFormula", compute_str=True):
    fhead = fdft.replace(".csv", "")
    fhead = fhead.replace("DFT_", "")
    fhead = fhead.replace(".org", "")

    keypairs = np.array(keypairs)
    npairs = len(keypairs)
    df_dft = pd.read_csv(fdft)
    df_dft = df_dft.set_index(ind4dft)

    df_tc = pd.read_csv(ftc)
    ts = df_tc["Temperature"].to_numpy()
    inds = compress2temperature(ts, temp)
    df_tc = df_tc.iloc[inds]
    df_tc = df_tc.set_index(ind4tc)

    keysinTC = []
    for ipair in range(npairs):
        keydft = keypairs[ipair, 0]
        keytc = keypairs[ipair, 1]
        keysinTC.append(keytc)
        thiskey = keydft + "_" + keytc

        df2 = df_tc.copy(deep=True)
        ys2 = df2[keytc].to_numpy()
        ts2 = df2["Temperature"].to_numpy()
        x2 = df2.index.to_numpy()
        x2 = np.compress(ys2 != IMP_VALUE, x2)
        ts2 = np.compress(ys2 != IMP_VALUE, ts2)
        ys2 = np.compress(ys2 != IMP_VALUE, ys2)

        ## all dfdft that in dftc
        df_dft_tc = df_dft.copy(deep=True)
        df_dft_tc = df_dft_tc.loc[x2]
        df_dft_tc[keytc] = ys2
        x2tc = np.copy(x2)

        df_dft_not = df_dft.copy(deep=True)
        inds = df_dft_not.index.to_numpy()
        xnot = np.array([], dtype=int)
        for thisind in inds:
            if thisind not in x2tc:
                xnot = np.append(xnot, [thisind])
        df_dft_not = df_dft_not.loc[xnot]
        compstrs = df_dft_not["Composition"].to_numpy()
        ynew = []
        for i in range(len(compstrs)):
            compstr = compstrs[i]
            vs = compstr2NDIP_keys(compstr,[keytc], style=2,
                                   Exception_1="ROM", Exception_2="ZERO")
            ynew.append(vs[0])
        df_dft_not[keytc] = ynew
        print(keytc)
        print(df_dft_not[keytc])
        df_dft = pd.concat([df_dft_tc, df_dft_not])

    if compute_str:
        ys_str = df_dft["Eform"] - df_dft["Eform_str"]
        if "Hmix_TC" in keysinTC:
            hs = df_dft["Hmix_TC"]
            hs_str = hs - ys_str
            hs_str = np.select([hs_str<=5.0, hs_str>5.0], [hs_str, IMP_VALUE4DFT])
            df_dft["Hmix_str_TC"] = hs_str
        if "Gmix_TC" in keysinTC:
            gs = df_dft["Gmix_TC"]
            gs_str = gs - ys_str
            gs_str = np.select([gs_str <= 5.0, gs_str > 5.0], [gs_str, IMP_VALUE4DFT])
            df_dft["Gmix_str_TC"] = gs_str
    print(f"finished fname: {fhead}")
    print("========================")
    return df_dft


def merge_tc2dft_elements(fdft):
    Exp_list = ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Al", "Co", "Fe", "Ni", "Mn", "Sc", "Cu", "Zn", "Y", "Sn", "Pd", "Re"]
    Exp_latts = np.array([2.95, 3.03, 2.91, 3.23, 3.30, 3.15, 3.20, 3.30, 3.17, 4.0495, 2.5071, 2.8665, 3.524, 8.9125, 3.309, 3.6149, 2.665, 3.6474, 5.8318, 3.8907, 2.761], dtype=float)
    Exp_cs = np.array([4.68, 3.03, 2.91, 5.15, 3.30, 3.15, 5.05, 3.30, 3.17, 4.0495, 4.0695, 2.8665, 3.524, 8.9125, 5.2733, 3.6149, 4.9468, 5.7306, 3.1819, 3.8907, 4.451], dtype=float)
    Exp_crys = ["hcp", "bcc", "bcc", "hcp", "bcc", "bcc", "hcp", "bcc", "bcc", "fcc", "hcp", "bcc", "fcc", "cubic29", "hcp", "fcc", "hcp", "hcp", "ct", "fcc", "hcp"]
    addkeys = ["radius_TC", "volume_TC", "bcc2hcp_Gmix_TC", "bcc2hcp_Gmix_TC_diff", "volume_EXP"]
    dftc = pd.read_csv(fTC_elements)
    temp = 300.0
    ts = dftc["Temperature"].to_numpy()
    inds = compress2temperature(ts, temp)
    dftc = dftc.loc[inds]
    tceles = dftc["Composition"].tolist()

    dfdft = pd.read_csv(fdft)
    dfteles = dfdft["Composition"].tolist()
    for i in range(len(dfteles)):
        dfteles[i] = dfteles[i].replace("100","")
    dftcrys = dfdft["Crystal"].tolist()
    usage = "ground"
    if "hcpC1" in fdft:
        usage = "hcp"
    elif "bccC1" in fdft:
        usage = "bcc"

    for ikey in range(len(addkeys)):
        thiskey = addkeys[ikey]
        if ikey == 0:
            dftrs = dfdft["radius"].tolist()
        elif ikey == 1:
            dftrs = dfdft["volume"].tolist()
        elif ikey == 2:
            dftrs = dfdft["bcc2hcp"].tolist()
        elif ikey == 3:
            dftrs = dfdft["bcc2hcp"].tolist()
        else:
            dftrs = dfdft["volume"].tolist()


        ys = dftrs[0:len(dftrs)]
        compute_minmax = False
        basinds = []
        for i in range(len(dfdft)):
            ele = dfteles[i]
            if ele == "min" or ele == "max":
                compute_minmax = True
            if ele in Element_Basic:
                basinds.append(i)
            if ele in tceles:
                iele = tceles.index(ele)
                cry = dftcrys[i]
                if ikey <= 3:
                    if usage == "hcp":
                        if ikey == 0:
                            ys[i] = dftc.at[iele, "radius_TC_hcp"]
                        elif ikey == 1:
                            ys[i] = dftc.at[iele, "volume_TC_hcp"]
                        elif ikey == 2:
                            ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC"]
                        elif ikey == 3:
                            ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC_diff"]
                    elif usage == "bcc":
                        if ikey == 0:
                            ys[i] = dftc.at[iele, "radius_TC"]
                        elif ikey == 1:
                            ys[i] = dftc.at[iele, "volume_TC"]
                        elif ikey == 2:
                            ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC"]
                        elif ikey == 3:
                            ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC_diff"]
                    else:
                        if cry.upper() == "HCP":
                            if ikey == 0:
                                ys[i] = dftc.at[iele, "radius_TC_hcp"]
                            elif ikey == 1:
                                ys[i] = dftc.at[iele, "volume_TC_hcp"]
                            elif ikey == 2:
                                ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC"]
                            elif ikey == 3:
                                ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC_diff"]
                        elif cry.upper() == "BCC":
                            if ikey == 0:
                                ys[i] = dftc.at[iele, "radius_TC"]
                            elif ikey == 1:
                                ys[i] = dftc.at[iele, "volume_TC"]
                            elif ikey == 2:
                                ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC"]
                            elif ikey == 3:
                                ys[i] = dftc.at[iele, "bcc2hcp_Gmix_TC_diff"]
                elif ikey == 4:
                    expiele = Exp_list.index(ele)
                    expcry = Exp_crys[expiele]
                    a = Exp_latts[expiele]
                    c = Exp_cs[expiele]
                    if expcry == "bcc":
                        ys[i] = a * a * a / 2.0
                    elif expcry == "hcp":
                        ys[i] = a * a * c * np.sqrt(3) / 4.0
                    elif expcry == "fcc":
                        ys[i] = a * a * a / 4.0
                    elif expcry == "cubic29":
                        ys[i] = a * a * a / 29.0
                    elif expcry == "ct":
                        ys[i] = dftc.at[iele, "volume_TC"]
                    else:
                        ys[i] = dftc.at[iele, "volume_TC"]

                else:
                    pass

            else:
                pass

        if compute_minmax:
            basinds = np.array(basinds).astype(int)
            ys = np.array(ys)
            ysbasic = ys[basinds]
            ys[-2] = np.min(ysbasic)
            ys[-1] = np.max(ysbasic)
        dfdft[thiskey] = ys
    dfdft.to_csv(fdft, index=False)







