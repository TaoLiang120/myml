import copy
from itertools import combinations

import numpy as np
from pymatgen.core.composition import Composition

from myml.myglobal import Constants, compstr2sconf, compute_sequential_concs, Element_sublatt
from myml.myglobal import Element_external, key2elekey
from myml.myelements.myelements import DF_Laves, DF_B2Latt, DF_external
from myml.expansion.expansion import compstr2NDIP_keys, compstr2all_NDIP_keys

kb = Constants["kb"]
float_format = Constants["float_format"]
nele_default = len(Element_sublatt)


def compstr2ROM(compstr, key):
    elekey = key2elekey(key)
    comp = Composition(compstr)
    nele = len(comp.elements)
    thisv = 0.0
    for i in range(nele):
        iele = comp.elements[i]
        ifrac = comp.get_atomic_fraction(iele)
        iind = Element_external.index(iele.symbol)
        ival = DF_external.iloc[iind][elekey]
        thisv += ival * ifrac
    return thisv


def scaling_factor(n, from_ncompon=3):
    return (1.0 + 2.0 * (n - 1) / float(n) - 2.0 * (from_ncompon - 1) / float(from_ncompon))


class mySublatt:
    def __init__(self, compstr, Eform, temp=300.0, style=1, isScaled=False, Scale_Eform=False):
        self.compstr = compstr
        self.Eform = Eform
        self.temp = temp
        self.style = style
        self.group_symbols = Element_sublatt[0:len(Element_sublatt)]
        self.comp = Composition(compstr)

        if self.style == 1:
            self.Sconf = compstr2sconf(self.compstr)
        else:
            self.Sconf = 0.0

        self.elements = []
        self.concs = []
        self.isSublatt = True
        for i, ele in enumerate(self.comp):
            sym = ele.symbol
            self.elements.append(sym)
            self.concs.append(self.comp.get_atomic_fraction(ele))
            if sym not in Element_sublatt:
                self.isSublatt = False
        self.concs = np.array(self.concs)
        self.concs = np.around(self.concs, decimals=4)
        self.nele = len(self.elements)

        self.Scale_Eform = Scale_Eform
        self.scaling = 1.0
        if self.Scale_Eform:
            if not isScaled:
                if self.style == 1:
                    self.scaling = scaling_factor(self.nele, from_ncompon=2)
                else:
                    self.scaling = scaling_factor(self.nele, from_ncompon=3)
        self.gmix = self.Eform * self.scaling - self.Sconf * self.temp

        #self.DATA_PATH = "ORG_DATA"
        self.steps = np.array([0.01, 0.03, 0.05, 0.07, 0.09])
        self.ediff = 1000.0
        self.substrs = None
        self.ediff2 = 1000.0
        self.substrs_2 = None

    def compute_substr_energy(self, compstr, key, all_NDIP_dict=None):
        if len(compstr) == 0:
            return 0.0
        else:
            comp = Composition(compstr)
            ncompon = len(comp.elements)
            if ncompon == 1:
                if key == "Eform_BOKAS":
                    thiskey = "bcc2hcp"
                else:
                    thiskey = "bcc2hcp_Gmix_TC"
                sym = comp.elements[0].symbol
                isym = Element_external.index(sym)
                v = 0.0 - DF_external.loc[isym, thiskey]
                ## if v < 0.0, bcc is more stable. Need to set to zero since the reference energy of bcc SS is zero.
                ## if v > 0.0, hcp is more stable. Need to add this value for bcc SS.
                if v < 0.0:
                    v = 0.0
                return v
            else:
                if all_NDIP_dict is None:
                    vlist = compstr2NDIP_keys(compstr, [key], style=1,
                                              Scale_Bokas=True, Scale_Eform=self.Scale_Eform,
                                              Exception_1="ROM", Exception_2="ZERO")
                else:
                    vlist = compstr2all_NDIP_keys(compstr, [key], all_NDIP_dict,
                                                  style=1, Scale_Bokas=True, Scale_Eform=self.Scale_Eform,
                                                  Exception_1="ROM", Exception_2="ZERO")

        return vlist[0]

    @staticmethod
    def get_subsconf(substrs, BAratio, Slice):
        subsconf = 0.0
        for i in range(len(substrs)):
            substr = substrs[i]
            if i == 0:
                subsconf += compstr2sconf(substr) / Slice
            else:
                subsconf += compstr2sconf(substr) * BAratio / Slice
        return subsconf

    @staticmethod
    def get_eform_substr(substrs, df_form, key_app=""):
        A = substrs[0]
        B = substrs[1]
        cA = Composition(A)
        cB = Composition(B)
        nA = len(cA.elements)
        nB = len(cB.elements)
        eform = 0.0
        thisctot = 0.0
        for i in range(nA):
            iele = cA.elements[i]
            isym = iele.symbol
            iconc = cA.get_atomic_fraction(iele)
            for j in range(nB):
                jele = cB.elements[j]
                jsym = jele.symbol
                jconc = cB.get_atomic_fraction(jele)
                thisc = iconc * jconc
                thiskey = isym + jsym + key_app
                try:
                    e = df_form.loc[thiskey, "Eform"]
                except:
                    e = 0.0
                eform += thisc * e
                thisctot += thisc
        return eform

    @staticmethod
    def get_pretty_substr(concs, syms, multiplier=100.0):
        compstr = ""
        for i in range(len(concs)):
            c = concs[i] * multiplier
            c = int(c)
            if c > 0:
                compstr += syms[i] + str(c)
        return compstr

    def get_gmix_substr(self, substrs, concs, df_form=None, all_NDIP_dict=None):
        """
        Args:
            substrs: asubstr, bsubstr
            concs: total concentration of asubstr and bsubstr
            df_form: look up table for intermetallics
            need to calculation configurational entropy. 0 means the input is Gibbs energy of mixing.
        Returns: gmix of

        """
        Astr = substrs[0]
        Bstr = substrs[1]
        cA = concs[0]
        cB = concs[1]
        key = "Eform_BOKAS"
        if self.style != 1:
            key = "Gmix_TC"

        if self.style == 1:
            eform = (df_form.loc[Astr, "Eform"] * cA +
                     self.compute_substr_energy(Bstr, key, all_NDIP_dict=all_NDIP_dict) * cB)
            sconf = compstr2sconf(Astr) * cA + compstr2sconf(Bstr) * cB
            gmix = eform - sconf * self.temp
        else:
            gmix = (self.compute_substr_energy(Astr, key, all_NDIP_dict=all_NDIP_dict) * cA +
                    self.compute_substr_energy(Bstr, key, all_NDIP_dict=all_NDIP_dict) * cB)
            '''
            print(f"astr:{Astr} eA:{self.compute_substr_energy(Astr, key, all_NDIP_dict=all_NDIP_dict)}")
            print(f"Bstr:{Bstr} eA:{self.compute_substr_energy(Bstr, key, all_NDIP_dict=all_NDIP_dict)}")
            print("0000")
            '''
        return gmix

    def get_substr_combination4IM(self, BA_ratio=1):
        if self.nele < 3:
            return None
        ncoms = [2]

        list_substrs = []
        list_subconcs = []
        list_dist_subconcs = []
        for ncom in ncoms:
            inds = np.arange(self.nele, dtype=int)
            combs = combinations(inds, ncom)
            for comb in list(combs):
                a_ids = np.array([], dtype=int)
                a_syms = np.array([], dtype=str)
                a_concs = np.zeros(self.nele, dtype=float)
                b_concs = copy.deepcopy(self.concs)
                for i in range(ncom):
                    iele = comb[i]
                    a_ids = np.append(a_ids, [iele])
                    a_syms = np.append(a_syms, [self.elements[iele]])
                    a_concs[iele] += self.concs[iele]
                    b_concs[iele] -= self.concs[iele]
                #print(f"comb: {comb} a_ids: {a_ids} a_syms:{a_syms} a_concs:{a_concs} b_concs:{b_concs}")
                aaind = Element_sublatt.index(a_syms[0])
                bbind = Element_sublatt.index(a_syms[1])
                if aaind < bbind:
                        inds = np.array([1, 0], dtype=int)
                        a_ids = a_ids[inds]
                        a_syms = a_syms[inds]
                        tmp = a_concs[a_ids[0]]
                        a_concs[a_ids[0]] = a_concs[a_ids[1]]
                        a_concs[a_ids[1]] = tmp
                if a_concs[a_ids[0]] >= a_concs[a_ids[1]] / BA_ratio:
                        left = a_concs[a_ids[0]] - a_concs[a_ids[1]] / BA_ratio
                        a_concs[a_ids[0]] = a_concs[a_ids[1]] / BA_ratio
                        b_concs[a_ids[0]] += left
                else:
                        left = a_concs[a_ids[1]] - a_concs[a_ids[0]] * BA_ratio
                        a_concs[a_ids[1]] = a_concs[a_ids[0]] * BA_ratio
                        b_concs[a_ids[1]] += left
                if BA_ratio > 1:
                        astr = a_syms[0] + a_syms[1] + str(BA_ratio)
                else:
                        astr = a_syms[0] + a_syms[1]
                bstr = mySublatt.get_pretty_substr(b_concs, self.elements, multiplier=100.0)
                list_substrs.append([astr, bstr])
                list_subconcs.append([np.sum(a_concs), np.sum(b_concs)])
                list_dist_subconcs.append((a_concs, b_concs))
                #print(f"comb: {comb} a_ids: {a_ids} a_syms:{a_syms} a_concs:{a_concs} b_concs:{b_concs}")
                #print(f"substr:{[astr, bstr]} subconcs:{[np.sum(a_concs), np.sum(b_concs)]}")
                #print("---end of this comb---")
            #print("=== end of ncom ===")
        #print(list_substrs)
        #print(list_subconcs)
        #print("+++ end of all +++")
        return list_substrs, list_subconcs, list_dist_subconcs

    def get_substr_combination4SS(self):
        if self.nele < 3:
            return None

        ncoms = []
        ndiv = int(self.nele / 2)
        ncom_min = ndiv + self.nele - ndiv * 2
        for i in range(ndiv):
            ncom = self.nele - i - 1
            if ncom >= ncom_min:
                ncoms.append(ncom)

        list_substrs = []
        list_subconcs = []
        list_dist_subconcs = []
        for ncom in ncoms:
            inds = np.arange(self.nele, dtype=int)
            combs = combinations(inds, ncom)
            list_comb = list(combs)
            ncombs = len(list_comb)
            if self.nele % 2 == 0 and ncom == int(self.nele / 2):
                ncombs = int(ncombs / 2)
            for icomb in range(ncombs):
                comb = list_comb[icomb]
                comb_rest = np.delete(inds, comb)

                a_concs = np.zeros(self.nele, dtype=float)
                for i in range(ncom):
                    iele = comb[i]
                    a_concs[iele] += self.concs[iele]
                b_concs = np.zeros(self.nele, dtype=float)
                for i in range(len(comb_rest)):
                    iele = comb_rest[i]
                    b_concs[iele] += self.concs[iele]
                astr = mySublatt.get_pretty_substr(a_concs, self.elements, multiplier=100.0)
                bstr = mySublatt.get_pretty_substr(b_concs, self.elements, multiplier=100.0)
                list_substrs.append([astr, bstr])
                list_subconcs.append([np.sum(a_concs), np.sum(b_concs)])
                list_dist_subconcs.append((a_concs, b_concs))
                #print(f"comb: {comb} a_concs:{a_concs} b_concs:{b_concs}")
                #print(f"substr:{[astr, bstr]} subconcs:{[np.sum(a_concs), np.sum(b_concs)]}")
                #print("---end of this comb---")
            #print("=== end of ncom ===")
        #print(list_substrs)
        #print(list_subconcs)
        #print("+++ end of all +++")
        return list_substrs, list_subconcs, list_dist_subconcs


    def fine_tune_concentration(self, dist_concs):
        """
        Args:
            dist_concs: a tuple of A_concs and B_concs
        Returns:
        """

        a_concs = copy.deepcopy(dist_concs[0])
        b_concs = copy.deepcopy(dist_concs[1])
        #cA = np.sum(a_concs)
        #cB = np.sum(b_concs)
        #a_steps = np.around(cA * self.steps, decimals=2)
        #b_steps = np.around(cB * self.steps, decimals=2)
        a_steps = copy.deepcopy(self.steps)
        b_steps = copy.deepcopy(self.steps)

        list_substrs = []
        list_subconcs = []
        for i in range(len(a_concs)):
            c = a_concs[i]
            for istep in range(len(b_steps)):
                cstep = b_steps[istep]
                if c > cstep:
                    thisa = copy.deepcopy(a_concs)
                    thisb = copy.deepcopy(b_concs)
                    thisa[i] -= cstep
                    thisb[i] += cstep
                    astr = mySublatt.get_pretty_substr(thisa, self.elements, multiplier=100.0)
                    bstr = mySublatt.get_pretty_substr(thisb, self.elements, multiplier=100.0)
                    list_substrs.append([astr, bstr])
                    list_subconcs.append([np.sum(thisa), np.sum(thisb)])


        for i in range(len(b_concs)):
            c = b_concs[i]
            for istep in range(len(a_steps)):
                cstep = a_steps[istep]
                if c > cstep:
                    thisa = copy.deepcopy(a_concs)
                    thisb = copy.deepcopy(b_concs)
                    thisa[i] += cstep
                    thisb[i] -= cstep
                    astr = mySublatt.get_pretty_substr(thisa, self.elements, multiplier=100.0)
                    bstr = mySublatt.get_pretty_substr(thisb, self.elements, multiplier=100.0)
                    list_substrs.append([astr, bstr])
                    list_subconcs.append([np.sum(thisa), np.sum(thisb)])

        return list_substrs, list_subconcs

    def get_substrs1(self):
        concs = compute_sequential_concs(self.compstr, elements=self.group_symbols)
        Astr = ""
        Bstr = ""
        cumcenc = np.cumsum(concs)
        Aind = np.where(cumcenc > 2 / 3)
        Aind = Aind[0][0]
        for isym in range(len(concs)):
            sym = self.group_symbols[isym]
            conc = concs[isym]
            if isym > Aind:
                if conc >= 5e-3:
                    cA = int(round(100 * conc, 0))
                    Astr += sym + str(cA)
            elif isym < Aind:
                if conc >= 5e-3:
                    cB = int(round(100 * conc, 0))
                    Bstr += sym + str(cB)
            else:
                cthres = cumcenc[Aind]
                Ashare = (cthres - 2.0 / 3.0) / conc
                Bshare = 1.0 - Ashare
                cA = int(round(100 * conc * Ashare, 0))
                cB = int(round(100 * conc * Bshare, 0))
                if cA < 1: cA = 1
                if cB < 1: cB = 1
                Astr += sym + str(cA)
                Bstr += sym + str(cB)
        substrs = [Astr, Bstr]
        return substrs

    @staticmethod
    def get_substrs_ranks(elements, ipairs, concs):
        def check_radius(ind, cnows, isym, jsym, ABstr):
            cs = int(round(100 * cnows[ind], 0))
            rdiff = [0.0, 0.0]
            for itype in range(2):
                if itype == 0:
                    c1 = ABstr[0] + isym + str(cs)
                    c2 = ABstr[1] + jsym + str(cs)
                else:
                    c1 = ABstr[0] + jsym + str(cs)
                    c2 = ABstr[1] + isym + str(cs)
                r1 = compstr2ROM(c1, "radius_TC")
                r2 = compstr2ROM(c2, "radius_TC")
                rdiff[itype] = abs(r1 - r2)
            if rdiff[0] < rdiff[1]:
                Swap = False
            elif rdiff[0] > rdiff[1]:
                Swap = True
            else:
                Swap = False
            return Swap

        def update_ipairs(ipairs, dep_list, prefer):
            npair = len(ipairs)
            for i in range(npair):
                ipair = ipairs[i]
                if ipair is not None:
                    pair0 = ipair[0]
                    pair1 = ipair[1]
                    isValid = True
                    for idel in dep_list:
                        if idel == pair0 or idel == pair1:
                            isValid = False
                            break
                    if not isValid: ipairs[i] = None

            ipnext = npair
            if prefer is not None:
                for i in range(npair):
                    ipair = ipairs[i]
                    if ipair is not None:
                        if prefer in list(ipair):
                            ipnext = i
                            break
            else:
                for i in range(npair):
                    ipair = ipairs[i]
                    if ipair is not None:
                        ipnext = i
                        break
            return ipnext, ipairs

        ABstr = ["", ""]
        ipnext = 0
        npair = len(ipairs)
        cnows = np.array([0.0, 0.0])
        prefer = None
        prefer_on = None
        for ip in range(npair):
            if ipnext < npair:
                ipair = ipairs[ipnext]
                if ipair is None:
                    pass
                else:
                    i = ipair[0]
                    j = ipair[1]
                    isym = elements[i]
                    jsym = elements[j]
                    iconc = concs[i]
                    jconc = concs[j]
                    dep_list = []
                    if iconc > jconc:
                        cnows += jconc
                        ind = 1
                        if prefer == i and prefer_on == 1:
                            ind = 0
                        elif prefer == j and prefer_on == 0:
                            ind = 0
                        elif prefer_on is None:
                            Swap = check_radius(ind, cnows, isym, jsym, ABstr)
                            if Swap:
                                ind = 0

                        if ind == 1:
                            ind4prefer = 0
                        else:
                            ind4prefer = 1
                        cs = int(round(100 * cnows[ind], 0))
                        thisstr = jsym + str(cs)
                        ABstr[ind] += thisstr
                        cnows[ind] = 0.0
                        prefer = i
                        prefer_on = ind4prefer
                        concs[i] -= jconc
                        concs[i] = round(concs[i], 6)
                        concs[j] = 0.0
                        dep_list.append(j)
                    elif iconc < jconc:
                        cnows += iconc
                        ind = 0
                        if prefer == i and prefer_on == 1:
                            ind = 1
                        elif prefer == j and prefer_on == 0:
                            ind = 1
                        elif prefer_on is None:
                            Swap = check_radius(ind, cnows, isym, jsym, ABstr)
                            if Swap:
                                ind = 1

                        if ind == 1:
                            ind4prefer = 0
                        else:
                            ind4prefer = 1
                        cs = int(round(100 * cnows[ind], 0))
                        thisstr = isym + str(cs)
                        ABstr[ind] += thisstr
                        cnows[ind] = 0.0
                        prefer = j
                        prefer_on = ind4prefer
                        concs[i] = 0.0
                        concs[j] -= iconc
                        concs[j] = round(concs[j], 6)
                        dep_list.append(i)
                    else:
                        cnows += jconc
                        Swap = False
                        if prefer == i and prefer_on == 1:
                            Swap = True
                        elif prefer == j and prefer_on == 0:
                            Swap = True
                        elif prefer_on is None:
                            ind = 0
                            Swap = check_radius(ind, cnows, isym, jsym, ABstr)

                        cs0 = int(round(100 * cnows[0], 0))
                        cs1 = int(round(100 * cnows[1], 0))
                        if Swap:
                            thisstr0 = jsym + str(cs0)
                            thisstr1 = isym + str(cs1)
                        else:
                            thisstr0 = isym + str(cs0)
                            thisstr1 = jsym + str(cs1)
                        ABstr[0] += thisstr0
                        ABstr[1] += thisstr1
                        cnows[0] = 0.0
                        cnows[1] = 0.0
                        concs[i] = 0.0
                        concs[j] = 0.0
                        dep_list.append(i)
                        dep_list.append(j)
                        prefer = None
                        prefer_on = None
                    ipnext, ipairs = update_ipairs(ipairs, dep_list, prefer)
            else:
                pass

        ind = np.argmax(cnows)
        if cnows[ind] > 5e-3:
            ADD = True
        else:
            ADD = False

        for i in range(len(concs)):
            sym = elements[i]
            Modify = False
            if prefer == i and ADD: Modify = True
            if Modify:
                cs = int(round(50 * concs[i], 0))
                cs_mod = int(round(50 * concs[i] + 100 * cnows[ind], 0))
                if ind == 0:
                    ind0 = 1
                else:
                    ind0 = 0
                ABstr[ind0] += sym + str(cs)
                ABstr[ind] += sym + str(cs_mod)
            else:
                cs = int(round(50 * concs[i], 0))
                if cs > 0:
                    ABstr[0] += sym + str(cs)
                    ABstr[1] += sym + str(cs)

        for i in range(len(ABstr)):
            comp = Composition(ABstr[i])
            ABstr[i] = comp.reduced_formula
        return ABstr

    def get_ediff(self, list_gmixes, list_substrs):
        ediff = 1000.0
        substrs = []
        ind = -10000
        if len(list_substrs) > 0:
            list_gmixes = np.array(list_gmixes)
            ind = np.argmin(list_gmixes)
            substrs = list_substrs[ind]
            ediff = list_gmixes[ind] - self.gmix
        return ediff, substrs, ind


class B2Latt(mySublatt):
    def __init__(self, compstr, Eform, temp=300.0, style=1, isScaled=False, Scale_Eform=False):
        super().__init__(
            compstr,
            Eform,
            temp=temp,
            style=style,
            isScaled=isScaled,
            Scale_Eform=Scale_Eform,
        )

        self.b2_ranks = ['AlNi', 'AlCo', 'ReTa', 'ReTi', 'HfRe', 'HfNi', 'CoHf', 'ReV', 'FeTi', 'CoTi', 'NiZr', 'NbRe',
                         'FeHf', 'NiTi', 'ReZr', 'CoZr', 'AlZr', 'AlFe', 'ZnZr', 'MnTi', 'MnV', 'AlTi', 'AlMn', 'MnTa',
                         'AlHf', 'HfZn', 'NiZn', 'HfMn', 'TiZn', 'FeZr', 'MnNb', 'MoTa', 'AlCu', 'AlRe', 'CuHf', 'FeTa',
                         'CuZr', 'MnZr', 'CuZn', 'CoMn', 'FeV', 'MoV', 'MoNb', 'CoTa', 'NiTa', 'MnRe', 'CrRe', 'CoFe',
                         'CuTi', 'TaW', 'CrV', 'AlCr', 'MoTi', 'HfMo', 'NbNi', 'AlNb', 'VW', 'FeNb', 'ReW', 'MoRe',
                         'MnZn', 'CoZn', 'AlV', 'CrTa', 'NbW', 'CoNb', 'CoV', 'CrMn', 'NiV', 'MoW', 'HfTi', 'FeRe',
                         'HfZr', 'NbTa', 'TiZr', 'CuNi', 'MoZr', 'NbTi', 'TaV', 'CrW', 'FeMn', 'AlZn', 'CrTi', 'FeNi',
                         'CrNb', 'CrHf', 'HfNb', 'AlMo', 'AlTa', 'TiW', 'CrMo', 'NbV', 'MnW', 'CoNi', 'TaTi', 'TiV',
                         'NbZn', 'HfV', 'FeZn', 'MnMo', 'NbZr', 'HfW', 'HfTa', 'CuMn', 'NiRe', 'CrZr', 'TaZr', 'VZn',
                         'CuNb', 'FeW', 'FeMo', 'CoRe', 'CrFe', 'VZr', 'TaZn', 'CoCu', 'CuTa', 'WZr', 'CrNi', 'CoMo',
                         'MnNi', 'MoNi', 'CuFe', 'CoW', 'CrZn', 'AlW', 'NiW', 'CuV', 'CoCr', 'MoZn', 'ReZn', 'CuRe',
                         'CuMo', 'WZn', 'CrCu', 'CuW']

        self.BAratio = 1.0
        self.Slice = 2.0
        self.thres4conc = 0.0

    def get_estimation4CCSub(self):
        if self.isSublatt:
            list_substrs = []
            list_gmixes = []
            elements = []
            concs = []
            for i, ele in enumerate(self.comp):
                sym = ele.symbol
                elements.append(sym)
                concs.append(self.comp.get_atomic_fraction(ele))
            concs = np.array(concs)
            concs = np.around(concs, decimals=6)

            nele = len(elements)
            inds = np.arange(nele, dtype=int)
            combs = combinations(inds, 2)
            ipairs = []
            ipairranks = []
            for comb in list(combs):
                i = comb[0]
                j = comb[1]
                isym = elements[i]
                jsym = elements[j]
                ipair = (i, j)
                pair = [isym, jsym]
                pair.sort()
                pair = pair[0] + pair[1]
                irank = self.b2_ranks.index(pair)
                ipairs.append(ipair)
                ipairranks.append(irank)
            ipairs = np.array(ipairs)
            ipairranks = np.array(ipairranks)
            inds = np.argsort(ipairranks)
            ipairs = ipairs[inds]
            ipairs = list(ipairs)
            ipairs1 = copy.deepcopy(ipairs)
            concs1 = copy.deepcopy(concs)
            substrs = mySublatt.get_substrs_ranks(elements, ipairs1, concs1)
            subsconf = mySublatt.get_subsconf(substrs, self.BAratio, self.Slice)
            eform = mySublatt.get_eform_substr(substrs, DF_B2Latt, key_app="")
            list_substrs.append(substrs)
            list_gmixes.append(eform - subsconf * self.temp)
            self.ediff, substrs, ind = self.get_ediff(list_gmixes, list_substrs)
            self.substrs = tuple(substrs)
        return self.ediff, self.substrs

    def get_estimation(self, all_NDIP_dict=None):
        if self.isSublatt:
            list_substrs, list_concs, list_dist_subconcs = self.get_substr_combination4IM(BA_ratio=1)
            list_gmixes = []
            for i in range(len(list_concs)):
                substrs = list_substrs[i]
                concs = list_concs[i]
                gmix = self.get_gmix_substr(substrs, concs, df_form=DF_B2Latt, all_NDIP_dict=all_NDIP_dict)
                list_gmixes.append(gmix)
            self.ediff2, substrs_2, ind = self.get_ediff(list_gmixes, list_substrs)
            self.substrs_2 = tuple(substrs_2)
        return self.ediff2, self.substrs_2
    

class Laves(mySublatt):
    def __init__(self, compstr, Eform, temp=300.0, style=1, isScaled=False, Scale_Eform=False):
        super().__init__(
            compstr,
            Eform,
            temp=temp,
            style=style,
            isScaled=isScaled,
            Scale_Eform=Scale_Eform,
        )
        self.BAratio = 2.0
        self.Slice = 3.0
        self.thres4conc = 0.3
        self.laves_rank = ['ZrAl2', 'HfNi2', 'HfAl2', 'ZrNi2', 'HfCo2', 'TiNi2', 'HfMn2', 'ZrCo2', 'TiAl2', 'HfFe2',
                           'TiMn2', 'ZrZn2', 'TiCo2', 'AlNi2', 'ZrMn2', 'NbAl2', 'HfZn2', 'ZrFe2', 'TiFe2', 'TaMn2',
                           'TaNi2', 'TaCo2', 'TiZn2', 'NbNi2', 'NbMn2', 'HfW2', 'HfMo2', 'TaAl2', 'NbCo2', 'NbZn2',
                           'AlMn2', 'AlCo2', 'ZrW2', 'AlCu2', 'HfCr2', 'VMn2', 'ZrMo2', 'WMn2', 'TaFe2', 'NbFe2',
                           'MoMn2', 'TiCr2', 'AlFe2', 'TaZn2', 'TaCr2', 'MnNi2', 'ZrCu2', 'ZnNi2', 'HfCu2', 'TaV2',
                           'VNi2', 'ZrCr2', 'VCo2', 'WNi2', 'MoNi2', 'MoAl2', 'NbV2', 'NbCr2', 'WV2', 'VFe2',
                           'WCo2', 'VAl2', 'HfV2', 'TiMo2', 'NiZn2', 'ZnCu2', 'TiCu2', 'AlTi2', 'HfNb2', 'MoCo2',
                           'MoZn2', 'MnZn2', 'MoV2', 'ZrNb2', 'ZrV2', 'TiW2', 'CrMn2', 'CuZn2', 'VZn2', 'AlV2',
                           'HfTa2', 'AlZn2', 'MoFe2', 'HfTi2', 'ZrTa2', 'TaMo2', 'FeNi2', 'AlCr2',
                           'ZnCo2', 'ZrTi2', 'VCr2', 'WFe2', 'TiV2', 'TaW2', 'WCr2', 'NiAl2', 'NbMo2',
                           'CuNi2', 'MoCr2', 'NbW2', 'NbCu2', 'MnFe2', 'FeZn2', 'WAl2', 'WZn2', 'MnV2', 'MnCu2',
                           'TaNb2', 'NiCo2', 'NbTi2', 'TaTi2', 'ZnAl2', 'ZnTi2', 'ZnFe2', 'NiFe2', 'NbTa2', 'FeMn2',
                           'CoZn2', 'ZnMn2', 'CrFe2']


    def get_estimation4CCSub(self):
        if self.isSublatt:
            list_substrs = []
            list_gmixes = []
            substrs = self.get_substrs1()
            subsconf = mySublatt.get_subsconf(substrs, self.BAratio, self.Slice)
            eform = mySublatt.get_eform_substr(substrs, DF_Laves, key_app="2")
            list_substrs.append(substrs)
            list_gmixes.append(eform - subsconf * self.temp)
            self.ediff, substrs, ind = self.get_ediff(list_gmixes, list_substrs)
            self.substrs = tuple(substrs)
        return self.ediff, self.substrs



    def get_estimation(self, all_NDIP_dict=None):
        if self.isSublatt:
            list_substrs, list_concs, list_dist_subconcs = self.get_substr_combination4IM(BA_ratio=2)
            list_gmixes = []
            for i in range(len(list_concs)):
                substrs = list_substrs[i]
                concs = list_concs[i]
                gmix = self.get_gmix_substr(substrs, concs, df_form=DF_Laves, all_NDIP_dict=all_NDIP_dict)
                list_gmixes.append(gmix)
            self.ediff2, substrs_2, ind = self.get_ediff(list_gmixes, list_substrs)
            self.substrs_2 = tuple(substrs_2)
        return self.ediff2, self.substrs_2


class Sub_SS(mySublatt):
    def __init__(self, compstr, Eform, temp=300.0, style=0, isScaled=False, Scale_Eform=False, fine_tune=True):
        super().__init__(
            compstr,
            Eform,
            temp=temp,
            style=style,
            isScaled=isScaled,
            Scale_Eform=Scale_Eform,
        )
        self.fine_tune = fine_tune
        self.ediff_fine = 1000.0
        self.substrs_fine = None

    def get_estimation(self, all_NDIP_dict=None):
        list_substrs, list_concs, list_dist_subconcs = self.get_substr_combination4SS()
        list_gmixes = []
        for i in range(len(list_concs)):
            substrs = list_substrs[i]
            concs = list_concs[i]
            gmix = self.get_gmix_substr(substrs, concs, df_form=None, all_NDIP_dict=all_NDIP_dict)
            list_gmixes.append(gmix)
            #print(f"i: {i} substrs:{substrs} gmix:{gmix}")
        #print('-------')

        self.ediff2, substrs_2, ind = self.get_ediff(list_gmixes, list_substrs)
        self.substrs_2 = tuple(substrs_2)
        #print(f"bcc2subss: {self.ediff2} substrs:{substrs_2} ind:{ind}")
        #print("+++++++++++++++")

        #if self.fine_tune and self.gmix < 0.0 and self.ediff2 < 0:
        if self.fine_tune and self.ediff2 < 0:
            #print(f"Start of the fine_tune:")
            list_dist_subconcs = np.array(list_dist_subconcs)
            list_gmixes = np.array(list_gmixes)
            inds = np.arange(len(list_gmixes), dtype=int)
            inds = np.compress(list_gmixes < self.gmix, inds)
            list_gmixes = list_gmixes[inds]
            list_dist_subconcs = list_dist_subconcs[inds]

            list_fine_substrs = []
            list_fine_concs = []
            for isub in range(len(list_gmixes)):
                this_substrs, this_concs = self.fine_tune_concentration(list_dist_subconcs[isub])
                if len(this_substrs) > 0:
                    list_fine_substrs += this_substrs
                    list_fine_concs += this_concs

            list_fine_gmixes = []
            for i in range(len(list_fine_concs)):
                substrs = list_fine_substrs[i]
                concs = list_fine_concs[i]
                gmix = self.get_gmix_substr(substrs, concs, df_form=None, all_NDIP_dict=all_NDIP_dict)
                list_fine_gmixes.append(gmix)
                #print(f"i: {i} substrs:{substrs} gmix:{gmix}")
            #print('-------')

            self.ediff_fine, substrs_fine, ind = self.get_ediff(list_fine_gmixes, list_fine_substrs)
            self.substrs_fine = tuple(substrs_fine)
            #print(f"bcc2subss_fine:{self.ediff_fine} substrs_fine:{substrs_fine} ind:{ind}")
            #print(f"compstr:{self.compstr} gmix:{self.gmix}")
            #print(f"bcc2subss: {self.ediff2} substrs:{self.substrs_2}")
            #print(f"bcc2subss_fine:{self.ediff_fine} substrs_fine:{self.substrs_fine}")
            #print("=====")

            if self.ediff_fine < self.ediff2:
                return self.ediff_fine, self.substrs_fine
            else:
                return self.ediff2, self.substrs_2
        else:
            return self.ediff2, self.substrs_2