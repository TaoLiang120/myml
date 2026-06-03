import numpy as np
import json
import random
from itertools import combinations
from myml.myglobal import Element_list


def get_ncom_comps(ncom, bs=10, elements=Element_list, Only_EA=False):
    nele = len(elements)
    inds = np.arange(nele, dtype=int)
    combs = combinations(inds, ncom)
    comps = []
    ntot = 0
    itypes = np.zeros(ncom, dtype=int)
    isyms = np.array(["Ti"] * ncom)
    for comb in list(combs):
        for itype in range(ncom):
            itypes[itype] = comb[itype]
            isyms[itype] = elements[itypes[itype]]
        if Only_EA:
            thisstr = ""
            for itype in range(ncom):
                thisstr += isyms[itype]
            comps.append(thisstr)
            ntot += 1
        else:
            cs = np.zeros(ncom, dtype=int)
            if ncom == 2:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    cs[ncom - 1] = 100
                    for i in range(ncom - 1):
                        cs[ncom - 1] -= cs[i]
                    if cs[ncom - 1] < bs - 0.5:
                        pass
                    else:
                        thisstr = ""
                        for i in range(ncom):
                            thisstr += isyms[i] + str(cs[i])
                        comps.append(thisstr)
                        ntot += 1
            elif ncom == 3:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    for cs[1] in range(bs, 100 - cs[0] + 1, bs):
                        cs[ncom - 1] = 100
                        for i in range(ncom - 1):
                            cs[ncom - 1] -= cs[i]
                        if cs[ncom - 1] < bs - 0.5:
                            pass
                        else:
                            thisstr = ""
                            for i in range(ncom):
                                thisstr += isyms[i] + str(cs[i])
                            comps.append(thisstr)
                            ntot += 1
            elif ncom == 4:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    for cs[1] in range(bs, 100 - cs[0] + 1, bs):
                        for cs[2] in range(bs, 100 - cs[0] - cs[1] + 1, bs):
                            cs[ncom - 1] = 100
                            for i in range(ncom - 1):
                                cs[ncom - 1] -= cs[i]
                            if cs[ncom - 1] < bs - 0.5:
                                pass
                            else:
                                thisstr = ""
                                for i in range(ncom):
                                    thisstr += isyms[i] + str(cs[i])
                                comps.append(thisstr)
                                ntot += 1
            elif ncom == 5:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    for cs[1] in range(bs, 100 - cs[0] + 1, bs):
                        for cs[2] in range(bs, 100 - cs[0] - cs[1] + 1, bs):
                            for cs[3] in range(bs, 100 - cs[0] - cs[1] - cs[2] + 1, bs):
                                cs[ncom - 1] = 100
                                for i in range(ncom - 1):
                                    cs[ncom - 1] -= cs[i]
                                if cs[ncom - 1] < bs - 0.5:
                                    pass
                                else:
                                    thisstr = ""
                                    for i in range(ncom):
                                        thisstr += isyms[i] + str(cs[i])
                                    comps.append(thisstr)
                                    ntot += 1
            elif ncom == 6:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    for cs[1] in range(bs, 100 - cs[0] + 1, bs):
                        for cs[2] in range(bs, 100 - cs[0] - cs[1] + 1, bs):
                            for cs[3] in range(bs, 100 - cs[0] - cs[1] - cs[2] + 1, bs):
                                for cs[4] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] + 1, bs):
                                    cs[ncom - 1] = 100
                                    for i in range(ncom - 1):
                                        cs[ncom - 1] -= cs[i]
                                    if cs[ncom - 1] < bs - 0.5:
                                        pass
                                    else:
                                        thisstr = ""
                                        for i in range(ncom):
                                            thisstr += isyms[i] + str(cs[i])
                                        comps.append(thisstr)
                                        ntot += 1
            elif ncom == 7:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    for cs[1] in range(bs, 100 - cs[0] + 1, bs):
                        for cs[2] in range(bs, 100 - cs[0] - cs[1] + 1, bs):
                            for cs[3] in range(bs, 100 - cs[0] - cs[1] - cs[2] + 1, bs):
                                for cs[4] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] + 1, bs):
                                    for cs[5] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] - cs[4] + 1, bs):
                                        cs[ncom - 1] = 100
                                        for i in range(ncom - 1):
                                            cs[ncom - 1] -= cs[i]
                                        if cs[ncom - 1] < bs - 0.5:
                                            pass
                                        else:
                                            thisstr = ""
                                            for i in range(ncom):
                                                thisstr += isyms[i] + str(cs[i])
                                            comps.append(thisstr)
                                            ntot += 1
            elif ncom == 8:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    for cs[1] in range(bs, 100 - cs[0] + 1, bs):
                        for cs[2] in range(bs, 100 - cs[0] - cs[1] + 1, bs):
                            for cs[3] in range(bs, 100 - cs[0] - cs[1] - cs[2] + 1, bs):
                                for cs[4] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] + 1, bs):
                                    for cs[5] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] - cs[4] + 1, bs):
                                        for cs[6] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] - cs[4] - cs[5] + 1, bs):
                                            cs[ncom - 1] = 100
                                            for i in range(ncom - 1):
                                                cs[ncom - 1] -= cs[i]
                                            if cs[ncom - 1] < bs - 0.5:
                                                pass
                                            else:
                                                thisstr = ""
                                                for i in range(ncom):
                                                    thisstr += isyms[i] + str(cs[i])
                                                comps.append(thisstr)
                                                ntot += 1
            elif ncom == 9:
                for cs[0] in range(bs, 100 - (ncom - 1) * bs + 1, bs):
                    for cs[1] in range(bs, 100 - cs[0] + 1, bs):
                        for cs[2] in range(bs, 100 - cs[0] - cs[1] + 1, bs):
                            for cs[3] in range(bs, 100 - cs[0] - cs[1] - cs[2] + 1, bs):
                                for cs[4] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] + 1, bs):
                                    for cs[5] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] - cs[4] + 1, bs):
                                        for cs[6] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] - cs[4] - cs[5] + 1, bs):
                                            for cs[7] in range(bs, 100 - cs[0] - cs[1] - cs[2] - cs[3] - cs[4] - cs[5] -cs[6] + 1, bs):
                                                cs[ncom - 1] = 100
                                                for i in range(ncom - 1):
                                                    cs[ncom - 1] -= cs[i]
                                                if cs[ncom - 1] < bs - 0.5:
                                                    pass
                                                else:
                                                    thisstr = ""
                                                    for i in range(ncom):
                                                        thisstr += isyms[i] + str(cs[i])
                                                    comps.append(thisstr)
                                                    ntot += 1

    combs = combinations(inds, ncom)
    neq = len(list(combs))
    print(f"ncom:{ncom} ntot:{ntot} neq: {neq} ncons:{ntot / neq}")
    return comps

def get_ncoms_comps(ncoms, bs=10, elements=Element_list, Only_EA=False):
    if isinstance(Only_EA, bool):
        Only_EA = [Only_EA]*len(ncoms)
    allcomps = np.array([])
    for icom in range(len(ncoms)):
        ncom = ncoms[icom]
        thisonly = Only_EA[icom]
        comps = get_ncom_comps(ncom, bs=bs, elements=elements, Only_EA=thisonly)
        allcomps = np.append(allcomps, comps)
    return allcomps

def comps2json(comps, fhead="cs_", n=None):
    random.shuffle(comps)
    if n == None:
        fname = fhead + ".json"
        with open(fname, "w") as f:
            json.dump(list(comps), f)
    else:
        nt = len(comps)
        ng = int(nt / n)
        nr = nt - ng * n
        if nr > 0: ng += 1
        print(f"number group:{ng}")
        for i in range(ng):
            thiscs = comps[i * n: min((i + 1) * n, nt)]
            fname = fhead + str(i) + ".json"
            with open(fname, "w") as f:
                json.dump(list(thiscs), f)