from myml.data.reorganizer import DataReorganizer

'''
fname = "DovaleFarelo_Hardness.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer1()
DReorg.save_to()
'''

'''
fname = "Beniwal_Hardness.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer2()
DReorg.save_to()
print("finished " + fname)
'''

'''
import shutil
shutil.copy("EXTERNAL_DATABASE/Beniwal_Hardness_EXT.csv", "Beniwal_Hardness_EXT.csv")
fname = "MPEA_Hardness.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer7(DataAugment=True, fname="Beniwal_Hardness_EXT.csv")
DReorg.save_to()
print("finished " + fname)
'''

'''
fname = "Giles_YS.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer3()
DReorg.save_to()
print("finished " + fname)
'''

'''
import shutil
shutil.copy("EXTERNAL_DATABASE/Giles_YS_EXT.csv", "Giles_YS_EXT.csv")
fname = "MPEA_dataset.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer4(DataAugment=True, fname="Giles_YS_EXT.csv")
DReorg.save_to()
print("finished " + fname)
'''

'''
import shutil
shutil.copy("EXTERNAL_DATABASE/Giles_YS_EXT.csv", "Giles_YS_EXT.csv")
fname = "MPEA_dataset.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer4(DataAugment=False, fname="Giles_YS_EXT.csv", temp_augment=False)
DReorg.save_to(outfile = "MPEA_dataset_noAug_EXT.csv")
print("finished " + fname)
'''

'''
fname = "Compression.csv"
DReorg = DataReorganizer(fname)
### MIX = BCC + FCC, BCC = Phase must include BCC phase
#### ROM on bcc2hcp of elements, if no NDIP bcc2hcp model
#### ZERO on bcc2hcp, if no NDIP bcc2hcp model
### BCC + ZERO = 45 data; BCC + ROM = 107 data and MIX+ROM = 121 data
#DReorg.reorganizer5(Phases = "MIX", Exception_1="ROM", Apply_b2h=False)
DReorg.reorganizer5(Phases = "BCC", Exception_1="ROM")
DReorg.save_to()
print("finished " + fname)
'''


'''
fname = "Elongation.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer6(Phases = "BCCONLY", Exception_1="ROM", app_b2h=False)
DReorg.save_to()
print("finished " + fname)
'''



import shutil
fname = "JT_tensile_2025OCT25.csv"
shutil.copy("EXTERNAL_DATABASE/Elongation_EXT.csv", "Elongation_EXT.csv")
DReorg = DataReorganizer(fname)
DReorg.reorganizer8(Phases = "BCCONLY", Exception_1="ROM", fname="Elongation_EXT.csv", app_b2h=False)
DReorg.save_to(outfile="JT_tensile_EXT.csv")
print("finished " + fname)




import shutil
fname = "JT_compress.csv"
shutil.copy("EXTERNAL_DATABASE/Compression_EXT.csv", "Compression_EXT.csv")
DReorg = DataReorganizer(fname)
DReorg.reorganizer9(Phases = "BCC", Exception_1="ROM", fname="Compression_EXT.csv", app_b2h=False)
DReorg.save_to()
print("finished " + fname)

import shutil
fname = "JT_Vickers.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer11(Phases = "BCC", Exception_1="ROM")
DReorg.save_to()
print("finished " + fname)

import os
import pandas as pd
import copy
import shutil
import numpy as np

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from myml.data.data import myData
from myml.myglobal import Element_Basic, Element_list, Element_Additional, Element_negativity

shutil.copy("EXTERNAL_DATABASE/JT_compress_EXT.csv", "JT_compress_EXT.csv")
data_compress = myData("JT_compress_EXT.csv", Source="EXT")
gooddf = copy.deepcopy(data_compress.df)
ncompons = [3, 4, 5, 6, 7, 8, 9, 10, 11]
#gooddf, baddf = data_compress.select_by_ncompons(gooddf, ncompons)
rcs = gooddf["ReducedFormula"].to_numpy()
tms = gooddf["Tm_ROM"].to_numpy()
elongs = gooddf["Elongation_EXP"].to_numpy()
sbss = gooddf["ShearBurger_Surf"].to_numpy()
inds = []


compspace = ["W", "Mo", "Cr"]
for i in range(len(rcs)):
    rc = rcs[i]
    elong = elongs[i]
    tm = tms[i]
    sbs = sbss[i]

    comp = Composition(rc)
    conc = 0.0
    for el in comp.elements:
        if el.symbol in compspace:
            conc += comp.get_atomic_fraction(el)

    if conc >= 0.20:
        if sbs < 6.5:
            pass
        else:
            if tm < 2300:
                pass
            else:
                inds.append(i)
        print(f"=== rc:{rc} tm:{tm} sbs:{sbs} elong:{elong} ===")

    '''
    if tm > 2600 and sbs>5.5:
        if elong > 55:
            pass
        else:
            inds.append(i)
        print(f"=== rc:{rc} tm:{tm} sbs:{sbs} elong:{elong} ===")
    '''
        
inds = np.array(inds)
df_add = gooddf.iloc[inds]
elongs = np.zeros(len(df_add))
df_add["Elongation_EXP"] = elongs
print(len(df_add))
print("000")

shutil.copy("EXTERNAL_DATABASE/JT_tensile_EXT.csv", "JT_tensile_EXT.csv")
data_tensile = myData("JT_compress_EXT.csv", Source="EXT")
gooddf = copy.deepcopy(data_tensile.df)
print(f"org tensile:{len(gooddf)}")
ncompons = [3, 4, 5, 6, 7, 8, 9, 10, 11]
gooddf, baddf = data_tensile.select_by_ncompons(gooddf, ncompons)
print(f"after tensile:{len(gooddf)}")
outdf = pd.concat([gooddf, df_add])
print(f"after add compressive tensile:{len(outdf)}")

fout = "JT_tensile_compress0_EXT.csv"
fout = os.path.join("EXTERNAL_DATABASE", fout)
data_tensile.save_to(fout, df=outdf)
print(len(outdf))


#### only 9 elements ####
from myml.data.data_util import compute_df_concs

fnames = ["JT_compress_EXT.csv", "JT_tensile_EXT.csv"]
outfiles = ["JT_compress4paper_EXT.csv", "JT_tensile4paper_EXT.csv"]

cols = ["Composition", "CONC0", "CONC1", "CONC3", "CONC12", "CONC13"]


for i in range(len(fnames)):
    fname = fnames[i]
    thisdata = myData(fname, Source="EXT")
    gooddf, baddf = thisdata.select_by_CS(thisdata.df, Element_Basic, "INC", ncompon=None, style=1, set2gooddf=False)
    gooddf = compute_df_concs(gooddf, elements=Element_negativity)
    outfile = outfiles[i]
    outfile =  os.path.join("EXTERNAL_DATABASE", outfile)
    thisdata.save_to(outfile, df=gooddf)
    print(f"finished: {outfile} ndata:{len(gooddf)}")

checkfile = "JT_compress4paper_EXT.csv"
data1 = myData(os.path.join("EXTERNAL_DATABASE", checkfile), Source="EXT")
gooddf = copy.deepcopy(data1.df)
rcs1 = gooddf["ReducedFormula"].to_numpy()
cs1 = gooddf["Composition"].to_numpy()
tms1 = gooddf["Tm_ROM"].to_numpy()
elong1 = gooddf["Elongation_EXP"].to_numpy()
b2hs1 = gooddf["bcc2hcp"].to_numpy()
sbss = gooddf["ShearBurger_Surf"].to_numpy()
nadd = 0
inds = []
compspace = ["W", "Mo", "Cr"]
for i in range(len(rcs1)):
    rc = rcs1[i]
    cs = cs1[i]
    b2h = b2hs1[i]
    elong = elong1[i]
    sbs = sbss[i]
    tm = tms1[i]

    comp = Composition(rc)
    conc = 0.0
    for el in comp.elements:
        if el.symbol in compspace:
            conc += comp.get_atomic_fraction(el)

    if conc >= 0.20:
        if sbs < 6.5:
            pass
        else:
            if tm < 2300:
                pass
            else:
                inds.append(i)
        print(f"=== rc:{rc} tm:{tm} sbs:{sbs} elong:{elong} ===")

    '''
    if tm > 2600 and sbs>5.5:
        if elong > 55:
            pass
        else:
            inds.append(i)
        print(f"=== rc:{rc} tm:{tm} sbs:{sbs} elong:{elong} ===")
    '''

inds = np.array(inds)
df_add = gooddf.iloc[inds]
elongs = np.zeros(len(df_add))
df_add["Elongation_EXP"] = elongs

print(len(df_add))
ftensile = "JT_tensile4paper_EXT.csv"
data = myData(os.path.join("EXTERNAL_DATABASE", ftensile), Source="EXT")
outdf = pd.concat([data.df, df_add])
fout = "JT_tensile_compress0_9eles_EXT.csv"
fout = os.path.join("EXTERNAL_DATABASE", fout)
data.save_to(fout, df=outdf)
print(len(outdf))
print(f"=== finished {fout} ===")
