import os
import numpy as np
from myml.myglobal import get_compspaces, Element_Basic, Element_list, load_complist_from_json
from myml.data.data import myData, DATA_PATH
from myml.models.models import DataFill_NDIP
from myml.expansion.expansion import DataExpansion, CSExpansion


def datafill_with_NDIP(fname, ncompon, source="INTERNAL",
                       keys=None, elements=Element_Basic,
                       screen=['inf', 'inf'], undotype=1,
                       fill_transform=False, update_transform=False,
                       loadmodel=False, savemodel=True, savefile=True):
    thisdata = myData(os.path.join(DATA_PATH, fname), Source=source)
    compspaces = get_compspaces(ncompon, elements=elements)
    thisDFill = DataFill_NDIP(thisdata, ncompon)
    thisDFill.fill_compspaces(compspaces, keys=keys,
                              fill_transform=fill_transform, update_transform=update_transform,
                              undotype=undotype, extended=True, screen=screen,
                              loadmodel=loadmodel, savemodel=savemodel, savefile=savefile)


def dataexpansion_with_NDIP(fname, keys, from_ncompon,
                            source="INTERNAL", Save_OrgData=False, Scale_Eform=False):
    thisdata = myData(os.path.join(DATA_PATH, fname), Source=source)
    thisDExp = DataExpansion(thisdata, keys, from_ncompon,
                             modelname="NDIP", mname_header=None, features=None,
                             style4Eform="NDIP", Scale_Eform=Scale_Eform)
    thisDExp.fill_data(style=1, Add_Predict=False, Save_Pred2Data=False, OutKey_dict=None, Save_OrgData=Save_OrgData)


def csexpansion_from_json(jsonfile, fname, mode,
                          source="INTERNAL", fill_transform=True, Load_all_NDIP=True,
                          loadmodel=True, savemodel=False, savefile=True, Scale_Eform=False):
    compstrs = load_complist_from_json(jsonfile)
    compstrs = np.array(compstrs)
    #print(len(compstrs))
    #compstrs = compstrs[0:10]

    thisdata = myData(os.path.join(DATA_PATH, fname), Source=source)
    CSE = CSExpansion(compstrs, thisdata, modelname="NDIP", mode=mode,
                      fill_transform=fill_transform, keys=None, subID=0, Temp=300.0, Scale_Eform=Scale_Eform)
    CSE.fill_data(style=1, loadmodel=loadmodel, Load_all_NDIP=Load_all_NDIP,
                  savemodel=savemodel, savefile=savefile, outfile=None)
