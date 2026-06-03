import os, sys
import numpy as np
import pandas as pd
import json
from myml.predict.predict import Screener
from myml.data.data import DATA_PATH

fname= "my_predicts.csv"
screenkey = "Yield_M1"
temp = 1500.0

print(f"temp:{temp} filename:{fname}")
thisscr = Screener(fname, from_database=False)
thisscr.value_screener("temperature", temp-100, temp+100)


'''
#thisscr.value_screener("Gmix_TC", "inf", 0.01)
#thisscr.value_screener("volume_DELTA", "inf", 0.2)
#thisscr.value_screener("bcc2laves", -0.02, "inf")
thisscr.value_screener("bcc2b2", -0.02, "inf")
thisscr.screen_elements(["Hf"], style="EXCLUDE")
thisscr.value_screener("bcc2hcp_Gmix_TC2CPA", 0.1, 0.42)
thisscr.value_screener("Elong_M0", 13.0, "inf")
thisscr.ratio_screener(screenkey, 0.85, "inf")
#thisscr.ratio_screener(screenkey, 0.5, "inf")
thisscr.ratio_screener("delta_Eb0", 0.40, "inf")
#thisscr.percentage_screener("delta_Eb0", 0.5, "inf")
#thisscr.percentage_screener(screenkey, 0.50, "inf")
#thisscr.ratio_screener(screenkey, 0.5, "inf")
'''

ShortCols = ["Composition", "LattPara", "radius_TC",
            "Gmix_TC", "bcc2hcp", "bcc2laves", "bcc2b2",
            "Exp_Bulk_ROM", "Exp_Shear_ROM", "Exp_Youngs_ROM",
            "Tm_ROM", "temperature", "Yield_M1",
            "Compress_M2", "Elong_M2",
            "Vicker_M1"]
thisscr.display_screener(DisplayCols=None, sort_by=[screenkey])
thisscr.save_screener(outfile="myscreen.csv", ShortCols=ShortCols)
print("===========")
  
        
