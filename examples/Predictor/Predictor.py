import os, sys
import numpy as np
import pandas as pd

from myml.predict.predict import ExtendPredictor

float_format = '%.8f'
finput = "predict.input"

thisPre = ExtendPredictor.from_finput(finput)
thisPre.get_all()

thisPre.display_predictions()



 
