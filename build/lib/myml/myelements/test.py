import pandas as pd
showcols = ["Composition", "Exp_Shear"]
df = pd.read_csv("External.csv")
print(df[showcols])
a = 23.26552782 + 17.63561142 + 17.98483995 + 15.6279375
a /= 4
print(a)