import os
import json
import pandas as pd

bcc_keys = ["Au", "Az", "BOG", "Bulk", "Bprime", "C11", "C12", "C44",
            "Composition", "CompID", "Crystal", "Eform", "Eform_str",
            "Group", "LattPara", "Poisson", "Shear",
            "Etot", "sws", "Youngs", "bcc2hcp", "bcc2sublatt", "Smix", "Gmix"]

hcp_keys = ["Au", "BOG", "Bulk", "Bprime", "C11", "C12", "C44", "C13", "C33",
            "Composition", "CompID", "Crystal", "Eform", "Eform_str",
            "Group", "LattPara", "COA", "Poisson", "Shear",
            "Etot", "sws", "R0", "CS0", "Youngs", "Smix", "Gmix"]

sfs_keys = ["Composition", "CompID", "Crystal", "SF Plane", "Direction", "Displacement",
            "SF Energy", "Group", "Lattice parameters", "Total energy", "sws", "Smix", "Gmix"]

sublatt_keys = ["Au", "Az", "BOG", "Bulk", "Bprime", "C11", "C12", "C44",
                "Composition", "CompID", "Crystal", "Eform", "Eform_str",
                "Group", "LattPara", "Poisson", "Shear",
                "Etot", "sws", "Youngs", "bcc2sublatt", "subID", "substrs", "Smix", "Gmix"]

feature_keys = ["Au", "Az", "BOG", "Bulk", "Bprime", "C11", "C12", "C44",
                "Eform", "Eform_str", "LattPara", "Poisson", "Shear",
                "sws", "Youngs", "bcc2hcp", "bcc2sublatt", "Gmix"]

bcc_extreme_file = "bccC1.csv"
DF_bcc = pd.read_csv(os.path.join(os.path.dirname(__file__), bcc_extreme_file))

hcp_extreme_file = "hcpC1.csv"
DF_hcp = pd.read_csv(os.path.join(os.path.dirname(__file__), hcp_extreme_file))

Element_basic_file = "elementBasic.csv"
DF_basic = pd.read_csv(os.path.join(os.path.dirname(__file__), Element_basic_file))

Element_extreme_keys = ["LattPara", "Bulk", "C11", "C12", "C44", "Youngs", "Shear", "sws"]
Element_extreme_file = "elementC1.csv"
DF_ground = pd.read_csv(os.path.join(os.path.dirname(__file__), Element_extreme_file))

additional_keys = ["Au", "Az", "BOG", "Bprime", "Eform", "Eform_str", "Poisson", "bcc2hcp", "bcc2sublatt", "Gmix"]

eos_keys = ["sws", "Bulk", "Eform", "Eform_str"]
#eos_keys = ["sws"]
hcp_eos_keys = ["COA"]
transformkey = "bcc2hcp"
elastic_keys = []
BOKAS_key = "Eform_BOKAS"
TC_keys = ["radius_TC", "Hmix_TC", "Gmix_TC", "Hmix_str_TC", "Gmix_str_TC", "bcc2hcp_Gmix_TC", "bcc2hcp_Gmix_TC2CPA"]
thermo_scale_keys = ["Eform", "Eform_str", "Eform_BOKAS", "Hmix_TC", "Gmix_TC", "Hmix_str_TC", "Gmix_str_TC"]

ndip_all_keys = eos_keys + [BOKAS_key] + TC_keys + [transformkey] + elastic_keys

#eos_boost_keys = ["Eform", "Gmix", "bcc2hcp", "Bulk"]
#elastic_boost_keys = ["BOG", "C11", "C12", "C44", "Poisson", "Shear", "Youngs"]


Element_external_file = "External.csv"
DF_external = pd.read_csv(os.path.join(os.path.dirname(__file__), Element_external_file))

with open(os.path.join(os.path.dirname(__file__), "omegas.json"), "rt") as f:
    s = f.read()
omega_binary = json.loads(s)

Lavesfile = "Laves.csv"
DF_Laves = pd.read_csv(os.path.join(os.path.dirname(__file__), Lavesfile))
DF_Laves = DF_Laves.set_index("Compstr")

B2Lattfile = "B2Latt.csv"
DF_B2Latt = pd.read_csv(os.path.join(os.path.dirname(__file__), B2Lattfile))
DF_B2Latt = DF_B2Latt.set_index("Compstr")
