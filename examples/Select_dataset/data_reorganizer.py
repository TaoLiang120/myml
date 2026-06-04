from myml.data.reorganizer import DataReorganizer
fname = "MPEA_dataset.csv"
DReorg = DataReorganizer(fname)
DReorg.reorganizer4(DataAugment=False, temp_augment=False)
DReorg.save_to(outfile = "MPEA_dataset_EXT.csv")
print("finished " + fname)

