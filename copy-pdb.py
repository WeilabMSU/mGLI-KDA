import os
import pandas as pd

datasets = ["large", "medium", "small"]

for dataset in datasets:
    PDBs = pd.read_csv(f"datasets/Set-364/{dataset}set.csv", header=0)["PDBID"]
    for PDB in PDBs:
        os.system(f"cp datasets/Set-364/{PDB}_CA_A2.pdb datasets/Bfactor-{dataset}")
