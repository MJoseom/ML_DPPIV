import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem import AllChem


"""Import molecules"""
def ImporMolecules(file,  label):
    mols=[]
    with open(file,'r') as f:
        for index,line in enumerate(f):
            mol=Chem.MolFromSmiles(line.split()[0]) # Converting SMILES codes into rdkit mol 
            mol.SetProp('_Name',line.split()[1]) # Adding the name for each molecule
            mol.SetProp('Label',label) 
            mols.append(mol)
    return mols

actives = ImporMolecules("./actives_DPPIV_std.smi",  "active")
inactives = ImporMolecules("./inactives_DPPIV_std.smi",  "inactive")
print("Active db:",  len(actives),  "Inactive db:",  len(inactives))
molecules = actives + inactives
print("Total db:",  len(molecules))

def Fingerprints(mol):
    df = pd.DataFrame(columns = ["name",  "fpMorgan2",  "fpMorgan3",  "fpMaccs",  "fpTopo",  "fpPairs"])
    fingerprints = {}
    fingerprints["name"] = mol.GetProp('_Name')
    
    """Calculate fingerprints: Morgan == ECFP4"""
    fpMorgan2 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=2048))
    fingerprints["fpMorgan2"] = fpMorgan2.tolist()

    """Calculate fingerprints: Morgan == ECFP4"""
    fpMorgan3 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True, nBits=2048)) 
    fingerprints["fpMorgan3"] = fpMorgan3.tolist()

    """Calculate fingerprints: MACCS"""
    fpMaccs = np.array(MACCSkeys.GenMACCSKeys(mol)) 
    fingerprints["fpMaccs"] = fpMaccs.tolist()

    """Calculate fingerprints: Topological RDKit"""
    fpTopo = np.array(Chem.RDKFingerprint(mol)) 
    fingerprints["fpTopo"] = fpTopo.tolist()

#    """Calculate fingerprints: Topological Torsions --> Too many bits to be in the dataframe"""
#    fpTorsions = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol) 

    """Calculate fingerprints: Atom Pairs"""
    fpPairs = np.array(Pairs.GetAtomPairFingerprintAsBitVect(mol))
    fingerprints["fpPairs"] = fpPairs.tolist()
    print(len(fingerprints["fpMorgan2"]),  len(fingerprints["fpMorgan3"]),  len(fingerprints["fpMaccs"]),  len(fingerprints["fpTopo"]),  len(fingerprints["fpPairs"]))
    df = df.append(fingerprints,  ignore_index=True)
    df.to_csv("./Fingerprints_DPPIV.csv",  index=False,  mode="a",  header=False)

df = pd.DataFrame(columns = ["name",  "fpMorgan2",  "fpMorgan3",  "fpMaccs",  "fpTopo",  "fpPairs"])
df.to_csv("./Fingerprints_DPPIV.csv",  index=False)
for mol in molecules:
    Fingerprints(mol)

