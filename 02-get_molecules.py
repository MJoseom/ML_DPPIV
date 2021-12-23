import pandas as pd
#from rdkit import Chem
#from chembl_structure_pipeline import standardizer
import subprocess
#from rdkit.Chem import inchi
from openbabel import pybel

#Generate the SMILES file
def getSMILES(dataframe,  label):
    """Function to generate a file with the SMILES from a ChEMBL dataframe"""
    df = pd.read_csv(dataframe,  sep=",")
    df = df[['Smiles','Molecule ChEMBL ID']].to_csv( "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/"+label+"_DPPIV_ChEMBL.smi",  header=False,  index=False,  sep="\t")

#Generate the SDF file
def getSDF(dataframe,  label):
    """Function to generate the 3D molecules exporting to SDF file from a ChEMBL dataframe"""
    df = pd.read_csv(dataframe,  sep=",")
    moleculesdict = pd.Series(df['Smiles'].values,index=df['Molecule ChEMBL ID']).to_dict()
    
    outsdf = Chem.SDWriter("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/"+label+"_DPPIV.sdf")
    for name,  smile in moleculesdict.items():
        print(name)
        mol = Chem.MolFromSmiles(smile) 
        #Standarize, Remove Salts and ions
        std_mol = standardizer.standardize_molblock(Chem.MolToMolBlock(mol))
        mol, _ = standardizer.get_parent_molblock(std_mol)
        mol = Chem.MolFromMolBlock(mol)
        #Identify the molecule
        mol.SetProp("_Name",name)
        #Identify the group of the molecule
        mol.SetProp('Group', label)
        #Export to SDF file
        outsdf.write(mol)
    outsdf.close()
    
    molfile = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/"+label+"_DPPIV.sdf"
    outsmi = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/"+label+"_DPPIV_std.smi"    
    # 3D structure
    sdf3d = " ".join(["/usr/bin/obabel",  molfile,  "-O",  molfile, "--gen3D" ])
    subprocess.call(sdf3d.split())
    # Remove H
    noH = " ".join(["/usr/bin/obabel",  molfile,  "-O",  molfile, "-d" ])
    subprocess.call(noH.split())
    # Protonation in a specific pH
    protonation = " ".join(["/usr/bin/obabel",  molfile,  "-O",  molfile, "-p 7.4" ])
    subprocess.call(protonation.split())
    # Output to SMILES
    smiles = " ".join(["/usr/bin/obabel",  molfile,  "-O",  outsmi])
    subprocess.call(smiles.split())
    
                
actives = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/DPPIV_actives.csv"
getSMILES(actives,  "actives")
getSDF(actives,  "actives")

inactives = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/DPPIV_inactives.csv"
getSMILES(inactives,  "inactives")
getSDF(inactives,  "inactives")

#https://github.com/chembl/ChEMBL_Structure_Pipeline
#https://doi.org/10.1186/s13321-020-00456-1

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

"""Check Inchikeys"""
def CheckInchikeys(mollist):
    inchikeys = {}
    repetits = []
    for mol in mollist:
        inchikey = inchi.MolToInchiKey(mol)
        if inchikey not in inchikeys.values():
            inchikeys[mol.GetProp('_Name')] = inchikey
        else:
            repetits.append(mol.GetProp('_Name'))
            duplicate = [name for name, inchi in inchikeys.items() if inchi == inchikey]
            print("Duplicates:",  mol.GetProp('_Name'), duplicate,  mol.GetProp('Label'))
    return inchikeys,  repetits

inchikeys,  repetits = CheckInchikeys(ImporMolecules("./actives_DPPIV_std.smi",  "active") + ImporMolecules("./inactives_DPPIV_std.smi",  "inactive"))
print("Total inchikeys:",  len(inchikeys))
print("Duplicate molecules:",  repetits)

def RemovingDuplicates(molfile,  duplicatelist):
    mols = open(molfile,  "r")
    outfile = open(molfile.replace(".smi",  "02.smi"), "w")
    for line in mols:
        name = line.split()[-1].split("\n")[0]
        if not name in duplicatelist:
            outfile.write(line)

RemovingDuplicates("./actives_DPPIV_std.smi",  repetits)
RemovingDuplicates("./inactives_DPPIV_std.smi",  repetits)

def RemovingDuplicatesSDF(smilefile,  sdfile):
    sdfiles = pybel.readfile("sdf",  sdfile)
    smifile = pybel.readfile("smi",  smilefile)
    names = [mol.title for mol in smifile]
    out = pybel.Outputfile("sdf", sdfile.replace(".sdf",  "02.sdf"), overwrite=False)
    for mol in sdfiles: 
        if mol.title in names:
            out.write(mol)
RemovingDuplicatesSDF("./actives_DPPIV_std02.smi",  "./actives_DPPIV.sdf")
RemovingDuplicatesSDF("./inactives_DPPIV_std02.smi",  "./inactives_DPPIV.sdf")
