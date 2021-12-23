from rdkit import Chem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors
import numpy as np
import pandas as pd

def ReadMoleculesSDF(sdffile):
    molList = []
    sdf= Chem.SDMolSupplier(sdffile)
    print(len(sdf))
    for mol in sdf:
        try:
            name = mol.GetProp("_Name")
            molList.append(mol)
        except:
            print("cannot be treated with RDKit")
    print("Molecules:",  len(molList))
    return molList
    
def GenerateDescriptors(molList,  descriptorsList,  label):
    baseData= np.arange(1, len(descriptorsList)+3)
    for mol in molList:
        descriptors = []
        descriptors.append(mol.GetProp("_Name"))
        descriptors.append(label)
        
        #PMI1
        descriptors.append(Descriptors3D.PMI1(mol))
        #PMI2
        descriptors.append(Descriptors3D.PMI2(mol))
        #PMI3
        descriptors.append(Descriptors3D.PMI3(mol))
        #NPR1
        descriptors.append(Descriptors3D.NPR1(mol))
        #NPR2
        descriptors.append(Descriptors3D.NPR2(mol))
        #RadiusOfGyration
        descriptors.append(Descriptors3D.RadiusOfGyration(mol))
        #InertialShapeFactor
        descriptors.append(Descriptors3D.InertialShapeFactor(mol))
        #Eccentricity
        descriptors.append(Descriptors3D.Eccentricity(mol))
        #Asphericity
        descriptors.append(Descriptors3D.Asphericity(mol))
        #SpherocityIndex
        descriptors.append(Descriptors3D.SpherocityIndex(mol))
        #PBF
        descriptors.append(rdMolDescriptors.CalcPBF(mol))
        #WHIM
        descriptors.append(rdMolDescriptors.CalcWHIM(mol))
        #Autocorr3D
        descriptors.append(rdMolDescriptors.CalcAUTOCORR3D(mol))
        #RDF
        descriptors.append(rdMolDescriptors.CalcRDF(mol))
        #MORSE
        descriptors.append(rdMolDescriptors.CalcMORSE(mol))
        #GETAWAY
        descriptors.append(rdMolDescriptors.CalcGETAWAY(mol))
        
        baseData=np.vstack([baseData, descriptors])
    descriptors = pd.DataFrame(data=baseData[1:],columns=["Name"] + ["Group"] +descriptorsList)
    return descriptors
    
descriptors3D = ['PMI1','PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration', 'InertialShapeFactor', 'Eccentricity', 'Asphericity', 'SpherocityIndex', 'PBF', 'WHIM', 'Autocorr3D', 'RDF', 'MORSE', 'GETAWAY']

activesfile = "/home/mojeda/Documents/Other_things/Machine_Learning/actives_DPPIV.sdf" 
descriptors = GenerateDescriptors(ReadMoleculesSDF(activesfile), descriptors3D, 1)
descriptors.to_csv('/home/mojeda/Documents/Other_things/Machine_Learning/actives_DPPIV_Descriptors3D.csv', sep= ",",  index=False, header=True) 

inactivesfile = "/home/mojeda/Documents/Other_things/Machine_Learning/inactives_DPPIV.sdf" 
descriptors = GenerateDescriptors(ReadMoleculesSDF(inactivesfile), descriptors3D, 0)
descriptors.to_csv('/home/mojeda/Documents/Other_things/Machine_Learning/inactives_DPPIV_Descriptors3D.csv', sep= ",",  index=False, header=True) 


###Hacer conformaciones para los descriptores 3D???????
###Que hacer con los descriptores que tienen una lista como output? media?
