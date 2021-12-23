import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import dendrogram, linkage

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

#molecules = molecules[: : 20]

"""Calculate fingerprints: Morgan"""
molsfps= [AllChem.GetMorganFingerprint(mol,2,useFeatures=True) for mol in molecules]
print("Molecules fp:",  len(molsfps))

"""Calculate Similarity: Tanimoto """
size=len(molecules)
hmap=np.empty(shape=(size,size))
table=pd.DataFrame()
for index, i in enumerate(molsfps):
    for jndex, j in enumerate(molsfps):
        similarity=DataStructs.TanimotoSimilarity(i , j)
        hmap[index,jndex]=similarity
        table.loc[molecules[index].GetProp('_Name'),molecules[jndex].GetProp('_Name')]=similarity
print(table.head(10))
print(hmap)
table.to_csv("./Similarity_dataset_DPPIV.csv",  index=True)


"""Clustering"""
nameList = [mol.GetProp('_Name') for mol in molecules]

linked = linkage(hmap,'single')
plt.figure(figsize=(12,15))

ax1=plt.subplot()
o=dendrogram(linked,  
            orientation='left',
            labels=nameList,
            distance_sort='descending',
            show_leaf_counts=True,  
            color_threshold=0, 
            above_threshold_color='grey')

ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.title('Similarity clustering',fontsize=20,weight='bold')
plt.tick_params ('both',width=2,labelsize=8)
ax = plt.gca()
xlbls = ax.get_ymajorticklabels()

#for lbl in xlbls:
#    if lbl.get_text() == "A":
#        lbl.set_color("green")
#    else:
#        lbl.set_color("red")

plt.tight_layout()
#plt.show() 

# This will give us the clusters in order as the last plot
new_data=list(reversed(o['ivl']))
labelList = [mol.GetProp('Label') for mol in molecules]
nameList = [ "A" if label == "active" else "D" for label in labelList]
# we create a new table with the order of HCL
hmap_2=np.empty(shape=(size,size))
for index,i in enumerate(new_data):
    for jndex,j in enumerate(new_data):
        hmap_2[index,jndex]=table.loc[i].at[j]
        
figure= plt.figure(figsize=(30,30))
gs1 = gridspec.GridSpec(2,7)
gs1.update(wspace=0.01)
ax1 = plt.subplot(gs1[0:-1, :2])
dendrogram(linked, orientation='left', distance_sort='descending',show_leaf_counts=True, labels=nameList,  color_threshold=0, 
            above_threshold_color='grey')
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='y', which='major', labelsize=4)
ax = plt.gca()
xlbls = ax.get_ymajorticklabels()
for lbl in xlbls:
    if lbl.get_text() == "A":
        lbl.set_color("green")
    else:
        lbl.set_color("red")

ax2 = plt.subplot(gs1[0:-1,2:6])
f=ax2.imshow (hmap_2, cmap='PRGn_r', interpolation='nearest')

ax2.set_title('Fingerprint Similarity',fontsize=20,weight='bold')
ax2.set_xticks (range(len(new_data)))
ax2.set_yticks (range(len(new_data)))
ax2.set_xticklabels (new_data,rotation=90,size=8)
ax2.set_yticklabels (new_data,size=8)
ax2.set_axis_off()

ax3 = plt.subplot(gs1[0:-1,6:7])
m=plt.colorbar(f,cax=ax3,shrink=0.75,orientation='vertical',spacing='uniform',pad=0.01)
m.set_label ('Fingerprint Similarity')

plt.tick_params ('both',width=2)
plt.plot()
plt.savefig("./structural_diversity.png")
plt.show()   
