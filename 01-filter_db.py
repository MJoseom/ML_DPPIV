#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def CleaningDB(file):
    print(os.path.basename(file))
    #Dataframe from ChEMBL
    db = pd.read_csv(file,  sep=";")
    print("Dataframe from ChEMBL:", db.shape[0])
    
    #Drop those molecules without Activity Values
    db = db.dropna(subset=['Standard Value'])
    print("Removing NA from the activity values:", db.shape[0])
    
    #Drop those molecules with different units of nM
    db = db[db['Standard Units'] == "nM"]
    print("Units different from nM:", db.shape[0])
    
    #Remove duplicates keeping the lowest bioactivity value 
    db = db.sort_values("Standard Value", ascending= True).drop_duplicates('Molecule ChEMBL ID').sort_index()
    print("Remove duplicates:", db.shape[0])
    
    #Molecule IDs to a list
    moldb = db['Molecule ChEMBL ID'].unique().tolist()
    print("List of final molecules:", len(moldb))
    
    return db,  moldb
    
def PlotBioactivity(db,  bioactivity):
    db = db.sort_values(by = ["Standard Value"])
    bioactivities =  db["Standard Value"].tolist()
#    print(bioactivities)
    plt.figure(figsize=(10, 10))
    plt.axvspan( 50, 1000, color='silver',  alpha=0.5)
    bioactivities = [i for i in bioactivities if i <= bioactivity]
    print("Plot of molecules",  len(bioactivities), "Minimum bioactivity value",  min(bioactivities),  "Maximum bioactivity value",  max(bioactivities))
    pd.Series(bioactivities).plot.hist(color='green',  bins=100,  edgecolor='black')
    plt.title('DPP-IV bioactivities from ChEMBL database \n ' + str(len(bioactivities)) +' Molecules IC50 <= '+ str(bioactivity)+' nM',   fontsize=20)
    plt.xlabel("IC50 (nM)",  fontsize=15)
    plt.ylabel('Number of molecules', fontsize=15)
    plt.xticks(np.arange(0, max(bioactivities)+1, 500.0),  fontsize=12)
    plt.yticks(np.arange(0,901, 500.0),  fontsize=12)
    plt.ylim(0, 901)
    plt.axvline(x=50,  linestyle = "dotted",  color= "silver",  linewidth=3)
    plt.axvline(x=1000,  linestyle = "dotted",  color= "silver",  linewidth=3)
    plt.savefig("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/01-Entregas/Figures/DPPIV_distributed_by_Bioactivities.png")
#    
## Cleaning DPP-IV, DPP8 and DPP9 databases from ChEMBL
dpp4 = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/02-ChEMBL_db/ChEMBL_DPPIV.csv"
dpp4,  moldpp4 = CleaningDB(dpp4)
#dpp8 = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/02-ChEMBL_db/ChEMBL_DPP8.csv"
#dpp8,  moldpp8 = CleaningDB(dpp8)
#dpp9 = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/02-ChEMBL_db/ChEMBL_DPP9.csv"
#dpp9,  moldpp9 = CleaningDB(dpp9)
#
##Plot bioactivities
PlotBioactivity(dpp4,  5000)
#
## Active set
#activesdpp4 = dpp4['Molecule ChEMBL ID'].loc[(dpp4["Standard Value"] <=50) & (dpp4["Standard Relation"] != "'>'")].tolist()
#activesdpp8 = dpp8['Molecule ChEMBL ID'].loc[(dpp8['Molecule ChEMBL ID'].isin(activesdpp4)) & (dpp8["Standard Value"] >=1000)]
#activesdpp9 = dpp9['Molecule ChEMBL ID'].loc[(dpp9['Molecule ChEMBL ID'].isin(activesdpp4)) & (dpp9["Standard Value"] >=1000)]
#actives = set(activesdpp8.tolist() + activesdpp9.tolist())
#print("Number of actives:",  len(actives))
#activesdf = dpp4[dpp4["Molecule ChEMBL ID"].isin(actives)]
#activesdf.to_csv("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/actives_DPPIV.csv",  index=False)
#
##Inactive set: Molecules whose IC50 is higher than 1000 nM for DPP-IV
#inactives = dpp4['Molecule ChEMBL ID'].loc[(dpp4["Standard Value"] >=1000) & (dpp4["Standard Relation"] != "'<'")].tolist()
#print("Number of inactives:",  len(inactives))
#inactivesdf = dpp4[dpp4["Molecule ChEMBL ID"].isin(inactives)]
#inactivesdf.to_csv("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/inactives_DPPIV.csv",  index=False)
#
##Gray area (IC50: 50-1000nM)
#grayarea = dpp4.loc[(dpp4["Standard Value"] <1000) & (dpp4["Standard Value"] >50)]
#print("Number of molecules in gray area:",  grayarea.shape[0])



