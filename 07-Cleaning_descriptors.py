import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt


"""Generate a unique dataframe containing the name of the molecules, bioactivity, smiles, Active/Inactive, 2D descriptors and 3D descriptors"""
directory = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/06-Descriptors/"

#Features
actives = pd.read_csv('/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/actives_DPPIV.csv',  sep=",",  index_col="Molecule ChEMBL ID")
inactives = pd.read_csv('/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/inactives_DPPIV.csv',  sep=",",  index_col="Molecule ChEMBL ID")
actives["Group"] = "Active"
inactives["Group"] = "Inactive"
featuresdf = pd.concat([actives,  inactives])
print("Total molecules:", featuresdf.shape[0])
#featuresdf.to_csv("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/03-DataSets/dataset_DPPIV_ChEMBL.csv")

#Descriptors files
desc2DActives = pd.read_csv(directory+'Descriptors/actives_DPPIV_Descriptors2D.csv',  sep="," ,  index_col="Name")
desc2DInactives = pd.read_csv(directory+'Descriptors/inactives_DPPIV_Descriptors2D.csv',  sep=",",  index_col="Name")
desc2Ddf = pd.concat([desc2DActives,  desc2DInactives])
print("Total 2D descriptors:", desc2Ddf.shape)

desc3DActives = pd.read_csv(directory+'Descriptors/actives_DPPIV_Descriptors3D.csv',  sep=",",  index_col="Name")
desc3DInactives = pd.read_csv(directory+'Descriptors/inactives_DPPIV_Descriptors3D.csv',  sep=",",  index_col="Name")
desc3Ddf = pd.concat([desc3DActives,  desc3DInactives])
print("Total 3D descriptors:", desc3Ddf.shape)

def SeparateColumns(df, column,  number):
    names = [column + "_"+str(i) for i in range(number)]
    df[names] = df[column].str.replace('[', '').str.replace(']', '').str.split(', ', expand=True).astype(float)
    df = df.drop([column],  axis =1)
    return df
    
desc3Ddf = SeparateColumns(desc3Ddf, "WHIM",  114 )
desc3Ddf = SeparateColumns(desc3Ddf, "Autocorr3D",  80 )
desc3Ddf = SeparateColumns(desc3Ddf, "RDF",  210 )
desc3Ddf = SeparateColumns(desc3Ddf, "MORSE",  224 )
desc3Ddf = SeparateColumns(desc3Ddf, "GETAWAY",  273 )

#print("Total 3D descriptors:", desc3Ddf.shape)
#desc3Ddf.to_csv("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/05-Descriptors/3Ddescriptors_DPPIV.csv")

descFingerprints = pd.read_csv(directory+'Descriptors/Fingerprints_DPPIV.csv',  sep=",",  index_col="name")
#print("Total Fingerprints:", descFingerprints.shape)
descFingerprints = descFingerprints[['fpMorgan2', 'fpMaccs', 'fpTopo']]
print("Total Fingerprints:", descFingerprints.shape)

descFingerprints = SeparateColumns(descFingerprints, "fpMorgan2",  2048 )
descFingerprints = SeparateColumns(descFingerprints, "fpMaccs",  167 )
descFingerprints = SeparateColumns(descFingerprints, "fpTopo",  2048 )

featuresdf.index.names = ['Name']
descriptors = pd.concat([featuresdf[['Standard Value',  'Smiles']],  desc2Ddf, desc3Ddf.iloc[:, 1:], descFingerprints], axis=1,  join="inner").rename_axis('Name')
descriptors.to_csv(directory+"02-Descriptors_DPPIV_splitted.csv")
print("Total:", descriptors.shape)
print("Shape of Descriptors:",  descriptors.iloc[:, 3:].shape)

"""Missing values"""
null_counts = descriptors.isnull().sum()
print("Number of null values:\n{}".format(null_counts[null_counts > 0]))
descriptorsClean = descriptors.dropna(axis=1, how='any',subset=None)
print("Shape of df -Missing values-:",  descriptorsClean.iloc[:, 3:].shape)

"""Columns That Have A Low Variance"""
## split data into inputs and outputs
data = descriptorsClean.iloc[:, 3:].values
X = data[:, :]
print(X.shape)
threshold = 0.05
## define thresholds to check
thresholds = np.arange(0.0, 1, threshold)
## apply transform with each threshold
results = list()
for t in thresholds:
    ## define the transform
    transform = VarianceThreshold(threshold=t)
    ## transform the input data
    X_sel = transform.fit_transform(X)
    ## determine the number of input features
    n_features = X_sel.shape[1]
    print('>Threshold=%.2f, Features=%d' % (t, n_features))
    ## store the result
    results.append(n_features)
## plot the threshold vs the number of selected features
plt.plot(thresholds, results)
plt.show()

threshold = 0.2
variances = descriptorsClean.iloc[:, 3:].var(axis=0)
to_drop = variances[variances <= threshold]
print("Variances <=",  str(threshold),   len(to_drop))
descriptorsClean = descriptorsClean.drop(to_drop.index, axis=1)
print("Shape of df - variances-:",  descriptorsClean.iloc[:, 3:].shape)

"""Correlation of the descriptors"""
corr_matrix = descriptorsClean.iloc[:, 3:].corr().abs()
#print(corr_matrix)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] >= 0.8)]
descriptorsClean = descriptorsClean.drop(to_drop, axis=1)
print("Shape of df - correlation-:",  descriptorsClean.iloc[:, 3:].shape)

print(descriptorsClean.columns.tolist())
descriptorsClean.to_csv(directory+"03-Descriptors_DPPIV_cleaned.csv",  index=True)

#
#"""Duplicate Rows and Columns"""
#dups = descriptorsClean.iloc[:, 4:].duplicated()
#print("Duplicate rows:",  dups.any())
#
#dups = descriptorsClean.T.duplicated()
#print("Duplicate columns:",  dups.any())
#print(descriptorsClean.T[dups].index)
#descriptorsClean = descriptorsClean.drop(columns=descriptorsClean.T[dups].index)
#print("Shape of df -duplicates-:",  descriptorsClean.iloc[:, 4:].shape)

#"""Identify Columns That Contain a Single Value"""
#singleValue = [descriptors.columns[i] for i in range(descriptors.shape[1]) if len(np.unique(descriptors.iloc[:, i])) == 1]
#print("Columns with a single value:", singleValue )
#descriptorsClean = descriptorsClean.drop(columns=singleValue)
#print("Shape of df -single value-:",  descriptorsClean.iloc[:, 4:].shape)
#
#"""Consider Columns That Have Very Few Values"""
#fewValues = []
#for i in descriptorsClean.columns[4:]:
#    percentage = float(len(np.unique(descriptorsClean.loc[:, i]))) / descriptorsClean.shape[0] * 100
#    if percentage < 1:
#        fewValues.append(i)
##        print(i,  round(percentage,  2))
#print("Columns that have unique values that are less than 1% of the number of rows:",  len(fewValues))
#descriptorsClean = descriptorsClean.drop(columns=fewValues)
#print("Shape of df -few values-:",  descriptorsClean.iloc[:, 4:].shape)
