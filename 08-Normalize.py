import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler

directory = "/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/06-Descriptors/"

"""Identification of outliers"""

descriptors = pd.read_csv(directory+"03-Descriptors_DPPIV_cleaned.csv",  sep=",")
print(descriptors.shape)

z = np.abs(stats.zscore(descriptors.iloc[:, 4:]))
print([len(set(i)) for i in np.where(z > 3)])

"""Descriptors normalization"""

"""z-score"""
# create a scaler object
std_scaler = StandardScaler()
std_scaler
# fit and transform the data
Normdescriptors = pd.DataFrame(std_scaler.fit_transform(descriptors.iloc[:, 4:]), columns=descriptors.iloc[:, 4:].columns)

Normdescriptors = pd.concat([descriptors.iloc[:, :4 ], Normdescriptors],  axis=1)
print("Total descriptors normalized:", Normdescriptors.shape)
Normdescriptors.to_csv(directory+"04-Descriptors_DPPIV_clean_normalized.csv",  index=False)

"""min-max scaling"""

#def normalize_property(descriptor):
#    descriptor = [value - min(descriptor) for value in descriptor]
#    DescNorm = [value / max(descriptor) for value in descriptor]
#    return DescNorm
#    
#def normalize(df):
#    result = df.copy()
#    for feature_name in df.columns:
#        max_value = df[feature_name].max()
#        min_value = df[feature_name].min()
#        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#    return result
#
#Normdescriptors =  normalize(descriptors.iloc[:, 3: ])
#Normdescriptors = pd.concat([descriptors.iloc[:, :3 ],Normdescriptors],   axis=1)
#print("Total descriptors normalized:", Normdescriptors.shape)
#Normdescriptors.to_csv("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/05-Descriptors/Descriptors_DPPIV_normalized.csv")
#
