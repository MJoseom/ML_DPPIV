import pandas as pd
import numpy as np
import statsmodels.api as sm
    
descriptors = pd.read_csv("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/06-Descriptors/04-Descriptors_DPPIV_clean_normalized.csv")
print("Initial shape of df:",  descriptors.shape)

#descriptors["pIC50"] = -np.log10(descriptors["Standard Value"]/1000000000)
#y = descriptors["pIC50"]
y = descriptors["Group"]
x_columns = descriptors.columns[4:-9].tolist()

def get_stats(data,  x_columns, y):
    x = data[x_columns]
    results = sm.OLS(y, x).fit()
#    print(results.summary())
    return results.pvalues

pvalue = 100
while pvalue > 0.05:
    pvalues = get_stats(descriptors, x_columns,  y)
    idpvalue = str(pvalues.idxmax())
    pvalue = pvalues.max()
    x_columns.remove(idpvalue)
    print(idpvalue, pvalue,  len(x_columns))

print(x_columns)
descriptors = pd.concat([descriptors.iloc[:,  :4],  descriptors[x_columns]], axis=1)
print("Shape of df:",  descriptors.shape)
descriptors.to_csv("/home/mjose/Documentos/UOC_Master_Bioestadistica/10-TFM/06-Descriptors/05-Descriptors_DPPIV_final.csv",  index=False)




