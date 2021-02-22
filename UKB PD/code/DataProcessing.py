# +

'''
Complete model steps:
1. clean up dataset to produce X and y
2. Run several reps
3. Run cross fold validation
4. Produce AUC ROC curve of entire set of folds and reps
5. Produce SHAP scores averages and means for plots and also interaction scores
6. Deduce individual variable interactions

'''
import pandas as pd
import numpy as np

def mergewidedf(df1,df2,feats=['Age_Today']):
    df1_temp=pd.merge(df1[['eid']],df2[np.append(feats,'eid')], on='eid',how='left')
    df1[feats]=df1_temp[feats]
    return df1

def replace_genotype(df,gen_lkup,genotypes):
    df['Genotype']=0
    for col in df.columns:
        if col in genotypes and col!='Genotype':
            mask_col=(df[col]==1)
            df['Genotype'][mask_col]=gen_lkup[col]
            df.drop(columns=col,inplace=True)
    return df

