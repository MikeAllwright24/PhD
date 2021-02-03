# +
import re
import pandas as pd
import numpy as np

def findcols(df,text):
    return [col for col in df.columns if text in col]

def findcols_com(strs,df):
    selcols=[col for col in df.columns if re.search(strs,col)]
    return selcols

def varstodrop(df,strs,defvars):
    selcols=[col for col in df.columns if re.search(strs,col)]
    selcols=np.append(selcols,defvars)
    return selcols

def col_spec_chars(df):
    df.columns=df.columns.str.replace(',','_')
    df.columns=df.columns.str.replace('<','_')
    df.columns=df.columns.str.replace('>','_')
    df.columns=df.columns.str.replace('[','_')
    df.columns=df.columns.str.replace(']','_')
    return df
# -


