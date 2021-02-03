# +
from time import time
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
path="../../data/"

from utils import *
from MLModels import *
from DataProcessing import *


config = dict(scale_pos_weight = 6,subsample = 1, min_child_weight = 3, max_depth = 5, gamma= 2, 
              colsample_bytree= 0.6,smote=1)

mod_xgb=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, learning_rate=0.1,
           max_delta_step=0,  missing=None,
           n_estimators=60, n_jobs=4, nthread=4, objective='binary:logistic',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=config['scale_pos_weight'],
           min_child_weight=config['min_child_weight'],
           gamma=config['gamma'], colsample_bytree=config['colsample_bytree'],max_depth=config['max_depth'],
           seed=42, silent=None, subsample=1, verbosity=1)

drpvars1='alanine_aminotransferase_f30620|aspartate_aminotransferase_f30650|alan_aspar_rat|Unknown|\
answered_sexual_history_questions|\
speciality_of_consultant|APOE4|\
hospital_episode_type|history_of_psychiatric|samesex|year_of_birth|age|sexual_partners|using_computer|sex_inference'

drpvarsapoe='AD|Unknown|answered_sexual_history_questions|speciality_of_consultant|APOE4|\
hospital_episode_type|history_of_psychiatric|samesex|year_of_birth|Genotype|age'

drvarsdef=['DIAB','DIAB_bef','CERVASC','CERVASC_bef','PD','CERVASC','EPIL','AD']

genos=[
 'Genotype_e1/e4',
 'Genotype_e2/e2',
 'Genotype_e2/e3',
 'Genotype_e2/e4',
 'Genotype_e3/e3',
 'Genotype_e3/e4',
 'Genotype_e4/e4']

genotypes=['Genotype_e1/e2',"Genotype_e2/e2","Genotype_e2/e3","Genotype_e3/e3","Genotype_e2/e4","Genotype_e1/e4",
          "Genotype_e3/e4","Genotype_e4/e4"]

gen_lkup={'Genotype_e2/e2':1,"Genotype_e1/e2":2,"Genotype_e2/e3":3,"Genotype_e3/e3":4,
          "Genotype_e2/e4":5,"Genotype_e1/e4":6,"Genotype_e3/e4":7,"Genotype_e4/e4":8}

gen_lkup_rev={1:'Genotype_e2/e2',2:"Genotype_e1/e2",3:"Genotype_e2/e3",4:"Genotype_e3/e3",
          5:"Genotype_e2/e4",6:"Genotype_e1/e4",7:"Genotype_e3/e4",8:"Genotype_e4/e4"}

def int_vars(df,var1,vars=genos):
    for var in vars:
        df[str(var1)+" "+var]=df[var]*df[var1]
    return df

def newfeats(df,shapsum,depvar='AD',feats=30):
    cols=[col for col in df if col in np.asarray(shapsum.head(feats)['column_name'])]
    
    if 'eid' in cols:
        df_out=df[np.append(['AD'],cols)]   
    else:
        df_out=df[np.append(['AD','eid'],cols)]
    return df_out

AD_model_full=pd.read_pickle('%s%s' % (path,'AD_model_full.p'))


def cleandf(df):
    df=replacenullsmean(df)
    df['alan_aspar_rat']=round((df['aspartate_aminotransferase_f30650']\
                                      /df['alanine_aminotransferase_f30620']),1)

    #change to any of the pollution metrics in the top 40%
    df['polluted']=0
    df['polluted'][(df['particulate_matter_air_pollution_pm25_absorbance_2010_f24007']>10)]=1


    df=col_spec_chars(df)
    #df=replace_genotype(df,gen_lkup,genotypes)
    return df

AD_model_full_mod=cleandf(AD_model_full)

dropvars_full=varstodrop(AD_model_full_mod,drpvars1,drvarsdef)
# -


drpvars_cv='CERVASCALL_bef|AD|aspartate_aminotransferase_f30650|alan_aspar_rat|Unknown|\
answered_sexual_history_questions|\
speciality_of_consultant|APOE4|STROKE|Stroke|\
hospital_episode_type|history_of_psychiatric|samesex|year_of_birth|age|sexual_partners|using_computer|sex_inference'
dropvars_CV=varstodrop(AD_model_full_mod,drpvars_cv,drvarsdef)

newmodelrun(AD_model_full_mod,dropvars=dropvars_CV,depvar='CERVASCALL_bef',reps=5,splits=3,model=mod_xgb,rebalance=1)

# +

mask_alan_aspar_rat=(AD_model_full_mod['alan_aspar_rat']<3)

AD_model_full_mod[mask_alan_aspar_rat].shape[0]/AD_model_full_mod.shape[0]

AD_model_full_mod[mask_alan_aspar_rat]['alan_aspar_rat'].hist(bins=50)


# +

model_fit_regr(AD_model_full_mod[mask_alan_aspar_rat],dropvars_full,'alan_aspar_rat',model=xgb_regr)

# -

model_fit_regr(AD_model_full_mod[mask_alan_aspar_rat],dropvars_full,'aspartate_aminotransferase_f30650',model=xgb_regr)


