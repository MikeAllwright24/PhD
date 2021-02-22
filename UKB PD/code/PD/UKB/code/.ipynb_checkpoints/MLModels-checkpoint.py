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
from xgboost import XGBClassifier

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
import shap
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

config = dict(scale_pos_weight = 1,max_depth=3,smote=1)
#preprocessing

def replacenullsmean(df):
    for col in df.columns:
        nullmask=(df[col].isna())
        df[col][nullmask]=df[col][~nullmask].mean()
    return df

# cross fold validation

mod_xgb=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=config['max_depth'], min_child_weight=1, missing=None,
           n_estimators=60, n_jobs=4, nthread=4, objective='binary:logistic',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=config['scale_pos_weight'],
           seed=42, silent=None, subsample=1, verbosity=1)

def predvars(model,yscore_tot,ypred_tot,ytest_tot,eids,aucs,y_test,X_test,base_fpr,tprs,e_test):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)
    yscore_tot=np.append(yscore_tot,y_score[:, 1])
    ypred_tot=np.append(ypred_tot,y_pred)
    ytest_tot=np.append(ytest_tot,y_test)
    eids=np.append(eids,e_test)
    return aucs,tprs,yscore_tot,ypred_tot,ytest_tot,eids

# AUC ROC curves

def kfoldvalidation(X,y,model,splits=5):
    models_out=[]
    for train_index, test_index in kf.split(X1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_out=model.fit(X_train,y_train)
        #models_out.update(model_out)
    return models_out


def plot_ROCAUC(tprs, base_fpr,aucs,ytest_tot,ypred_tot,yscore_tot,title='ROC AUC curve'):
    
    colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet'] 
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)
    
    print ("Accuracy : %.4g" % metrics.accuracy_score(ytest_tot, ypred_tot))
    print(sum(ypred_tot))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(ytest_tot, yscore_tot))

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    
    plt.figure(figsize=(12, 8))
    plt.plot(base_fpr, mean_tprs, 'b', alpha = 0.8,
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.title(title)
    #plt.axes().set_aspect('equal', 'datalim')
    plt.show()

    
    
def newmodelrun(df,dropvars,model=mod_xgb,depvar='AD',reps=1,splits=2,shap=1,
                rebalance=0,max_display=15,plot_type="dot",plots=1,modtype='class'):
    
    df_cop=df.copy()
    X_df=df.drop(columns=dropvars)
    X = np.asarray(X_df)
    y = np.asarray(df[depvar])
    tprs = []
    aucs = []
    yscore_tot=np.array([])
    ypred_tot=np.array([])
    ytest_tot=np.array([])
    
    eids=np.array([])
    iteration=0
    importance_df_full=pd.DataFrame()
    
    list_shap_values = list([])
    list_test_sets = list([])   
    base_fpr = np.linspace(0, 1, 101)
      
    e=df['eid']
   
    for reps in range(reps):
        
        print("iteration: "+str(reps))
        
        if rebalance==1:
   
            if df_cop.shape[0]*2/3<len(eids):
                eids=np.asarray([])
                
            dfY=df_cop[(df_cop[depvar]==1)]
            dfN=df_cop[(df_cop[depvar]==0)&(~df_cop['eid'].isin(eids))].sample(dfY.shape[0])
            df=pd.concat([dfY,dfN],axis=0)
            
            e=df['eid']
            X_df=df.drop(columns=dropvars)
            X = np.asarray(X_df)
            y = np.asarray(df[depvar])
            
        
        kf = KFold(n_splits=splits,shuffle=True)
        for train_index, test_index in kf.split(X):
            
            #print(len(e))
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            e_test=e[test_index]
            model.fit(X_train,y_train)

            aucs,tprs,yscore_tot,ypred_tot,ytest_tot,eids=\
            predvars(model,yscore_tot,ypred_tot,ytest_tot,eids,aucs,y_test,X_test,base_fpr,tprs,e_test)
            
            if shap==1:
                list_shap_values,list_test_sets,importance_df_full=\
                shapmeasures(model,X_df,X_test,list_shap_values,list_test_sets,importance_df_full,iteration,test_index)

            iteration=iteration+1
        dfscore=pd.DataFrame({'eid':eids,'score':yscore_tot,'actual':ytest_tot,'pred':ypred_tot})
        dfscore=pd.DataFrame(dfscore.groupby(['eid']).agg({'score':['mean','std'],'actual':'max','pred':'max'})).reset_index()
        dfscore.columns=['eid','mean_score','std_score','actual','predicted']
    
    if plots==1:
        plot_ROCAUC(tprs,base_fpr,aucs,ytest_tot,ypred_tot,yscore_tot)
    
    if shap==1:
        shapsum=shap_plots_vals(list_test_sets,list_shap_values,X,X_df,y,importance_df_full,
                                max_display,plot_type,plots=plots)
        return shapsum,dfscore
    
    else:
        return dfscore

#SHAP

def shapmeasures(model,X_df,X_test,list_shap_values,list_test_sets,importance_df_full,iteration,test_index):
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(X_test)
    
    #interaction values
    
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_df.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df['shap_importance']=pd.to_numeric(importance_df['shap_importance'])
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    importance_df['iteration']=iteration
    importance_df['rank']=np.arange(len(importance_df))
    list_shap_values.append(shap_values)
    list_test_sets.append(test_index)
    importance_df_full=pd.concat([importance_df_full,importance_df],axis=0)
    
    return list_shap_values,list_test_sets,importance_df_full

def ABS_SHAP(df_shap,df,numfeats=10):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = False)
    k2=k2.head(numfeats)
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    plt.show()
    


def shap_plots_vals(list_test_sets,list_shap_values,X,X_df,y,importance_df_full,max_display=30,plot_type='dot',plots=1):
    test_set = list_test_sets[0]
    shap_values = np.array(list_shap_values[0])
    for i in range(0,len(list_test_sets)):#maybe put -1 here to remove last one
        test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
        shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)
    #bringing back variable names

   
    X_test = pd.DataFrame(X[test_set],columns=X_df.columns)
    
    if plots==1:
        print("SHAP summary dot plot for selected feature number")
        shap.summary_plot(shap_values, X_test,max_display=max_display,plot_type=plot_type)
        print("SHAP summary dot plot for 35 features")
        shap.summary_plot(shap_values, X_test,max_display=35,plot_type=plot_type)
        #print("SHAP summary bar plot for selected feature number")
        #ABS_SHAP(shap_values,X_test,numfeats=max_display)
        #print("SHAP summary dot plot for 35 features")
        #ABS_SHAP(shap_values,X_test,numfeats=35)
        
    
    #shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(pd.DataFrame(X_test.iloc[:2000,:],columns=X.columns))

    #shap.dependence_plot("Age_Today", shap_values, X_test)
    y_test_out=y[test_set]
    #y_pred_prob=y1[test_set]

    shapsum=pd.DataFrame(importance_df_full.groupby(['column_name']).
                 agg({'shap_importance':['mean','std'],'rank': ['min','max','mean','median','std']})).reset_index()
    shapsum.columns=['column_name','mean_shap_imp','std_shap_imp','min_rank','max_rank','mean_rank',
                     'median_rank','std_rank']

    shapsum.sort_values(by='mean_shap_imp',ascending=False,inplace=True)
    
    if plots!=1:
        print(shapsum[['column_name','mean_shap_imp']].head(20))
    
    return shapsum

def createpdpplots(df,colsel,explainer,numfeats=10):

    shap_interaction_values = explainer.shap_interaction_values(df)

    tmp = np.abs(shap_interaction_values).mean(0)
    for i in range(tmp.shape[0]):
        tmp[i,i] = 0
    inds = np.argsort(-tmp.mean(0))[:numfeats]
    for col in df.columns[inds]:
        for col1 in df.columns[inds]:
            if col!=col1 and col1==colsel:
                shap.dependence_plot((col,col1), shap_interaction_values, df,
                                            interaction_index=None,show=False)
                plt.show()
                


    
xgb_regr=xgb.XGBRegressor(random_state=42, base_score=0.5)       
def model_fit_regr(df,dropvars,depvar,model=xgb_regr):
    X=df.drop(columns=dropvars)
    #setting y as the target variable
    y = df[depvar]
   
    #Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 100)
    model = model
    model.fit(X_train,y_train)
    
    print(model.score(X_train, y_train)) 
    print(model.score(X_test, y_test) )
    print(model.score(X, y) )
       
    y_pred = model.predict(X_test)
    
    df=pd.DataFrame({'actual':y_test,'predicted':y_pred}).sort_values(by='actual').reset_index()
    df['index']=np.arange(df.shape[0])
    df.groupby('index')['actual','predicted'].mean().plot(kind='line')
    plt.show()
    
    #rsq=r2_score(y_pred, y_test) 

    #rsq2=r2_score(y_pred, y_test, multioutput='variance_weighted')
    
    exp = shap.TreeExplainer(model)
    shap_values = exp.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    plt.show()

    return model,y_test,y_pred

# +
# complete model steps

                

