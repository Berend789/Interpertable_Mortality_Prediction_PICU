# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import pylab as plt
import pandas as pd
import numpy as np
import sklearn.impute as skin
from sqlalchemy import create_engine
import locale
import os

output_folder=r"C:\Users\berend\Documents\Python_Scripts\vrije_stage\Results_PICUR_PICEV2"
hour=72
locale.setlocale(locale.LC_ALL,'fr_FR')

from all_own_functions import cnfl,value_filtering
from All_standard_variables import conv_dict, dtype_dict,all_cols


lab_cols=['lab_bl_b2m', 'lab_bl_bil_d', 'lab_bl_bil_i', 'lab_bl_ca2', 'lab_bl_catot', 'lab_bl_cc', 
'lab_bl_cl', 'lab_bl_cr', 'lab_bl_CRP', 'lab_bl_f', 'lab_bl_gluc', 'lab_bl_hb', 'lab_bl_ht', 'lab_bl_k', 
'lab_bl_lactate', 'lab_bl_leuco', 'lab_bl_mg', 'lab_bl_na', 'lab_bl_tr', 'lab_bl_ur',
'lab_bg_be','lab_bg_hco3','lab_bg_pco2','lab_bg_ph','lab_bg_po2','lab_bg_sat']

pars_dates=['pat_bd','pat_datetime']

dfList=[]
reader=pd.read_csv(f"Total_database_{hour}hour.csv",delimiter=';',header=0,usecols=all_cols,converters=conv_dict,parse_dates=pars_dates,chunksize=(hour*60*100),low_memory=False)
for df in reader:

    import datetime as dt
    if 'pat_bd' in list(df.columns):
        df.replace(to_replace='NULL', value=np.nan,inplace=True)
        df=value_filtering(df)
        df['pat_datetime_temp']=pd.to_datetime(df['pat_datetime']).dt.date
        df['pat_bd']=pd.to_datetime(df['pat_bd']).dt.date
        df['Age'] = (df['pat_datetime_temp'] - df['pat_bd']).dt.days
        df['Age']=df['Age'].divide(365)
        df=df.drop('pat_datetime_temp',axis=1)
        df=df.drop('pat_datetime',axis=1)
        df=df.drop('pat_bd',axis=1)
    df['pat_hosp_id'] = df['pat_hosp_id'].astype('int')


    if any(df['Age'] > 12) == True:
        df.loc[(df['Age'] > 12),'Age']=50
        df.loc[(df['Age'] <= 12) & (df['Age'] > 4),'Age']=51
        df.loc[(df['Age'] > 1) & (df['Age'] <=4),'Age']=52
        df.loc[(df['Age'] > 0.5)  & (df['Age'] <=1),'Age']=53
        df.loc[(df['Age'] <= 0.5),'Age']=54
        df['Age'] = df['Age'].subtract(50)
        df['Age'] = df['Age'].astype('int32')






    df[lab_cols] = df[lab_cols].select_dtypes(include=float).astype(np.float32)

    from scipy.fftpack import fft,fftshift,fftfreq
    from scipy import signal
    import scipy as sp
    from all_own_functions import fft_feat
    from statsmodels.stats.descriptivestats import sign_test
    from tsfresh.feature_extraction.feature_calculators import abs_energy,number_cwt_peaks
    

    def lab_values_feature_building(dfl,columns,pat):
        """"df=pd.DataFrame(columns=[],index=dfl['pat_hosp_id'].unique())
        print(df.head())#data={'dummy':0},index=['a'])
        for pat, group in dfl.groupby('pat_hosp_id',sort=False):
            print('start_group')
            list_columns=[] """
        df=pd.DataFrame()
        group=dfl

        
        for colum in columns:
            if (group[colum].dtypes == float):
                temp=pd.DataFrame(group[colum].describe().to_numpy()[None],
                columns=[colum+'_count',colum+'_mean',colum+'_std',colum+'_min',colum+'_25%',colum+'_50%',colum+ '_75%',colum+'_max'],index=[pat])
                skew=pd.DataFrame(data=group[colum].skew(),columns=[(colum+'_skew')],index=[pat])
                kurt=pd.DataFrame(group[colum].kurtosis(),columns=[(colum+'_kurtosis')],index=[pat])
                a=sign_test(group[colum],mu0=group[colum].mean())
                high_mean=pd.DataFrame(data=a[0],columns=[(colum+'_count_above_mean')],index=[pat])
                high_median=pd.DataFrame(data=(sign_test(group[colum],mu0=group[colum].median()))[0],columns=[(colum+'_count_above_median')],index=[pat])
                ab_en=pd.DataFrame(data=abs_energy(group[colum]),columns=[(colum+'_absenergy')],index=[pat])
                cwt_peaks=pd.DataFrame(data=number_cwt_peaks(group[colum],10),columns=[(colum+'_number_CWT_peaks')],index=[pat])
                fft_aggr=fft_feat(group[colum],pat,colum)
                df=pd.concat([temp,df],axis=1)
                df=pd.concat([ab_en,df],axis=1)
                df=pd.concat([skew,df],axis=1)
                df=pd.concat([kurt,df],axis=1)
                df=pd.concat([high_mean,df],axis=1)
                df=pd.concat([high_median,df],axis=1)
                df=pd.concat([cwt_peaks,df],axis=1)
                df.append(fft_aggr)
            elif (group[colum].dtypes == np.float32):
                temp=pd.DataFrame(group[colum].describe().to_numpy()[None],
                columns=[colum+'_count',colum+'_mean',colum+'_std',colum+'_min',colum+'_25%',colum+'_50%',colum+ '_75%',colum+'_max'],index=[pat])
                l=[colum+'_count',colum+'_std',colum+'_max',colum+'_min']
                temp=temp[l]
                df=pd.concat([temp,df],axis=1)
            elif (group[colum].dtypes == object):

                temp=pd.DataFrame(group[colum].describe().to_numpy()[None],
                columns=[colum+'_count',colum+'_unique',colum+'_top',colum+'_freq'],index=[pat])
                
                df=pd.concat([temp,df],axis=1)
            else:
                temp=pd.DataFrame(group[colum].describe(datetime_is_numeric=True)['max'], 
                columns=[colum+'_max'],index=[pat])  #[colum+'_count',colum+'_mean',colum+'_min',colum+'_25%',colum+'_50%',colum+'_75%',colum+ '_max'],index=[pat])
                
                df=pd.concat([temp,df],axis=1)
            del temp        
        return df     
        
    def population_descriptives(df,columns):
        grouped = df.groupby('pat_hosp_id',sort=False)
        df1=grouped.get_group((list(grouped.groups)[0]))
        df1=lab_values_feature_building(df1,columns,df1['pat_hosp_id'].iloc[0])
        for pat,group in grouped:
            df_temp = lab_values_feature_building(group,columns,pat)
            if 'Death' in group['Status'].unique():
                df_temp['Label']='Death'
            else:
                df_temp['Label']='Alive'
            df1=df1.append(df_temp)
            del df_temp
        return df1



    
    column_list=(list(df.columns))
    column_list.remove('Status')
    column_list.remove('pat_hosp_id')
    df_try_full=population_descriptives(df,column_list)
    df_try_full = df_try_full[~df_try_full.index.duplicated()]
    df_try_full.loc[df_try_full.index[0],'Label']='Alive'
    print(df_try_full.info())
    print(df_try_full.head())
    print(df_try_full['Label'].unique())



    dfList.append(df_try_full)



import typing
class Wraptastic:
    def __init__(self, transform: typing.Callable):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df.values)
        return pd.DataFrame(data=transformed, columns=df.columns, index=df.index)

from sklearn.preprocessing import RobustScaler
ss=Wraptastic(RobustScaler())

df_coms=pd.concat(dfList,sort=False)
df_coms.to_csv(f'{hour}PICUREd.csv')
float_columns=list(df_coms.select_dtypes(include=['float64','float32','int32']).columns)


df_temp=df_coms[float_columns]
df_temp=(df_temp.groupby('Age_max',sort=False).apply(ss).drop('Age_max',axis="columns"))
float_columns.remove('Age_max')
df_coms[float_columns]=df_temp
print(df_coms.info())
df_coms.to_csv(f'{hour}PICURED_PICE.csv')

conv_pice={'BaseExcess':cnfl,'FiO2':cnfl,'PaO2':cnfl,'PIM2Score':cnfl,'PIM2Mort':cnfl,'PIM3Score':cnfl,
'PIM3Mort':cnfl,'GlucoseMax':cnfl,'PaO2Min':cnfl,'PCO2Max':cnfl,'PHMax':cnfl,'PHMin':cnfl,'PotassiumMax':cnfl,'PTMax':cnfl,'TemperatureMax':cnfl,'TemperatureMin':cnfl,
'BicarbonateMin':cnfl,'BicarbonateMax':cnfl,'UreaMax':cnfl,'WhiteBloodCountMin':cnfl,'PRISM4Mortality':cnfl,'Age(days)':cnfl}
df_pice=pd.read_csv('scores.csv',converters=conv_pice,header=0,delimiter=';')
df_pice.drop(df_pice.tail(1).index,inplace=True)
df_pice.dropna(axis=0,how='all',inplace=True)
del df_pice['AdmissionDate']
index_names = df_pice[df_pice['HospitalNumber'] == 'RPH'].index
df_pice.drop(index_names, inplace=True)
a=df_pice[df_pice['HospitalNumber'].duplicated(keep=False)]
a=a[a['HospitalNumber'].duplicated(keep='first')]
duplicate_patients=a['HospitalNumber'].tolist()
df_pice=df_pice.set_index('HospitalNumber')
df_pice=df_pice.sort_index()
pimLowRisk = ["Asthma","Bronchiolitis","Croup","ObstructiveSleepApnea","DiabeticKetoacidosis",'SeizureDisorder']
pimHighRisk = ["CerebralHemorrhage","CardiomyopathyOrMyocarditis","HIVPositive","HypoplasticLeftHeartSyndrome","NeurodegenerativeDisorder","NecrotizingEnterocolitis"]
pimVeryHighRisk = ["CardiacArrestInHospital","CardiacArrestPreHospital","SevereCombinedImmuneDeficiency","LeukemiaorLymphoma","BoneMarrowTransplant","LiverFailure"]

df_pice['RiskDiagnoses'] = (df_pice['RiskDiagnoses'].fillna(value='Unknown'))

for i,row in df_pice.iterrows():
    try:
        if any(substring in df_pice.loc[i,'RiskDiagnoses'] for substring in pimVeryHighRisk) == True:
            df_pice.loc[i,'RiskDiagnoses'] = 3
        elif any(substring in df_pice.loc[i,'RiskDiagnoses'] for substring in pimHighRisk)==True :
            df_pice.loc[i,'RiskDiagnoses']= 2
        elif any(substring in df_pice.loc[i,'RiskDiagnoses'] for substring in pimLowRisk)==True:
            df_pice.loc[i,'RiskDiagnoses'] = 1
        else: df_pice.loc[i,'RiskDiagnoses'] = 0
    except TypeError:
        continue
if 'Status' in list(df_pice.columns):
    del df_pice['Status']
print(df_pice.info())



df_pice=df_pice[~df_pice.index.duplicated(keep='last')]
scaler=RobustScaler()
float_columns_pice = list(df_pice.select_dtypes(include=['float64']).columns)

df_pice_temp=df_pice[float_columns_pice]
df_pice_scaled=pd.DataFrame(scaler.fit_transform(df_pice_temp), columns=float_columns_pice,index=df_pice_temp.index)
print(df_pice_scaled.info())
df_pice[float_columns_pice]=df_pice_scaled



    
from sklearn.preprocessing import LabelBinarizer
from collections import Counter 
lb=LabelBinarizer()
df_coms['Label']=lb.fit_transform(df_coms['Label'])
print(df_coms['Label'].unique())
a=list(df_coms.columns)
a.remove('Label')
for column in a:
    if 'top' in column:
        df_coms[column]=df_coms[column].astype('object')
    elif 'count' in column:
        df_coms[column]=df_coms[column].fillna(value=0)
        df_coms[column]=df_coms[column].astype('int8')
    else:
        df_coms[column]=df_coms[column].astype('float64')

df_com=df_coms.join(df_pice)
df_com = df_com.reset_index()

for i,row in df_com.iterrows():
    if  df_com.index[i] in duplicate_patients:
        df_com.loc[i,'ReHospitalisation'] = 1
    else:
        df_com.loc[i,'ReHospitalisation'] = 0

x=df_com.drop(['index','Label'],axis=1)
y=df_com['Label']

print(x.info())

print(x.head())
print(y.value_counts())


import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


c=x.corrwith(y,method='pearson')
d=x.corrwith(y,method='spearman')
relevant_features_pe = c[c>0.05]
relevant_features_sp = d[d>0.05]
print(relevant_features_pe)
x_pe=x[relevant_features_pe.index]
x_sp=x[relevant_features_sp.index]




from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC,SVR,LinearSVC
from sklearn.metrics import classification_report, confusion_matrix,average_precision_score,f1_score,roc_curve,roc_auc_score,plot_confusion_matrix
from sklearn.calibration import calibration_curve
import sklearn.metrics as metrics
from all_own_functions import f_importances
import os


def machine_learning_function(x_train,x_test,y_train,y_test,model,wrapper=0):
    float_columns=list(x_train.select_dtypes(include=['float64']).columns)
    int_columns=list(x_train.select_dtypes(include=['int32']).columns)
    cat_list=list(x_train.select_dtypes(include=['object']).columns)
    
    if 'Label' in float_columns: float_columns.remove('Label')

    float_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent'))])
    int_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='constant',fill_value=0))])
    cat_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='constant',fill_value='Unknown')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocess = ColumnTransformer(transformers=[('float',float_transformer,float_columns),('int',int_transformer,int_columns),('cat',cat_transformer,cat_list)],                       remainder='passthrough')

    if wrapper == 1:
        clf = Pipeline(steps=[('preprocessor',preprocess),('Feature_selection',SelectFromModel(LinearSVC(max_iter=5000),max_features=20)),('classifier',(model))])
    elif wrapper == 2:
        clf = Pipeline(steps=[('preprocessor',preprocess),('Feature_selection',SelectFromModel(DecisionTreeClassifier(),max_features=20)),('classifier',(model))])
    elif wrapper == 3:
        clf = Pipeline(steps=[('preprocessor',preprocess),('Feature_selection',SelectFromModel(LogisticRegression(),max_features=20)),('classifier',(model))])
    else:
        clf = Pipeline(steps=[('preprocessor',preprocess),('classifier',model)])

    clf=clf.fit(x_train,y_train)
    y_pred_clas=clf.predict(x_test)
    # Predict the probabilities, function depends on used classifier

    try:
        y_pred_prob=clf.predict_proba(x_test)
        y_pred_prob=y_pred_prob[:,1]
    except:
        try:
            y_pred_prob=clf.decision_function(x_test)
        except:
            y_pred_prob=y_pred_clas
    
    # failsafe to inpute NaN probabilities with 0
    inds = np.where(np.isnan(y_pred_prob))
    if inds:
        y_pred_prob=np.nan_to_num(y_pred_prob, nan=0)
    return clf, y_pred_clas,y_pred_prob, y_test,x_test


from sklearn.metrics import roc_auc_score
from math import sqrt

def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score,brier_score_loss
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# speciify colours for plot later on
colours=['k-', 'g--', 'r:', 'c-.', 'm-+', 'y-*', 'k-o']

# Stratiefied fold for cros-validation
fold=StratifiedKFold(3)

# Names and classiefiers to be used in the loop
names = [ "Random Forest", "Sigmoid SVM"]

classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,class_weight={0:0.1,1:0.9}),
    SVC(kernel='sigmoid',probability=True,class_weight={0:0.1,1:0.9}),
    ]

# Names and dataframes used in the loop
data_names=["All_data",'Pearson_Cor',"Wrap_Lin_SVM","Wrap_Dec_Tree"]
x_data=[x,x_pe, x, x]
auc_pim=0
# split data in train and test data

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1 / 5,random_state=1)

# calculate ROC and AUC outside of loop
fpr_pim, tpr_pim, _ = roc_curve(y_test,  x_test['PIM3Score'])
auc_pim = roc_auc_score(y_test, x_test['PIM3Score'])

# check if result files do not already exist, updating pdf is not possible
try:
    pdf = PdfPages(os.path.join(output_folder,f"Figures of {hour} hour results.pdf"))
    pdf_ROC = PdfPages(os.path.join(output_folder,f"ROC of {hour} hour results.pdf"))
    pdf_cal = PdfPages(os.path.join(output_folder,f"Cal of {hour} hour results.pdf"))
    pdf_hist = PdfPages(os.path.join(output_folder,f"Hist of {hour} hour results.pdf"))
except PermissionError:
    os.remove(os.path.join(output_folder,f"Figures of {hour} hour results.pdf"))
    os.remove(os.path.join(output_folder,f"ROC of {hour} hour results.pdf"))
    os.remove(os.path.join(output_folder,f"Results_{hour}hours_scores.txt"))
    os.remove(os.path.join(output_folder,f"Cal of {hour} hour results.pdf"))
    os.remove(os.path.join(output_folder,f"Hist of {hour} hour results.pdf"))
    pdf = PdfPages(os.path.join(output_folder,f"Figures of {hour} hour results.pdf"))
    pdf_ROC= PdfPages(os.path.join(output_folder,f"ROC of {hour} hour results.pdf"))
    pdf_cal = PdfPages(os.path.join(output_folder,f"Cal of {hour} hour results.pdf"))
    pdf_hist = PdfPages(os.path.join(output_folder,f"Hist {hour}  hour results.pdf"))

# loop over different data and feature selection techniques
for data_name, xd in zip(data_names, x_data):
    # transform columns based on dataframe used
    columns = xd.columns.tolist()
    x_train_d = x_train[columns]
    x_test_d = x_test[columns]
    
    # Add variable for wrapper based feature selection
    if 'Wrap' in data_name:
        wrapper += 1
    else:
        wrapper=0
    
    # create variables for temp storage lateron
    temp_fpr=dict()
    temp_tpr=dict()
    temp_auc=dict()
    temp_probtrue={}
    temp_probpred={}
    temp_score={}

    # loop over different classifiers
    for name, clf_s in zip(names, classifiers):

        clf,y_pred_clas,y_pred_prob,y_test,x_test_d = machine_learning_function(x_train_d,x_test_d,y_train,y_test,clf_s,wrapper)

        # calculate scoring metrics
        report=classification_report(y_test,y_pred_clas,target_names=['Alive','Death'])
        score=clf.score(x_test_d,y_test)
        average_precision = average_precision_score(y_test, y_pred_prob)
        f1_s=f1_score(y_test, y_pred_clas)

        # write scoring metrics to file
        with open(os.path.join(output_folder,f"Results_{hour}hours_scores.txt"),'a') as file:
            file.write(f"{data_name} with {name} Results for {hour} hours \n\n")
            file.write(f"Classification report \n {report} \n")
            file.write(f"Hold_out_scores {score} \n")
            file.write(f"Average precision score {average_precision} \n")
            file.write(f"F1 score {f1_s} \n\n\n")
        
        # plot confusion matrix
        plot_confusion_matrix(clf,x_test_d,y_test)
        plt.title(f"{data_name} with {name} classifier, Results from {hour} hour data")
        fig=plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)

        # plot ROC with AUC
        fpr, tpr, _ = roc_curve(y_test,  y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr,tpr,label=f"{name}, auc={auc}")
        plt.plot(fpr_pim,tpr_pim,label=f"PIM3Score, auc={auc_pim}")
        plt.title(f"{data_name} with {name} classifier, Results from {hour} hour data")
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig=plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)

        # temporarly store ROC and AUC per classifier
        temp_fpr.update({f'{name}': fpr})
        temp_tpr.update({f'{name}': tpr})
        temp_auc.update({f'{name}': auc})

         # plot callibration plot with brier_loss score
        
        prob_true, prob_pred = calibration_curve(y_test,  y_pred_prob,normalize=True)
        plt.plot(prob_pred,prob_true,label=f"{name} ")
        plt.title(f"{data_name} with {name} classifier, Results from {hour} hour data")
        plt.legend(loc=4)
        plt.ylabel('Fraction of Positives')
        fig=plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)

        # temporarly store ROC and AUC per classifier
        temp_probtrue.update({f'{name}': prob_true})
        temp_probpred.update({f'{name}': prob_pred})
        #temp_score.update({f'{name}':clf_score})
    
        print(f"{data_name} with {name} classifier, results from {hour} hour data")
        print("Original ROC area: {:0.3f}".format(roc_auc_score(y_test, y_pred_prob)))
        print("Original ROC area, PIM3: {:0.3f}".format(roc_auc_score(y_test, x_test['PIM3Score'])))

        n_bootstraps = 2000
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred_prob), len(y_pred_prob))
            if len(np.unique(y_test.iloc[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(y_test.iloc[indices], y_pred_prob[indices])
            bootstrapped_scores.append(score)
            #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
            confidence_lower, confidence_upper))
        a=roc_auc_ci(y_test,y_pred_prob)
        print(a)


        
    #PLot every roc from the used classifier per feature selection method 
    a=0
    for k,v in temp_fpr.items():
        plt.plot(v,temp_tpr.get(k),colours[a],label=f"{k}, auc={temp_auc.get(k)}",linewidth=1.5,markersize=1)
        a= a+1
    plt.plot(fpr_pim,tpr_pim,'b->',label=f"PIM3Score, auc={auc_pim}",linewidth=1.5,markersize=1)
    plt.legend(loc=4,fontsize='xx-small')
    plt.title(f'{data_name} ROC from {hour} hour data')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    fig=plt.gcf()
    pdf_ROC.savefig(fig)
    plt.close(fig)
    count=0

    #PLot calibration from the used classifier per feature selection method
    for k,v in temp_probtrue.items():
        plt.plot(temp_probpred.get(k),v,colours[count],label=f"{k}",linewidth=1.5,markersize=1)
        count += 1
    plt.legend(loc=4,fontsize='xx-small')
    plt.title(f'{data_name} Calibration Curve from {hour} hour data')
    plt.xlabel('Fraction of positives')
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    fig=plt.gcf()
    pdf_cal.savefig(fig)
    plt.close(fig)
    count=0

    # plot histogram from the used classifier per feature selection method
    for k,v in temp_probtrue.items():
        plt.hist(temp_probpred.get(k),range=(0,1),label=f"{k}",histtype='step',lw=2)
        count += 1
    plt.legend(loc=4,fontsize='xx-small')
    plt.title(f'{data_name} Calibration Curve from {hour} hour data')
    plt.xlabel('Fraction of positives')
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    fig=plt.gcf()
    pdf_hist.savefig(fig)
    plt.close(fig)


    del temp_fpr
    del temp_tpr
    del temp_auc
pdf.close()
pdf_ROC.close()
pdf_cal.close()
pdf_hist.close()



from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages
# Same functions as before except now specified for Regression algorithms

names = ["Logistic Regression"
        ]

classifiers = [
    LogisticRegression(),
    ]

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1 / 5)

try:
    pdf = PdfPages(os.path.join(output_folder,f"Figures of {hour} hour results_reg.pdf"))
    pdf_ROC= PdfPages(os.path.join(output_folder,f"ROC of {hour} hour results_regression.pdf"))
except PermissionError:
    os.remove(os.path.join(output_folder,f"ROC of {hour} hour results_regression.pdf"))
    os.remove(os.path.join(output_folder,f"Figures of {hour} hour results_reg.pdf"))
    os.remove(os.path.join(output_folder,f"Results_{hour}hours_scores_reg.txt"))
    pdf_ROC= PdfPages(os.path.join(output_folder,f"ROC of {hour} hour results_regression.pdf"))
    pdf = PdfPages(os.path.join(output_folder,f"Figures of {hour} hour results_reg.pdf"))
fpr_pim, tpr_pim, _ = roc_curve(y_test,  x_test['PIM3Score'])
auc_pim = roc_auc_score(y_test, x_test['PIM3Score'])


#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1 / 5,random_state=1)

temp_fpr=dict()
temp_tpr=dict()
temp_auc=dict()
for data_name, xd in zip(data_names, x_data):
    # transform columns based on dataframe used
    columns = xd.columns.tolist()
    x_train_d = x_train[columns]
    x_test_d = x_test[columns]
    
    # Add variable for wrapper based feature selection
    if 'Wrap' in data_name:
        wrapper += 1
    else:
        wrapper=0
    
    # create variables for temp storage lateron
    temp_fpr=dict()
    temp_tpr=dict()
    temp_auc=dict()
    temp_probtrue={}
    temp_probpred={}
    temp_score={}

    for name, clf_s in zip(names, classifiers):

        clf,y_pred_clas,y_pred_prob,y_test,x_test_d= machine_learning_function(x_train_d,x_test_d,y_train,y_test,clf_s,wrapper)

        # calculate scoring metrics
        report=classification_report(y_test,y_pred_clas,target_names=['Alive','Death'])
        score=clf.score(x_test_d,y_test)
        average_precision = average_precision_score(y_test, y_pred_prob)
        f1_s=f1_score(y_test, y_pred_clas)

        # write scoring metrics to file
        with open(os.path.join(output_folder,f"Results_{hour}hours_scores_reg.txt"),'a') as file:
            file.write(f"{data_name} with {name} Results for {hour} hours \n\n")
            file.write(f"Classification report \n {report} \n")
            file.write(f"Hold_out_scores {score} \n")
            file.write(f"Average precision score {average_precision} \n")
            file.write(f"F1 score {f1_s} \n\n\n")

        plot_confusion_matrix(clf,x_test_d,y_test)
        plt.title(f"{data_name} with {name} classifier, Results from {hour} hour data")
        fig=plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test,  y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)


        temp_fpr.update({f'{name}': fpr})
        temp_tpr.update({f'{name}': tpr})
        temp_auc.update({f'{name}': auc})
        print(f"{data_name} with {name} classifier, results from {hour} hour data")
        print("Original ROC area: {:0.3f}".format(roc_auc_score(y_test, y_pred_prob)))
        print("Original ROC area, PIM3: {:0.3f}".format(roc_auc_score(y_test, x_test['PIM3Score'])))
        n_bootstraps = 2000
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred_prob), len(y_pred_prob))
            if len(np.unique(y_test.iloc[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(y_test.iloc[indices], y_pred_prob[indices])
            bootstrapped_scores.append(score)
            #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
            confidence_lower, confidence_upper))
        a=roc_auc_ci(y_test,y_pred_prob)
        print(a)
    a=0
    for k,v in temp_fpr.items():
        plt.plot(v,temp_tpr.get(k),colours[a],label=f"{k}, auc={temp_auc.get(k)}",linewidth=1.5,markersize=1)
        a= a+1
    plt.plot(fpr_pim,tpr_pim,'b->',label=f"PIM3Score, auc={auc_pim}",linewidth=1.5,markersize=1)
    plt.legend(loc=4,fontsize='xx-small')
    plt.title(f'{name} ROC of {hour} hour data')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    fig=plt.gcf()
    pdf_ROC.savefig(fig)
    plt.close(fig)

    del temp_fpr
    del temp_tpr
    del temp_auc
pdf_ROC.close()
pdf.close()
        