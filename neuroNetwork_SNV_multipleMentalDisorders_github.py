import sys,os,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import f_regression
from sklearn.metrics import classification_report,confusion_matrix,hamming_loss,precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import multilabel_confusion_matrix
import statistics

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

muList=[]
f=open('mutation_class_SNV.txt')
line=f.readline().strip()
while line!='':
    muList.append(line)
    line=f.readline().strip()
f.close()

for theMu in muList:
    print(theMu)
    f_w=open('featureVectors/SNV_5M_MAF_processed/'+theMu+'_predict.txt','w')
    f_w.write('theRound'+'\t'+'Accuracy'+'\n')

    theAc=[]

    theFile=0
    while theFile<50:
       
       data_Y=pd.read_csv('featureVectors/SNV_5M_MAF_processed/training/'+theMu+'_pheno_'+str(theFile)+'.txt',sep='\t',header=0)
       data_X=pd.read_csv('featureVectors/SNV_5M_MAF_processed/training/'+theMu+'_'+str(theFile)+'.txt',sep='\t',header=0)
       
       f=open('featureVectors/SNV_5M_MAF_processed/training/'+theMu+'_pheno_'+str(theFile)+'.txt')
       line=f.readline().strip()
       temp=re.split('\t+',line)
       tempList=[]
       i=1
       while i<len(temp):
          tempList.append(temp[i])
          i=i+1
       f.close()
       Y=data_Y[tempList]

       f=open('featureVectors/SNV_5M_MAF_processed/training/'+theMu+'_'+str(theFile)+'.txt')
       line=f.readline().strip()
       temp=re.split('\t+',line)
       tempList=[]
       i=1
       while i<len(temp)-1:
          tempList.append(temp[i])
          i=i+1
       f.close()
       X=data_X[tempList]

       if theMu=='nonsynonymousSNV':
          tupList=[]
          num_layer=20
          #num_neuro=73
          num_neuro=25
          i=0
          while i<num_layer:
             tupList.append(num_neuro)
             i=i+1
          tupList=tuple(tupList)
          clf = MLPClassifier(random_state=0, max_iter=1000, alpha=0.05, learning_rate_init=0.0001, activation='tanh', solver='adam', learning_rate = 'constant', hidden_layer_sizes=tupList)
          
       if theMu=='frameshift':
          tupList=[]
          num_layer=20
          #num_neuro=25
          num_neuro=100
          i=0
          while i<num_layer:
             tupList.append(num_neuro)
             i=i+1
          tupList=tuple(tupList)
          clf = MLPClassifier(random_state=0, max_iter=1000, alpha=0.0001, learning_rate_init=0.0001, activation='tanh', solver='adam', learning_rate = 'constant', hidden_layer_sizes=tupList)
          
       if theMu=='streamUTR':
          tupList=[]
          num_layer=20
          #num_neuro=75
          num_neuro=25
          i=0
          while i<num_layer:
             tupList.append(num_neuro)
             i=i+1
          tupList=tuple(tupList)
          clf = MLPClassifier(random_state=0, max_iter=1000, alpha=0.0001, learning_rate_init=0.0001, activation='tanh', solver='adam', learning_rate = 'constant', hidden_layer_sizes=tupList)

       if theMu=='ncRNA':
          tupList=[]
          num_layer=20
          num_neuro=25
          i=0
          while i<num_layer:
             tupList.append(num_neuro)
             i=i+1
          tupList=tuple(tupList)
          clf = MLPClassifier(random_state=0, max_iter=1000, alpha=0.0001, learning_rate_init=0.0001, activation='tanh', solver='adam', learning_rate = 'constant', hidden_layer_sizes=tupList) 

       if theMu=='intronicSNV':
          tupList=[]
          num_layer=20
          #num_neuro=75
          num_neuro=25
          i=0
          while i<num_layer:
             tupList.append(num_neuro)
             i=i+1
          tupList=tuple(tupList)
          clf = MLPClassifier(random_state=0, max_iter=1000, alpha=0.0001, learning_rate_init=0.0001, activation='tanh', solver='adam', learning_rate = 'constant', hidden_layer_sizes=tupList)
          
       if theMu=='intergenicSNV':
          tupList=[]
          num_layer=20
          #num_neuro=25
          num_neuro=100
          i=0
          while i<num_layer:
             tupList.append(num_neuro)
             i=i+1
          tupList=tuple(tupList)
          clf = MLPClassifier(random_state=0, max_iter=1000, alpha=0.0001, learning_rate_init=0.0001, activation='tanh', solver='lbfgs', learning_rate = 'constant', hidden_layer_sizes=tupList)

       if theMu=='stop':
          tupList=[]
          num_layer=20
          num_neuro=100
          i=0
          while i<num_layer:
             tupList.append(num_neuro)
             i=i+1
          tupList=tuple(tupList)          
          clf = MLPClassifier(random_state=0, max_iter=1000, alpha=0.0001, learning_rate_init=0.0001, activation='tanh', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=tupList)  
   
       scaler = StandardScaler()
       scaler.fit(X)
       X = scaler.transform(X)
       clf.fit(X,Y)

       #testing
       data_Y=pd.read_csv('featureVectors/SNV_5M_MAF_processed/testing/'+theMu+'_pheno_'+str(theFile)+'.txt',sep='\t',header=0)
       data_X=pd.read_csv('featureVectors/SNV_5M_MAF_processed/testing/'+theMu+'_'+str(theFile)+'.txt',sep='\t',header=0)
       f=open('featureVectors/SNV_5M_MAF_processed/testing/'+theMu+'_pheno_'+str(theFile)+'.txt')
       line=f.readline().strip()
       temp=re.split('\t+',line)
       tempList=[]
       i=1
       while i<len(temp):
          tempList.append(temp[i])
          i=i+1
       f.close()
       Y_test=data_Y[tempList]

       f=open('featureVectors/SNV_5M_MAF_processed/testing/'+theMu+'_'+str(theFile)+'.txt')
       line=f.readline().strip()
       temp=re.split('\t+',line)
       tempList=[]
       i=1
       while i<len(temp)-1:
          tempList.append(temp[i])
          i=i+1
       f.close()
       X_test=data_X[tempList]
       X_test = scaler.transform(X_test)

       y_pred= clf.predict(X_test)
       theScore=clf.score(X_test,Y_test)
       print(metrics.accuracy_score(Y_test, y_pred))
       print(hamming_loss(Y_test, y_pred))
       #print(multilabel_confusion_matrix(Y_test,y_pred))
       print(classification_report(Y_test,y_pred))
       f_w.write(str(theFile)+'\t'+str(theScore)+'\t'+str(hamming_loss(Y_test, y_pred))+'\n')

       #target_names = ['ADHD','developmental_speech_language_disorder','development_delays','depressive_disorder','Anxiety','Oppositional_defiant_disorder','ASD','intellectual_disabilities']
       #print(classification_report(Y_test,y_pred,target_names=target_names))
       f_w_1=open('featureVectors/SNV_5M_MAF_processed/testing/'+theMu+'_'+str(theFile)+'_predict.txt','w')
       #f_w_1.write(multilabel_confusion_matrix(Y_test,y_pred)+'\n')
       f_w_1.write(classification_report(Y_test,y_pred)+'\n')
       f_w_1.close()

       #print(str(theFile)+" Accuracy:",metrics.accuracy_score(Y_test, y_pred))
       #theAc.append(float(metrics.accuracy_score(Y_test, y_pred)))
       #f_w.write(str(theFile)+'\t'+str(y_pred)+'\n')
       #for thePredict in y_pred:
       #   f_w.write(str(thePredict)+'\n')
       #cm = confusion_matrix(Y_test,y_pred.round())
       #theAc.append(float(accuracy(cm)))
    
       theFile=theFile+1
    #for a in theAc:
    #   f_w.write(str(a)+'\t')
    #print(sum(theAc)/float(len(theAc)))
    #f_w.write(str((sum(theAc)/float(len(theAc))))+'\t')
    #f_w.write(str(statistics.stdev(theAc))+'\n')
    f_w.close()
