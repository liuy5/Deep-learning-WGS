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
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
import statistics

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

muList=[]
f=open('mutation_class_CNV.txt')
line=f.readline().strip()
while line!='':
    muList.append(line)
    line=f.readline().strip()
f.close()

f_w=open('accuracy_for_50rounds_shuffle.txt','w')
f_w.write('CNV'+'\t'+'accuracy_mean'+'\t'+'std'+'\n')
for theMu in muList:
    print(theMu)
    f_w.write(theMu+'\t')

    theAc=[]

    theFile=0
    while theFile<50:
        data=pd.read_csv('featureVectors/CNV/AA_mutationBased_ADHD/training/'+theMu+'_'+str(theFile)+'.txt',sep='\t',header=0)
        Y = data[['whether_disease']]
        Y = Y.values.ravel()
        f=open('featureVectors/CNV/AA_mutationBased_ADHD/training/'+theMu+'_'+str(theFile)+'.txt')
        line=f.readline().strip()
        temp=re.split('\t+',line)
        tempList=[]
        i=1
        while i<len(temp):
            tempList.append(temp[i])
            i=i+1
        X=data[tempList]
        f.close()

        #the parameters were determined using gp_minimize function
        if theMu=='exonicDel':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='lbfgs', learning_rate = 'adaptive', hidden_layer_sizes=(55,55,55,55,55,55,55,55,55))
        if theMu=='exonicDup':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(97,97,97,97,97,97,97,97,97,97))        
        if theMu=='exonicOthers':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling',hidden_layer_sizes=(100,100,100,100,100,100,100,100))

        if theMu=='intergenicDel':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(55,55,55,55,55,55,55,55,55))
        if theMu=='intergenicDup':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(100,100,100,100,100,100,100,100,100))        
        if theMu=='intergenicOthers':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(81,81,81,81,81,81,81,81,81))

        if theMu=='intronicDel':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='logistic', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(60,60,60,60,60,60,60,60,60,60))
        if theMu=='intronicDup':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(63,63,63,63,63,63,63,63,63))     
        if theMu=='intronicOthers':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(63,63,63,63,63,63,63,63,63))

        if theMu=='splicingDel':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(74,74,74,74,74,74,74))
        if theMu=='splicingDup':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='sgd', learning_rate = 'invscaling', hidden_layer_sizes=(100,100,100,100,100,100,100))       
        if theMu=='splicingOthers':
            clf = MLPRegressor(random_state=0, max_iter=2000, alpha=0.0001, activation='relu', solver='lbfgs', learning_rate = 'constant', hidden_layer_sizes=(39,39,39,39,39,39,39))
    
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        clf.fit(X,Y)

        data=pd.read_csv('featureVectors/CNV/AA_mutationBased_ADHD/testing/'+theMu+'_'+str(theFile)+'.txt',sep='\t',header=0)
        Y_test = data[['whether_disease']]
        Y_test = Y_test.values.ravel()
        f=open('featureVectors/CNV/AA_mutationBased_ADHD/testing/'+theMu+'_'+str(theFile)+'.txt')
        line=f.readline().strip()
        temp=re.split('\t+',line)
        tempList=[]
        i=1
        while i<len(temp):
            tempList.append(temp[i])
            i=i+1
        X_test=data[tempList]
        f.close()
        X_test = scaler.transform(X_test)
        y_pred=clf.predict(X_test)
        cm = confusion_matrix(Y_test,y_pred.round())
        theAc.append(float(accuracy(cm)))
    
        theFile=theFile+1
    for a in theAc:
       f_w.write(str(a)+'\t')
    print(sum(theAc)/float(len(theAc)))
    f_w.write(str((sum(theAc)/float(len(theAc))))+'\t')
    f_w.write(str(statistics.stdev(theAc))+'\n')
f_w.close()
