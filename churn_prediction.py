import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve , roc_auc_score
# Importing data to our file
data=pd.read_csv('churn_prediction_simple.csv')
# showing description of data file here
print(data.info())
# Drop the rows where at least one element is missing.
data=data.dropna()
print(data.info())
# seperating dependent and independent variable
X=data.drop(columns=['churn','customer_id'])
Y=data['churn']
# scale your dataset
scale=StandardScaler()
x_scale=scale.fit_transform(X)
# splitting train and test dataset
# stratify arranges data into classs
# test size=0.8 means 80% is train data and 20% is test data
x_train,x_test,y_train,y_test=train_test_split(x_scale,Y,test_size=0.8,stratify = Y)
# Now implement logistic regression
# class_weight=balance is used to balance our data
classifier=LR(class_weight="balanced")
classifier.fit(x_train,y_train)
# predicting classes
predict=classifier.predict(x_test)
# predicting probability
predicted_probabilities = classifier.predict_proba(x_test) 
print(predict,predict.shape)
print(predicted_probabilities)
# making confusion martix
c_matrix=cm(y_test,predict)
print(c_matrix)
# now making accuracy matrix 
acc_matrix=classifier.score(x_test,y_test)
print(acc_matrix)
# precision matrix to avoid false positive
pre_matrix=precision_score(y_test,predict)
print(pre_matrix)
# recall matrix to avoid false -ve
Recall = recall_score(y_test, predict)
print(Recall)
# calculating f1_score
f1_matrix=f1_score(y_test,predict)
print(f1_matrix)
# AUC-ROC curve
fpr, tpr, threshold = roc_curve(y_test, predicted_probabilities[:,1])
plt.plot( fpr, tpr, color = 'green')
plt.plot( [0,1], [0,1], label = 'baseline', color = 'red')
plt.xlabel('FPR', fontsize = 15)
plt.ylabel('TPR', fontsize = 15)
plt.title('AUC-ROC', fontsize = 20)
plt.show()
print(roc_auc_score(y_test, predicted_probabilities[:,1]))
# now plotting the coeeficient
c=classifier.coef_.reshape(-1)
x=X.columns
coef_plot=pd.DataFrame({'coefficients': c,
                        'variable' : x})
coef_plot=coef_plot.sort_values(by="coefficients")
print(coef_plot.head())
plt.barh(coef_plot['variable'],coef_plot['coefficients'])
plt.xlabel( "Coefficient Magnitude", fontsize = 15)
plt.ylabel('Variables', fontsize = 15)
plt.title('Coefficient plot', fontsize = 20)
plt.show()