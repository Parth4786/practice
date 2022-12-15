import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('diabetes.csv')
df.head()
df.isnull().sum()

from sklearn.model_selection import train_test_split
X=df.drop(columns=['Outcome']) 
y=df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("NAIVE BAYES CLASSIFICATION")

from sklearn.naive_bayes import GaussianNB 
nb = GaussianNB()
nb.fit(X_train,y_train)
nb.score(X_test,y_test)
y_pred = nb.predict(X_test)


from sklearn.metrics import confusion_matrix,classification_report
print("Confusion Matrix") 
confusion_matrix(y_test,y_pred)


print("Classification Report") 
(classification_report(y_test,y_pred))


X=df.drop(columns=['Outcome']) 
y=df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn import tree 
dt = tree.DecisionTreeClassifier()

print("\nDECISION TREE CLASSIFICATION")
dt.fit(X_train,y_train) 
print("Testing Score") 
dt.score(X_test,y_test)

y_pred_dt = dt.predict(X_test)


print("Confusion Matrix") 
confusion_matrix(y_test,y_pred_dt)


print("Classification Report") 
print(classification_report(y_test,y_pred_dt))


nb_probs = nb.predict_proba(X_test) 
dt_probs = dt.predict_proba(X_test)


dt_probs = dt_probs[:, 1] 
nb_probs = nb_probs[:, 1] 
nb_probs

from sklearn.metrics import roc_curve, roc_auc_score


nb_auc = roc_auc_score(y_test, nb_probs) 
dt_auc = roc_auc_score(y_test, dt_probs) 
print('Decision Tree AUROC = ' + str(dt_auc)) 
print('Naive Bayes AUROC = ' + str(nb_auc))

nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs) 
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
 
import matplotlib.pyplot as plt


plt.plot(nb_fpr, nb_tpr, linestyle='--', label='Naive Bayes (AUROC = %0.3f)' % nb_auc) 
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree (AUROC = %0.3f)' % dt_auc)

plt.title('ROC Plot')
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#Run above for 5 datasets

import numpy as np 
import matplotlib.pyplot as plt 
barWidth= 0.25 
fig = plt.subplots(figsize=(12, 8))
naive_bayes = [78, 89,94, 94, 62]
decision_tree = [70,74, 98, 82, 51] 
br1 =np.arange(len(naive_bayes))
br2= [x + barWidth for x in br1]
plt.bar(br1, naive_bayes, color ='b', width = barWidth,edgecolor ='grey', label ='Naive Bayes') 
plt.bar(br2, decision_tree, color ='y', width = barWidth, edgecolor='grey', label ='Decision Tree')
plt.xlabel('Datasets', fontweight ='bold', fontsize = 15) 
plt.ylabel('Percentage', fontweight ='bold', fontsize = 15) 
plt.xticks([r+ barWidth for r in range(len(naive_bayes))], ['diabeties.csv', 'sonar.csv', 'BankNoteAuthentication.csv', 'ionosphere_data.csv', 'haberman.csv'],rotation=30)
plt.legend()
plt.show() 
#Above code is for comparison

#Below Code is for k-fold cross validation
from sklearn.model_selection import KFold, train_test_split,cross_val_score 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier 
from numpy import mean
from sklearn.metrics import accuracy_score 
cv = KFold(n_splits=10, shuffle=True, random_state=1) 
model = AdaBoostClassifier() 
def evaluate_model(cv, model):
    x, y = get_dataset()
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)	
    return np.mean(scores), scores.min(), scores.max() 
def naive_bayes_classification(X_train, X_test, y_train, y_test) :
    #Training gaussian model gnb = GaussianNB()
    gnb.fit(X_train, y_train) #Getting predictions	
    y_pred = gnb.predict(X_test) 
    return accuracy_score(y_test, y_pred) 
def  decision_tree_classification(X_train, X_test, y_train, y_test): 
     #Training decision tree	
     dtc = tree.DecisionTreeClassifier( criterion="entropy", max_depth=4, max_features=2, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter="best")
     dtc.fit(X_train, y_train)
        #Getting predictions	
     y_pred = dtc.predict(X_test)
     return accuracy_score(y_test, y_pred) 
n_splits=10
#K-Fold Cross Validation 
kf = KFold(n_splits=n_splits) 
avg_score = [0, 0] 
for trainIndex, testIndex in kf.split(df) :
    avg_score[0] += naive_bayes_classification(X_train, X_test,y_train, y_test) 
    avg_score[1] += decision_tree_classification(X_train, X_test,y_train, y_test) 
    print(f"Naive Bayes Avg. Accuracy = {avg_score[0]*100/10} %")	
print(f"Decision Tree Avg. Accuracy = {avg_score[1]*100/10} %")
#Bagging Ensemble model
estimators = [("naiveBayes", GaussianNB()), ("decisionTree",  tree.DecisionTreeClassifier())]
baggingEnsemble = VotingClassifier(estimators) 
baggingEnsemble.fit(X_train, y_train) 
y_pred = baggingEnsemble.predict(X_test) 
baggingAccuracy = accuracy_score(y_test, y_pred) 
print(f"Bagging Accuracy: {baggingAccuracy*100} %")
#Adaboost Ensemble model 
adaboostEnsemble = AdaBoostClassifier(n_estimators=3) 
adaboostEnsemble.fit(X_train, y_train) 
y_pred = adaboostEnsemble.predict(X_test) 
adaboostAccuracy = accuracy_score(y_test, y_pred) 
print(f"Adaboost Accuracy: {adaboostAccuracy*100} %")
#Plotting 
plt.bar([1,2,3,4], [avg_score[0]/10,avg_score[1]/10,baggingAccuracy,adaboostAccuracy],
color=["red","green","pink","blue"]) 
plt.xlabel("Model") 
plt.ylabel("Accuracy") 
plt.xticks([1,2,3,4],["Naive Bayes", "Decision Tree", "Bagging", "AdaBoost"])
plt.show()
