#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# [1] Import libraries

import numpy as np 
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     RandomizedSearchCV)
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report)
from sklearn.svm import SVC
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier )  
# In[2]:

# load dataset

# In[3]:

data_churn=pd.read_csv("C:/Users/1/Desktop/Machine learning midterm/churn_cust.csv")

# In[4]:

df= data_churn.copy()
data_churn1 = df[df["Exited"] == 1]

# In[5]:

# Display all columns and rows

# In[6]:

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None)

# In[7]:


# Get data overview

# In[8]:

def dataoveriew(data_churn, message):
    print('{message}:')
    print('Number of rows: ', data_churn.shape[0])
    print("Number of features:", data_churn.shape[1])
    print("Data Features:")
    print(data_churn.columns.tolist())
    print("Missing values:", data_churn.isnull().sum().values.sum())
    print("Unique values:")
    print(data_churn.nunique())
    
# In[9]:

dataoveriew(data_churn, 'Overview of the dataset')

# In[10]:

# Find data types

# In[11]:

data_churn.dtypes

# In[12]:


# Data sample

# In[13]:

data_churn.head()

# In[14]:

# Get data summary

# In[15]:

data_churn.describe()

# In[16]:

# explore target variable

# In[17]:

class_names=['No','Yes']
target_instance = data_churn["Exited"].value_counts().to_frame()
target_instance = target_instance.reset_index()
target_instance = target_instance.rename(columns={'index': 'Category'})
fig = px.pie(target_instance, values='Exited', names=('No','Yes'), color_discrete_sequence=["blue", "yellow"],
             title='Distribution of Churn')
fig.show()

# In[18]:

# Define function for bar chart

# In[19]:

def bar(feature, df=data_churn ):
    #Groupby the categorical feature
    temp_df = df.groupby([feature, 'Exited']).size().reset_index()
    temp_df = temp_df.rename(columns={0:'Count'})
    #Calculate the value counts of each distribution and it's corresponding Percentages
    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    #Calculate the value counts of each distribution and it's corresponding Percentages
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100,1) for element in div_list]
    #Defining string formatting for graph annotation
    #Numeric section
    def num_format(list_instance):
        formatted_str = ''
        for index,num in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{num}%, ' #append to empty string(formatted_str)
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{num}% & '
            else:
                formatted_str=formatted_str+f'{num}%'
        return formatted_str
    #Categorical section
    def str_format(list_instance):
        formatted_str = ''
        for index, cat in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{cat}, '
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{cat} & '
            else:
                formatted_str=formatted_str+f'{cat}'
        return formatted_str


    #Running the formatting functions
    num_str = num_format(percentage)
    cat_str = str_format(categories)
    
    #Setting graph framework
    fig = px.bar(temp_df, x=feature, y='Count',color='Exited', title=f'Churn rate by {feature}',barmode="group", color_discrete_sequence=["green","purple"])
    fig.add_annotation(
                text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.1,
                y=1.1,
                bordercolor='black',
                borderwidth=0.5)
    fig.update_layout(
        # margin space for the annotations on the right
        margin=dict(r=600),
    )
     
    return fig.show() 

# In[ ]:


# Bar plot of churn by Gender


# In[20]:


bar('Gender')


# In[21]:

# Bar plot of churn by country

# In[22]:

bar('Geography')


# In[22]:

# Bar plot of churn by active member

# In[23]:

bar('IsActiveMember')

# In[24]:

# Bar plot of churn by member has credit card

# In[24]:

bar('HasCrCard')

# In[25]:

# Bar plot by number of products

# In[26]:


bar('NumOfProducts')

# In[27]:

# Numeric variables plot

# In[28]:

# Convert float variables to numeric

# In[29]:

data_churn['Balance'] = pd.to_numeric(data_churn['Balance'],errors='coerce')
data_churn['EstimatedSalary'] = pd.to_numeric(data_churn['EstimatedSalary'],errors='coerce')

# In[30]:

# Define function for histogram

# In[31]:

def hist(feature):
    group_df = data_churn.groupby([feature, 'Exited']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Exited', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["blue", "yellow"])
    fig.show()

# In[32]:

# Histogram plot for Tenure

# In[33]:

hist('Tenure')

# In[34]:

# Histogram plot for Age

# In[35]:

hist('Age')

# In[36]:

# Histogram plot for Balance

# In[37]:

hist('Balance')

# In[38]:

# Define feature to plot histogram for credit score and estimated salary

# In[39]:

def hist(feature):
    group_df = data_churn.groupby([feature, 'Exited']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Exited', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["yellow", "blue"])
    fig.show()

# In[40]:

# Plot of creditscore

# In[41]:

hist('CreditScore')

# In[42]:

# Histogram plot of estimated salary

# In[43]:

hist('EstimatedSalary')

# In[44]:

# Create a new empty dataframe

# In[45]:

new_df = pd.DataFrame()

# In[46]:

# Update the new dataframe

# In[47]:

new_df['Tenure_new'] =  pd.qcut(data_churn['Tenure'], q=3, labels= ['low', 'medium', 'high'])
new_df['CreditScore_new'] =  pd.qcut(data_churn['CreditScore'], q=3, labels= ['low', 'medium', 'high'])
new_df['Age_new'] =  pd.qcut(data_churn['Age'], q=3, labels= ['low', 'medium', 'high'])
new_df['Exited'] = data_churn['Exited']

# In[48]:

bar('Tenure_new', new_df)

# In[49]:

# Bar of Age bin

# In[50]:

bar('Age_new', new_df)

# In[51]:

# Bar plot of credit score bin

# In[52]:

bar('CreditScore_new', new_df)

# In[53]:

# get variables in new dataframe

# In[54]:

data_churn1 = data_churn[['Gender', 'Age',
'CreditScore','Geography', 'IsActiveMember',
'Balance', 'NumOfProducts', 'HasCrCard', 'Tenure',
                          'EstimatedSalary'
]]

# In[55]:

# Find correlation between variables

# In[56]:

correlations = data_churn1.corrwith(data_churn.Exited)
correlations = correlations[correlations!=1]
positive_correlations = correlations[
correlations >0].sort_values(ascending = False)
negative_correlations =correlations[
correlations<0].sort_values(ascending = False)

# In[57]:

# Print positive and negative correlations

# In[58]:

print('Most Positive Correlations:', positive_correlations)
print('Most Negative Correlations:', negative_correlations)

# In[59]:

correlations = data_churn1.corrwith(data_churn.Exited)
correlations = correlations[correlations!=1]
correlations.plot.bar(
        figsize = (14, 6), 
        fontsize = 15, 
        color = 'blue',
        rot = 45, grid = True)
plt.title('Correlation with Churn Rate',
horizontalalignment="center", fontstyle = "normal", 
fontsize = "14", fontfamily = "sans-serif")

# In[60]:

# Get correlation between variables

# In[61]:

corr = data_churn.corr()
fig = px.imshow(corr,width=600, height=600)
fig.show()

# In[62]:

# In[ ]:

# Data preprocessing

# In[64]:

data_churn.drop(labels=['RowNumber','Surname'], axis=1, inplace=True)

# In[65]:

# One hot encoding

# In[66]:

data_churn = pd.get_dummies(data_churn, columns =["Geography", "Gender"], drop_first = False)

# In[ ]:

# In[67]:

data_churn.head()

# In[ ]:

# In[68]:

Y= data_churn['Exited']

# In[ ]:

# Get categorical dataframe

# In[69]:

cat_data_churn = data_churn[["Geography_Germany", "Geography_France","Geography_Spain", "Gender_Male","Gender_Female" ,"HasCrCard","IsActiveMember"]]

# In[70]:

X= data_churn.drop(labels=["Exited","Geography_Germany", "Geography_France","Geography_Spain", "Gender_Male","Gender_Female" ,"HasCrCard","IsActiveMember"],axis=1)
   

# In[ ]:

# get labels for modelling

# In[71]:

cols = X.columns
index = X.index

# In[72]:

# Transform dataset

# In[73]:

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns = cols, index = index)
X = pd.concat([X,cat_data_churn], axis = 1)  

# In[ ]:

# Get dataset shape


# In[74]:

print(X.shape, Y.shape)

# In[ ]:

# get training and test datasets

# In[75]:

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=40)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

# In[76]:

train_identity = X_train['CustomerId']
X_train = X_train.drop(columns = ['CustomerId'])

# In[77]:

test_identity = X_test['CustomerId']
X_test = X_test.drop(columns = ['CustomerId'])

# In[78]:

# Get identifiers

# In[79]:

# model 1 KNeighbors Classifier

# In[80]:

from sklearn.metrics import classification_report

# In[81]:

knn = KNeighborsClassifier()

# In[82]:

# Set up hyperparameter grid for tuning

# In[83]:

knn_param_grid = {'n_neighbors' : np.arange(5,26),
                  'weights' : ['uniform', 'distance']}

# In[84]:

# Tune hyperparameters

# In[85]:

knn_cv = GridSearchCV(knn, param_grid = knn_param_grid, cv = 5)

# In[86]:

# Fit knn to training data

# In[87]:

knn_cv.fit(X_train, Y_train)

# In[88]:

# Get info about best hyperparameters

# In[89]:

print("Tuned KNN Parameters: {}".format(knn_cv.best_params_))
print("Best KNN Training Score:{}".format(knn_cv.best_score_))

# In[90]:

# Predict knn on test data


# In[91]:

print("KNN Test Performance: {}".format(knn_cv.score(X_test, Y_test)))

# In[92]:

# Obtain model performance metrics

# In[93]:

knn_pred_prob = knn_cv.predict_proba(X_test)[:,1]
knn_auroc = roc_auc_score(Y_test, knn_pred_prob)
print("KNN AUROC: {}".format(knn_auroc))
knn_y_pred = knn_cv.predict(X_test)
print(classification_report(Y_test, knn_y_pred))

# In[94]:

# 2nd model Logistic Regression

# In[95]:

#  classifier Instantiation

# In[96]:

lr = LogisticRegression(random_state = 30)

# In[97]:

# tune by setting up hyperparameter grid

# In[98]:

lr_param_grid = {'C' : [0.0001, 0.001, 0.01, 0.05, 0.1] }

# In[99]:

# Now Tune hyperparamters

# In[100]:

lr_cv = GridSearchCV(lr, param_grid = lr_param_grid, cv = 5)

# In[101]:

# Fit lr to training data

# In[102]:

lr_cv.fit(X_train, Y_train)

# In[103]:

# Find best hyperparameters

# In[104]:

print("Tuned LR Parameters: {}".format(lr_cv.best_params_))
print("Best LR Training Score:{}".format(lr_cv.best_score_)) 

# In[105]:

# Predict lr on test data

# In[106]:

print("LR Test Performance: {}".format(lr_cv.score(X_test, Y_test)))

# In[107]:

# model performance metrics

# In[108]:

lr_pred_prob = lr_cv.predict_proba(X_test)[:,1]
lr_auroc = roc_auc_score(Y_test, lr_pred_prob)
print("LR AUROC: {}".format(lr_auroc))
lr_y_pred = lr_cv.predict(X_test)
print(classification_report(Y_test, lr_y_pred))

# In[109]:

# 3rd model Random Forest

# In[110]:

# classifier instantiation

# In[111]:

rf = RandomForestClassifier(random_state = 30)

# In[112]:

# Tune hyperparameter grid 

# In[113]:

rf_param_grid = {'n_estimators': [200, 250, 300, 350, 400, 450, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4]}

# In[114]:

# Tune hyperparameters

# In[115]:

rf_cv = RandomizedSearchCV(rf, param_distributions = rf_param_grid, cv = 5, 
                           random_state = 30, n_iter = 20)

# In[116]:

# Fit RF to training data

# In[117]:

rf_cv.fit(X_train, Y_train)

# In[118]:

# best hyperparameters


# In[119]:

print("Tuned RF Parameters: {}".format(rf_cv.best_params_))
print("Best RF Training Score:{}".format(rf_cv.best_score_)) 

# In[120]:

# Predict RF on test data

# In[121]:

print("RF Test Performance: {}".format(rf_cv.score(X_test, Y_test)))

# In[122]:

# model performance metrics

# In[123]:

rf_pred_prob = rf_cv.predict_proba(X_test)[:,1]
rf_auroc = roc_auc_score(Y_test, rf_pred_prob)
print("RF AUROC: {}".format(rf_auroc))
rf_y_pred = rf_cv.predict(X_test)
print(classification_report(Y_test, rf_y_pred))

# In[ ]:

# Feature importances of Random Forest

# In[124]:

rf_optimal = rf_cv.best_estimator_
rf_feat_importances = pd.Series(rf_optimal.feature_importances_, index=X_train.columns)
rf_feat_importances.nlargest(5).plot(kind='barh', color = 'r')
plt.title('Feature Importances from Random Forest Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.savefig('model-rf_feature_importances.png', dpi = 200, bbox_inches = 'tight')
plt.show()

# In[125]:

# 4th model Stochastic Gradient Boosting

# In[126]:

sgb = GradientBoostingClassifier(random_state = 30)

# In[127]:

# Tune hyperparameter grid 

# In[128]:

sgb_param_grid = {'n_estimators' : [200, 300, 400, 500],
                  'learning_rate' : [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                  'max_depth' : [3, 4, 5, 6, 7],
                  'min_samples_split': [2, 5, 10, 20],
                  'min_weight_fraction_leaf': [0.001, 0.01, 0.05],
                  'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  'max_features': ['sqrt', 'log2']}

# In[129]:

# hyperparamters tuning

# In[130]:

sgb_cv = RandomizedSearchCV(sgb, param_distributions = sgb_param_grid, cv = 5, 
                            random_state = 30, n_iter = 20)

# In[131]:

# Fit SGB to training data

# In[132]:

sgb_cv.fit(X_train, Y_train)

# In[133]:

# best hyperparameters

# In[134]:

print("Tuned SGB Parameters: {}".format(sgb_cv.best_params_))
print("Best SGB Training Score:{}".format(sgb_cv.best_score_))

# In[135]:

# Predict SGB on test data

# In[136]:

print("SGB Test Performance: {}".format(sgb_cv.score(X_test, Y_test)))

# In[137]:

# model performance metrics

# In[140]:

predictions = sgb_cv.predict(X_test)

# In[141]:

sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(Y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
print(classification_report(Y_test, sgb_y_pred))

# In[142]:

# SGB feature importances

# In[143]:

sgb_optimal = sgb_cv.best_estimator_
sgb_feat_importances = pd.Series(sgb_optimal.feature_importances_, 
                                 index=X_train.columns)
sgb_feat_importances.nlargest(8).plot(kind='barh', color = 'g')
plt.title('Feature Importances from Stochastic Gradient Boosting Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.savefig('model-sgb_feature_importances.png', dpi = 200, 
            bbox_inches = 'tight')
plt.show()

# In[144]:

# Define function for confusion matrix

# In[145]:

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score

# In[146]:

import itertools
from matplotlib import rc,rcParams
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.rcParams.update({'font.size': 19}),
    plt.imshow(cm, interpolation='nearest', cmap=cmap),
    plt.title(title,fontdict={'size':'16'}),
    plt.colorbar(),
    tick_marks = np.arange(len(classes)),
    plt.xticks(tick_marks, classes, rotation=45,fontsize=12,color="blue"),
    plt.yticks(tick_marks, classes,fontsize=12,color="blue"),
    rc('font', weight='bold'),
    fmt = '.1f',
    thresh = cm.max(),
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
           horizontalalignment="center",
           color="yellow"),
    
    plt.ylabel('True label',fontdict={'size':'16'}),
    plt.xlabel('Predicted label',fontdict={'size':'16'}),
    plt.tight_layout()

# In[147]:

# Plot confusion matrix

# In[148]:

sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(Y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions))

# In[149]:

# Confusion matrix of Gradient Boosting model

# In[150]:

sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(Y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
cm = confusion_matrix(Y_test, predictions) 
Churn_cust_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (12,8))
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(Churn_cust_cm, annot=True, fmt='g',cmap="YlGnBu" 
           )
class_names=['Non Churn','Churn']
tick_marks = np.arange(len(class_names))
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
ax.xaxis.set_label_position("top")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# In[151]:

# comparing models

# In[152]:

knn_fpr, knn_tpr, knn_thresh = roc_curve(Y_test, knn_pred_prob)
plt.plot(knn_fpr,knn_tpr,label="KNN: auc="+str(round(knn_auroc, 3)),
         color = 'blue')
lr_fpr, lr_tpr, lr_thresh = roc_curve(Y_test, lr_pred_prob)
plt.plot(lr_fpr,lr_tpr,label="LR: auc="+str(round(lr_auroc, 3)),
         color = 'red')
rf_fpr, rf_tpr, rf_thresh = roc_curve(Y_test, rf_pred_prob)
plt.plot(rf_fpr,rf_tpr,label="RF: auc="+str(round(rf_auroc, 3)),
         color = 'green')
sgb_fpr, sgb_tpr, sgb_thresh = roc_curve(Y_test, sgb_pred_prob)
plt.plot(sgb_fpr,sgb_tpr,label="SGB: auc="+str(round(sgb_auroc, 3)),
         color = 'yellow')
plt.plot([0, 1], [0, 1], color='gray', lw = 1, linestyle='--', 
         label = 'Random Guess')

plt.legend(loc = 'best', frameon = True, facecolor = 'lightgray')
plt.title('ROC Curve for Classification Models')
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.text(0.85,0.75, 'threshold = 0', fontsize = 12)
plt.arrow(0.85,0.8, 0.14,0.18, head_width = 0.06)
plt.text(0.05,0, 'threshold = 1', fontsize = 12)
plt.arrow(0.05,0, -0.03,0, head_width = 0.06)
plt.savefig('plot-ROC_4models.png', dpi = 800)
plt.show()

# In[153]:

# ROC curve of Gradient Boosting classifier

# In[154]:

sgb_fpr, sgb_tpr, sgb_thresh = roc_curve(Y_test, sgb_pred_prob)
plt.plot(sgb_fpr,sgb_tpr,label="SGB: auc="+str(round(sgb_auroc, 3)),
         color = 'yellow')

plt.plot([0, 1], [0, 1], color='gray', lw = 1, linestyle='--', 
         label = 'Random Guess')

plt.legend(loc = 'best', frameon = True, facecolor = 'lightgray')
plt.title('ROC Curve for Classification Models')
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.text(0.85,0.75, 'threshold = 0', fontsize = 12)
plt.arrow(0.85,0.8, 0.14,0.18, head_width = 0.06)
plt.text(0.05,0, 'threshold = 1', fontsize = 12)
plt.arrow(0.05,0, -0.03,0, head_width = 0.06)
plt.savefig('plot-ROC_4models.png', dpi = 800)
plt.show()

# In[155]:

# Check SGB model predictions with probabilities

# In[156]:


final_results = pd.concat([test_identity, Y_test], axis = 1).dropna()
final_results['Predictions'] = sgb_y_pred
final_results["Probchurn(%)"] = sgb_pred_prob
final_results["Probchurn(%)"] = final_results["Probchurn(%)"]*100
final_results["Probchurn(%)"]=final_results["Probchurn(%)"].round(2)
final_results = final_results[['CustomerId', 'Exited', 'Predictions', 'Probchurn(%)']]
final_results ['Ranking'] = pd.qcut(final_results['Probchurn(%)'].rank(method = 'first'),10,labels=range(10,0,-1))

# In[157]:


final_results.head(20)

# In[158]:

sorted_final_results=final_results.sort_values(["Probchurn(%)"],ascending=[False])

# In[159]:


print(sorted_final_results)

# In[ ]:

# In[160]:


X_test["prob_high_risk"] = sgb_pred_prob
df_risky = X_test[X_test["prob_high_risk"] > 0.9]
display(df_risky.head(10)[["prob_high_risk"]])

# In[ ]:








