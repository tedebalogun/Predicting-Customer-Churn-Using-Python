#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# Import libraries
# In[1]:
import numpy as np 
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
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


# In[2]:

data_churn = pd.read_csv("C:/Users/1/Desktop/Machine learning midterm/churn_cust.csv")

# get data overview


# In[3]:

def dataoveriew(df, message):
    print('{message}:')
    print('Number of rows: ', df.shape[0])
    print("Number of features:", df.shape[1])
    print("Data Features:")
    print(df.columns.tolist())
    print("Missing values:", df.isnull().sum().values.sum())
    print("Unique values:")
    print(df.nunique())
    
# In[4]:

dataoveriew(data_churn, 'Overview of the dataset')


# In[5]:

# Find data types

# In[6]:

data_churn.dtypes

# In[ ]:

# Get data summary

# In[6]:

data_churn.describe()


# In[7]:

# explore target variable

# In[8]:

target_instance = data_churn["Exited"].value_counts().to_frame()
target_instance = target_instance.reset_index()
target_instance = target_instance.rename(columns={'index': 'Category'})
fig = px.pie(target_instance, values='Exited', names='Category', color_discrete_sequence=["blue", "yellow"],
             title='Distribution of Churn')
fig.show()

# In[9]:

# Define function for bar chart

# In[10]:

#Defining bar chart function
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
    fig = px.bar(temp_df, x=feature, y='Count', color='Exited', title=f'Churn rate by {feature}', barmode="group", color_discrete_sequence=["green","purple"])
    fig.add_annotation(
                text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.4,
                y=1.3,
                bordercolor='black',
                borderwidth=1)
    fig.update_layout(
        # margin space for the annotations on the right
        margin=dict(r=400),
    )
     
    return fig.show()

# In[11]:


# Bar plot of churn by Gender


# In[12]:

bar('Gender')

# In[13]:

# Bar plot of churn by country


# In[14]:

bar('Geography')

# In[15]:

# Bar plot of churn by active member

# In[16]:

bar('IsActiveMember')

# In[17]:

# Bar plot of churn by member has credit card

# In[18]:

bar('HasCrCard')

# In[19]:

# Bar plot by number of products


# In[20]:

bar('NumOfProducts')

# In[21]:

# Numeric variables plot

# In[22]:

# Convert float variables to numeric

# In[23]:

data_churn['Balance'] = pd.to_numeric(data_churn['Balance'],errors='coerce')
data_churn['EstimatedSalary'] = pd.to_numeric(data_churn['EstimatedSalary'],errors='coerce')

# In[24]:

# Define function for histogram

# In[25]:


def hist(feature):
    group_df = data_churn.groupby([feature, 'Exited']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Exited', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["blue", "yellow"])
    fig.show()

# In[26]:

# Histogram plot for Tenure

# In[27]:


hist('Tenure')

# In[28]:

# Histogram plot for Age
# In[29]:

hist('Age')

# In[29 ]:
# Histogram plot for Balance

# In[30]:

hist('Balance')

# In[31]:

# Define feature to plot histogram for credit score and estimated salary

# In[32]:

def hist(feature):
    group_df = data_churn.groupby([feature, 'Exited']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Exited', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["yellow", "blue"])
    fig.show()

# In[ ]:
# Histogram plot of credit score

# In[33]:

hist('CreditScore')

# In[34]:

# Histogram plot of estimated salary

# In[35]:


hist('EstimatedSalary')

# In[36]:

# Create a new empty dataframe

# In[37]:

new_df = pd.DataFrame()

# In[38]:

# Update the new dataframe

# In[39]:

new_df['Tenure_new'] =  pd.qcut(data_churn['Tenure'], q=3, labels= ['low', 'medium', 'high'])
new_df['CreditScore_new'] =  pd.qcut(data_churn['CreditScore'], q=3, labels= ['low', 'medium', 'high'])
new_df['Age_new'] =  pd.qcut(data_churn['Age'], q=3, labels= ['low', 'medium', 'high'])
new_df['Exited'] = data_churn['Exited']

# In[40]:

# Bar chart of bin variables

# In[41]:

bar('Tenure_new', new_df)

# In[42]:

# Bar of Age bin

# In[43]:

bar('Age_new', new_df)

# In[44]:

# Bar plot of credit score bin

# In[45]:

bar('CreditScore_new', new_df)

# In[46]:

# get variables in new dataframe

# In[47]:

data_churn1 = data_churn[['Gender', 'Age',
'CreditScore','Geography', 'IsActiveMember',
'Balance', 'NumOfProducts', 'HasCrCard', 'Tenure',
                          'EstimatedSalary'
]]

# In[48]:

# Find correlation between variables

# In[49]:


correlations = data_churn1.corrwith(data_churn.Exited)
correlations = correlations[correlations!=1]
positive_correlations = correlations[
correlations >0].sort_values(ascending = False)
negative_correlations =correlations[
correlations<0].sort_values(ascending = False)


# In[50]:

# Print positive and negative correlations


# In[51]:


print('Most Positive Correlations:', positive_correlations)
print('Most Negative Correlations:', negative_correlations)

# In[52]:


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


# In[53]:

# Preprocessing data

# In[54]:

data_churn.keys()

# In[55]:

# Drop customer Id


# In[56]:

data_churn.drop(['RowNumber'],axis=1,inplace=True)

# In[57]:

data_churn.drop(['CustomerId'],axis=1,inplace=True)

# In[58]:

data_churn.drop(['Surname'],axis=1,inplace=True)

# In[ ]:

# Get correlation between variables

# In[59]:

corr = data_churn.corr()

# In[60]:

fig = px.imshow(corr,width=600, height=600)

# In[61]:


fig.show()

# In[62]:

# get labels for modelling

# In[63]:

def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})

# In[64]:


data_churn['Gender'] = data_churn['Gender'].map({'Male':1, 'Female':0})


# In[65]:


data_churn = pd.get_dummies(data_churn, drop_first=True)


# In[66]:


X=data_churn.drop(labels=['Exited'],axis=1)


# In[67]:


X.head()


# In[68]:


Y=data_churn['Exited']


# In[69]:


Y.head()


# In[70 ]:


# get training and test datasets


# In[71]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=40)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[72]:


# Remove identifiers


# In[73]:


X_text=X_test.get('CustomerId')
X_test.head()


# In[74]:

# model 1 KNeighbors Classifier

# In[75]:

from sklearn.metrics import classification_report


# In[76]:

knn = KNeighborsClassifier()


# In[77]:


# Set up hyperparameter grid for tuning


# In[78]:

knn_param_grid = {'n_neighbors' : np.arange(5,26),
                  'weights' : ['uniform', 'distance']}

# In[79]:

# Tune hyperparameters


# In[80]:


knn_cv = GridSearchCV(knn, param_grid = knn_param_grid, cv = 5)


# In[81]:


# Fit knn to training data


# In[82]:


knn_cv.fit(X_train, Y_train)


# In[83]:


# Get info about best hyperparameters


# In[84]:


print("Tuned KNN Parameters: {}".format(knn_cv.best_params_))
print("Best KNN Training Score:{}".format(knn_cv.best_score_))


# In[85]:


# Predict knn on test data


# In[86]:


print("KNN Test Performance: {}".format(knn_cv.score(X_test, Y_test)))


# In[87]:


# Obtain model performance metrics


# In[88]:


knn_pred_prob = knn_cv.predict_proba(X_test)[:,1]
knn_auroc = roc_auc_score(Y_test, knn_pred_prob)
print("KNN AUROC: {}".format(knn_auroc))
knn_y_pred = knn_cv.predict(X_test)
print(classification_report(Y_test, knn_y_pred))


# In[89]:


# 2nd model Logistic Regression


# In[90]:


#  classifier Instantiation


# In[91]:


lr = LogisticRegression(random_state = 30)


# In[92]:


# tune by setting up hyperparameter grid


# In[93]:


lr_param_grid = {'C' : [0.0001, 0.001, 0.01, 0.05, 0.1] }


# In[94]:


# Now Tune hyperparamters


# In[95]:


lr_cv = GridSearchCV(lr, param_grid = lr_param_grid, cv = 5)


# In[96]:


# Fit lr to training data


# In[97]:


lr_cv.fit(X_train, Y_train)


# In[98]:


# Find best hyperparameters


# In[99]:


print("Tuned LR Parameters: {}".format(lr_cv.best_params_))
print("Best LR Training Score:{}".format(lr_cv.best_score_)) 


# In[100]:


# Predict lr on test data


# In[101]:


print("LR Test Performance: {}".format(lr_cv.score(X_test, Y_test)))


# In[102]:


# model performance metrics


# In[103]:


lr_pred_prob = lr_cv.predict_proba(X_test)[:,1]
lr_auroc = roc_auc_score(Y_test, lr_pred_prob)
print("LR AUROC: {}".format(lr_auroc))
lr_y_pred = lr_cv.predict(X_test)
print(classification_report(Y_test, lr_y_pred))


# In[104]:


# 3rd model Random Forest


# In[105]:


# classifier instantiation


# In[106]:


rf = RandomForestClassifier(random_state = 30)


# In[107]:


# Tune hyperparameter grid 


# In[108]:


rf_param_grid = {'n_estimators': [200, 250, 300, 350, 400, 450, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4]}


# In[109]:


# Tune hyperparameters


# In[110]:


rf_cv = RandomizedSearchCV(rf, param_distributions = rf_param_grid, cv = 5, 
                           random_state = 30, n_iter = 20)


# In[111]:


# Fit RF to training data


# In[112]:


rf_cv.fit(X_train, Y_train)


# In[113 ]:


# best hyperparameters


# In[114]:


print("Tuned RF Parameters: {}".format(rf_cv.best_params_))
print("Best RF Training Score:{}".format(rf_cv.best_score_)) 
 


# In[115]:


# Predict RF on test data


# In[116]:


print("RF Test Performance: {}".format(rf_cv.score(X_test, Y_test)))


# In[117]:


# model performance metrics


# In[118]:


rf_pred_prob = rf_cv.predict_proba(X_test)[:,1]
rf_auroc = roc_auc_score(Y_test, rf_pred_prob)
print("RF AUROC: {}".format(rf_auroc))
rf_y_pred = rf_cv.predict(X_test)
print(classification_report(Y_test, rf_y_pred))


# In[119]:


# Feature importances of Random Forest


# In[120]:


rf_optimal = rf_cv.best_estimator_
rf_feat_importances = pd.Series(rf_optimal.feature_importances_, index=X_train.columns)
rf_feat_importances.nlargest(5).plot(kind='barh', color = 'r')
plt.title('Feature Importances from Random Forest Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.savefig('model-rf_feature_importances.png', dpi = 200, bbox_inches = 'tight')
plt.show()


# In[121]:


# 4th model Stochastic Gradient Boosting


# In[122]:


sgb = GradientBoostingClassifier(random_state = 30)


# In[123]:


# Tune hyperparameter grid 


# In[124]:


sgb_param_grid = {'n_estimators' : [200, 300, 400, 500],
                  'learning_rate' : [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                  'max_depth' : [3, 4, 5, 6, 7],
                  'min_samples_split': [2, 5, 10, 20],
                  'min_weight_fraction_leaf': [0.001, 0.01, 0.05],
                  'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  'max_features': ['sqrt', 'log2']}


# In[125]:


# hyperparamters tuning


# In[126]:


sgb_cv = RandomizedSearchCV(sgb, param_distributions = sgb_param_grid, cv = 5, 
                            random_state = 30, n_iter = 20)


# In[127]:


# Fit SGB to training data


# In[128]:


sgb_cv.fit(X_train, Y_train)


# In[129]:


# best hyperparameters


# In[130]:


print("Tuned SGB Parameters: {}".format(sgb_cv.best_params_))
print("Best SGB Training Score:{}".format(sgb_cv.best_score_))


# In[131]:


# Predict SGB on test data


# In[132]:


print("SGB Test Performance: {}".format(sgb_cv.score(X_test, Y_test)))


# In[133]:


# model performance metrics


# In[134]:


predictions = sgb_cv.predict(X_test)


# In[135]:


sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(Y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
print(classification_report(Y_test, sgb_y_pred))


# In[136]:


# SGB feature importances


# In[137]:


sgb_optimal = sgb_cv.best_estimator_
sgb_feat_importances = pd.Series(sgb_optimal.feature_importances_, 
                                 index=X_train.columns)
sgb_feat_importances.nlargest(5).plot(kind='barh', color = 'g')
plt.title('Feature Importances from Stochastic Gradient Boosting Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.savefig('model-sgb_feature_importances.png', dpi = 200, 
            bbox_inches = 'tight')
plt.show()


# In[138]:


# Define funstion for confusion matrix


# In[139]:


from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score


# In[140]:


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
           color="red"),

    plt.ylabel('True label',fontdict={'size':'16'}),
    plt.xlabel('Predicted label',fontdict={'size':'16'}),
    plt.tight_layout()


# In[141]:


# Plot confusion matrix


# In[142]:


sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(Y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions))


# In[143]:


# Confusion matrix of Gradient Boosting model


# In[144]:


sgb_pred_prob = sgb_cv.predict_proba(X_test)[:,1]
sgb_auroc = roc_auc_score(Y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
cm = confusion_matrix(Y_test, predictions) 
Churn_cust_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (28,20))
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(Churn_cust_cm, annot=True, fmt='g',cmap="YlGnBu" 
           )
class_names=[0,1]
tick_marks = np.arange(len(class_names))
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
ax.xaxis.set_label_position("top")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[145]:


# comparing models


# In[146]:


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
plt.text(0.85,0.75, 'threshold = 0', fontsize = 8)
plt.arrow(0.85,0.8, 0.14,0.18, head_width = 0.01)
plt.text(0.05,0, 'threshold = 1', fontsize = 8)
plt.arrow(0.05,0, -0.03,0, head_width = 0.01)
plt.savefig('plot-ROC_4models.png', dpi = 500)
plt.show()


# In[147]:


# ROC curve of Gradient Boosting classifier


# In[148]:


sgb_fpr, sgb_tpr, sgb_thresh = roc_curve(Y_test, sgb_pred_prob)
plt.plot(sgb_fpr,sgb_tpr,label="SGB: auc="+str(round(sgb_auroc, 3)),
         color = 'yellow')

plt.plot([0, 1], [0, 1], color='gray', lw = 1, linestyle='--', 
         label = 'Random Guess')

plt.legend(loc = 'best', frameon = True, facecolor = 'lightgray')
plt.title('ROC Curve for Classification Models')
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.text(0.85,0.75, 'threshold = 0', fontsize = 8)
plt.arrow(0.85,0.8, 0.14,0.18, head_width = 0.01)
plt.text(0.05,0, 'threshold = 1', fontsize = 8)
plt.arrow(0.05,0, -0.03,0, head_width = 0.01)
plt.savefig('plot-ROC_4models.png', dpi = 500)
plt.show()

