#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report


# In[3]:


loan_data = pd.read_csv('loan.csv', low_memory=False)
half_count = len(loan_data)/2
loan_data = loan_data.dropna(thresh = half_count, axis = 1)
loan_data = loan_data.reset_index()
#drop all columns that contain more thab 50% NA Values
loan_data = loan_data.drop(['url'], axis = 1) # drop URL and Desc columns, predominently descriptive and add no value
loan_data.shape


# In[3]:


data_dictionary = pd.read_excel('LCDataDictionary.xlsx') #importing our data dictionary, so as to identify the relevance of our features by checking the description
data_dictionary = data_dictionary.rename(columns = {'LoanStatNew': 'name'})
data_dictionary.shape
loan_data_dtypes = pd.DataFrame(loan_data.dtypes, columns = ['dtypes'])
loan_data_dtypes = loan_data_dtypes.reset_index()#inorder to create a new column 'index' which contains the names of all our features. 
loan_data_dtypes = loan_data_dtypes.rename(columns = {'index': 'name'})
loan_data_dtypes['first row'] = loan_data.iloc[0].values
loan_data_dtypes.head()
descriptive_data = loan_data_dtypes.merge(data_dictionary, on = 'name', how = 'left')
print(descriptive_data)
#The main reason I did this was to better envision which rows are important for our classification.
#it is also imperative to identify data leakage, i.e. features that attain their value after the user has applied for loan.
#For example, the funded_amt feature will only get a value after the loan has been granted. 
#After close examination, I will go ahead and drop all the rows that I believe do not contribute to our prediction. 


# In[4]:


cols = ['id','member_id','funded_amnt','funded_amnt_inv','earliest_cr_line','last_credit_pull_d','initial_list_status',
             'int_rate','sub_grade','emp_title','issue_d','zip_code','out_prncp','out_prncp_inv',
             'total_pymnt','total_pymnt_inv','total_pymnt_inv','total_rec_prncp','total_rec_int', 'total_rec_late_fee','recoveries', 'collection_recovery_fee', 'last_pymnt_d','last_pymnt_amnt']

loan_data = loan_data.drop(cols, axis = 1) #dropped int_rate and sub_grade due to the presence of the grade column, which is highly correlated to these two
#Just grade will help us to better form clusters. 
#Most of the others, like out_prncp_inv leak data from the future. 
#Now, let us consider the fico scores of the client. Fico scores are essentialy the credit scores of the client.


# In[6]:


loan_data.shape
print(descriptive_data[descriptive_data.name == 'loan_status'])
#loan_status is our target variable. However, it is a string, so we need to do some processing to get it into a numerical value. 
print(loan_data['loan_status'].value_counts())
#We only care about loans that were Charged off or Fully paid. Hence, we remove the rows corresponding to the others. 
loan_data = loan_data[(loan_data['loan_status'] == 'Fully Paid' )| (loan_data['loan_status']=='Charged Off')]
loan_data.shape


# In[7]:


categorize_targets = {'loan_status': {'Fully Paid': 1, 'Charged Off': 0 }}
loan_data.shape
loan_data = loan_data.replace(categorize_targets)
loan_data.shape
#remove columns that contain only one unique value




# In[8]:


print(loan_data['loan_status'])

for col in loan_data.columns:
    if len(loan_data[col].unique()) < 4:
        print(loan_data[col].value_counts())


# In[9]:


#since payment plan, policy code and application type have mainly a single value, we can drop them. 
drop_cols = ['application_type', 'pymnt_plan' , 'policy_code']
loan_data = loan_data.drop(drop_cols, axis = 1)


# In[10]:


null_counts = loan_data.isnull().sum()
print(null_counts)


# In[11]:


loan_data = loan_data.drop(['tot_coll_amt','tot_cur_bal','total_rev_hi_lim','next_pymnt_d'], axis = 1)
#Dropping those columns with way too many null values for my liking
loan_data = loan_data.dropna()


# In[12]:


#now, we need to deal with categorical variables, before we use SVM to classify the above. 
print(loan_data.dtypes.value_counts())
#we have 11 object colums, we need to examine them to determine whether they are Ordinal / Nominal.


# In[13]:


#examining the different object columns - 
print(loan_data.select_dtypes(include = ['object']))
#checking whether they are categorical

cols = ['home_ownership', 'grade','verification_status', 'emp_length', 'term', 'addr_state']

for colm in cols:
    print(loan_data[colm].value_counts())
#this proves that they are indeed categorical
#purpose and title seem to have alot of overlapping features. Furthermore, they purpose has less unique values, so we keep purpose. 
#however, add_state contains too many unique values, so best to drop
loan_data = loan_data.drop(['addr_state'],axis = 1)
    


# In[14]:


loan_data = loan_data.drop(['title'],axis = 1)


# In[15]:


#now, moving on to the rest of the categorical variables. 
#our ordinal variables are - grade and emp length
#our nominal variables are - home_ownership, verification status, purpose and term. 
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0

    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }
}

loan_data = loan_data.replace(mapping_dict)


# In[16]:


print(loan_data['emp_length'].value_counts())
#now performing onehotencoding on the nominal ones - 
nominal_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(loan_data[nominal_columns])
loan_data = pd.concat([loan_data, dummy_df], axis=1)
loan_data = loan_data.drop(nominal_columns, axis=1)


# In[17]:


#print(loan_data.head())
target_vector = loan_data["loan_status"]
print(len(target_vector))


# In[18]:


loan_data_features = loan_data.drop(["loan_status"], axis = 1)
loan_data_features.shape


# In[19]:


print(target_vector.head())


# In[1]:


x_train,x_test,y_train,y_test = train_test_split(loan_data_features, target_vector, test_size = .30)
p
print(x_train.shape)
print(y_train.shape)
#svc = SVC(kernel = 'linear').fit(x_train,y_train)
#y_pred = svc.predict(x_pred)

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

