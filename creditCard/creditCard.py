# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc

# # import dataset
dataf = pd.read_csv('D:\\GRADUATION PROJECT\\DataSets\\fraudTest.csv')

print(dataf['is_fraud'].value_counts())

# #Preprocessing the data 

# print(dataf.columns)
#Check for missing values 
# print(dataf.isnull().sum())

data = dataf.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'gender', 'street','zip', 'trans_num', 'unix_time' ], axis=1)
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])
# print(dataf.dtypes)

# Extract numeric features from trans_date_trans_time
data['hour'] = data['trans_date_trans_time'].dt.hour
data['day_of_week'] = data['trans_date_trans_time'].dt.dayofweek  
data['month'] = data['trans_date_trans_time'].dt.month
data['day'] = data['trans_date_trans_time'].dt.day
data['age'] = data['trans_date_trans_time'].dt.year - data['dob'].dt.year
data.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)
# print(data.columns)


categorical_cols = data.select_dtypes(include=['object']).columns

encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

#Split the data
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# from collections import Counter
# print("Before SMOTE:", Counter(y_train))

#Apllying Smote
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# print("After SMOTE:", Counter(y_train_smote))

import xgboost as xgb

# Convert to DMatrix (optional, but recommended for performance)
dtrain = xgb.DMatrix(X_train_smote, label=y_train_smote)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': (y_train_smote == 0).sum() / (y_train_smote == 1).sum(),  # imbalance handling
    'random_state': 42,
    'use_label_encoder': False
}

# Train model
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100
)

bst.save_model('D:\\GRADUATION PROJECT\\creditCard\\fraud_xgb_model.json')

import pickle

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Save encoders
with open('D:\\GRADUATION PROJECT\\creditCard\\label_encoders.pkl', "wb") as f:
    pickle.dump(encoders, f)

