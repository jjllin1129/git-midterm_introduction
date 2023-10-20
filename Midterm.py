import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

#%% 匯入資料
data_train = pd.read_csv('train.csv')
data_xtest = pd.read_csv('X_test.csv')

#%% 分出X Y（training data）
x_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]

#%%把NAN變成0
#列出你想要排除的columns name
columns_to_exclude = ['D_sms', 'D_sm1', 'D_tm0', 'D_Ra_Capacity', 'D_Ra_CDR']
# 要替換的columns
columns_to_fill = [col for col in x_train.columns if col not in columns_to_exclude]
#  NaN 替换为 0
x_train[columns_to_fill] = x_train[columns_to_fill].fillna(0)

#%% 對於D_windows進行處理 讓他在0~1之間
x_train['D_windows'] = x_train['D_windows'].apply(lambda x: min(1, max(0, x)))
#%% cross validation
num_trees = 10
kfold = KFold(n_splits=10)
model = XGBClassifier(n_estimators=num_trees)
results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())

#%% train XGBoost
xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)

#%% predict 
ypred = xgbc.predict(data_xtest)
ypred = pd.DataFrame(ypred, columns=['label'])
ypred['id'] = range(1, len(ypred) + 1)
ypred = ypred[['id', 'label']]
ypred.to_csv('submission.csv', index=False)
