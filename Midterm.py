import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
#%% create XGBoost model
my_xgboost = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=5000, eval_metric='logloss')
#%% import data
data_train = pd.read_csv('train.csv')
x_test = pd.read_csv('X_test.csv')
#%% split x and y (training data）
x_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
#%%train
my_xgboost.fit(x_train, y_train)
#%% predict
y_pred = my_xgboost.predict(x_test)
#%% nan to  zero
x_train = x_train.fillna(0)
#%% export submit
ypred = pd.DataFrame(y_pred, columns=['label'])
ypred['id'] = range(1, len(ypred) + 1)
ypred = ypred[['id', 'label']]
ypred.to_csv('submission.csv', index=False)
#%% plot important feature
feature_importance = my_xgboost.feature_importances_
feature_names = x_train.columns  
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), feature_importance, tick_label=feature_names)
plt.xticks(rotation=90)
plt.show()
#%% combine importand_feature and features name to a DataFrame
feature_importance_df = pd.DataFrame(feature_importance)
feature_importance_df['name'] = feature_names

