import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


cust_data = pd.read_csv("D:\cust_churn\customer_churn.csv")
print(cust_data.head())
y = cust_data.Churn
feature_columns = ['Age', 'Total_Purchase', 'Account_Manager', 'Years']
X = cust_data[feature_columns]
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.25, random_state=1)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(train_X, train_y)

pickle.dump(log_model, open("model.pkl", "wb"))
# val_predictions = log_model.predict(val_X)
# x = accuracy_score(val_y,val_predictions)
# print(x)