import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Dataset=pd.read_csv(r"C:\Users\Lenovo\Desktop\Pyhton_Practice\11-04-2022\Task\framingham.csv")
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = SimpleImputer(missing_values= np.nan, strategy= 'mean', fill_value=None, copy=True)
imputer.fit(X)
X = imputer.transform(X)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
