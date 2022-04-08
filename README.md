# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sai Eswar Kandukuri
RegisterNumber:  212221240020
*/

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))

```

## Output:
<img width="647" alt="1" src="https://user-images.githubusercontent.com/93427011/162414632-6bc5a9a0-4757-4b21-99a2-dfb4ff46dd97.png">
<img width="648" alt="Screenshot 2022-04-08 at 3 32 33 PM" src="https://user-images.githubusercontent.com/93427011/162414651-911f291c-a38a-422b-a2a1-518929207d46.png">
<img width="647" alt="2" src="https://user-images.githubusercontent.com/93427011/162414663-7507cf95-87a8-45ec-899d-444a3107db38.png">
<img width="647" alt="3" src="https://user-images.githubusercontent.com/93427011/162414674-c7e2e059-949b-45d2-a0ae-1dc93f4211d7.png">
<img width="647" alt="4" src="https://user-images.githubusercontent.com/93427011/162414686-425842ca-a6e4-429a-8e63-87605755e1ed.png">
<img width="647" alt="5" src="https://user-images.githubusercontent.com/93427011/162414701-914fce80-282c-4dac-b9e2-08847b632281.png">
<img width="647" alt="6" src="https://user-images.githubusercontent.com/93427011/162414730-b637a64f-6308-43f4-8b42-509482848c28.png">
<img width="647" alt="7" src="https://user-images.githubusercontent.com/93427011/162414746-8dcd5b3d-d097-41b9-838d-757b439dcbdc.png">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
