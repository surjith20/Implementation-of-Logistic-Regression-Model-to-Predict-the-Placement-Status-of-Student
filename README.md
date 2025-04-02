# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Surjith D
RegisterNumber:  212223043006
*/
```

```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver= "liblinear")
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:
![image](https://github.com/user-attachments/assets/71896723-159b-4463-ad9a-c5c10a89a960)
![image](https://github.com/user-attachments/assets/25afe8d6-483d-4c0c-acff-fa52ad652be0)
![image](https://github.com/user-attachments/assets/f00fcd8d-0aaf-4910-85e7-0dbdd6acfc77)
![image](https://github.com/user-attachments/assets/9a46233d-909f-4ea7-8ea3-60da937b1e02)
![image](https://github.com/user-attachments/assets/43349ff1-fd73-4497-b131-a2834edce374)
![image](https://github.com/user-attachments/assets/96e1ff24-42e6-4b00-9d3d-a67c7450f61d)
![image](https://github.com/user-attachments/assets/5e72c82a-7e76-47f0-bd67-dfa4402dc295)
![image](https://github.com/user-attachments/assets/90030462-1d2e-4909-bfee-853bd4327895)
![image](https://github.com/user-attachments/assets/4719f16f-d5b2-4761-b1d6-9af358b8778d)
![image](https://github.com/user-attachments/assets/0d9def67-2eb1-41fa-a576-0a5d7cd02977)
![image](https://github.com/user-attachments/assets/b1a554df-6803-4c9f-af08-22a417bcf65f)
![image](https://github.com/user-attachments/assets/2322afff-aae0-4c31-93d5-b41065ec3782)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
