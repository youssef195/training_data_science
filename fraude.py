import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv(r"C:\Users\9820937G\OneDrive - SNCF\Bureau\trainig data science\detection fraude bank\creditcard.csv")


##    0 = normal transaction
##   1 = fraude 

normal=df[df.Class==0]
fraude=df[df.Class==1]

df.groupby('Class').mean()

normal_sample=normal.sample(n=492)


ds=pd.concat([normal_sample,fraude],axis=0  )

ds.groupby('Class').mean()

x=ds.drop(columns=['Class'],axis=1)
y=ds['Class']

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, stratify= y, random_state=2)

model=LogisticRegression()

model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)

print(training_data_accuracy)

x_test_prediction=model.predict(x_test)    
test_data_accuracy_score=accuracy_score(x_test_prediction,y_test)

print(test_data_accuracy_score)
print(confusion_matrix(x_test_prediction,y_test))