import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data=pd.read_csv("student-mat.csv",sep=";")
data=data[["G1","G2","G3","studytime","famrel","failures","health","absences"]]
# print(data.head())
predict="G3"

X=np.array(data.drop([predict], 1))
y=np.array(data[predict])
x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(X,y,test_size=0.1)

"""best=0
for _ in range(20):
    x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(X,y,test_size=0.1)

    linear=linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test)
    print(acc)

    if acc>best:
        best=acc
        with open("studentgrade.pickle","wb") as f:
            pickle.dump(linear,f)"""

pickle_read= open("studentgrade.pickle","rb")
linear= pickle.load(pickle_read)

    
print('Coefficient: \n', linear.coef_)
print("intercept : \n", linear.intercept_)

predictions=linear.predict(x_test)
# print(predictions)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])
 
plot="health"
plt.scatter(data[plot],data["G3"])
plt.xlabel(plot)
plt.ylabel("final grade")
plt.show()