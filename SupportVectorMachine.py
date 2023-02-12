import sklearn
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

cancer=datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

X=cancer.data
y=cancer.target

x_train, x_test, y_train, y_test= train_test_split(X,y,test_size=0.2)

# print(x_train[:5],y_train[:5])

clf=svm.SVC( kernel="linear")
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)

print(acc)
