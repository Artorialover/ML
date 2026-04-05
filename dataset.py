from sklearn import datasets
from sklearn import model_selection

data=datasets.load_iris();
print(data)
print(data.target_names)

print(data.DESCR)
print(data.data)
print(data.target)
print(data.feature_names)

x_train,x_test,y_train,y_test=model_selection.train_test_split(data.data,data.target,test_size=0.2)
print(x_train)
print(y_train)
print(len(y_train))