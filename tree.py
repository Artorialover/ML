from sklearn import datasets
from sklearn import neighbors
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics

data=datasets.load_iris()
X=data.data
y=data.target

x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)


model=tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
y_predict=model.predict(x_test)


print(y_predict)
print(y_test)
acc=metrics.accuracy_score(y_test,y_predict)
print(f'score:{acc:.2f}')
print(f'score:{metrics.classification_report(y_test,y_predict,target_names=data.target_names)}')

