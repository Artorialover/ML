from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

data=datasets.load_iris()
X=data.data
y=data.target

x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)


scaler=preprocessing.StandardScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.transform(x_test)

model=linear_model.LogisticRegression()
model.fit(x_train_scaler,y_train)
y_predict=model.predict(x_test_scaler)
print(y_predict)
acc=metrics.accuracy_score(y_test,y_predict)
print(f'score:{acc:.2f}')
print(f'score:{metrics.classification_report(y_test,y_predict,target_names=data.target_names)}')

