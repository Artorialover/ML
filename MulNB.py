from sklearn import datasets
from sklearn import neighbors
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import feature_extraction

news=datasets.fetch_20newsgroups()
X=news.data
y=news.target

x_train,x_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)

tf=feature_extraction.text.TfidfVectorizer();
x_train_1=tf.fit_transform(x_train)
x_test_2=tf.transform(x_test)


model=naive_bayes.MultinomialNB()
model.fit(x_train_1,y_train)
y_predict=model.predict(x_test_2)
print(y_predict)
acc=metrics.accuracy_score(y_test,y_predict)
print(f'score:{acc:.2f}')
print(f'score:{metrics.classification_report(y_test,y_predict,target_names=news.target_names)}')

