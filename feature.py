from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn import feature_selection

data=[['Red'],['Blue'],['Green'],['Blue'],['Red']]

encoder=preprocessing.OneHotEncoder(sparse_output=True,handle_unknown='ignore')
x_encoder=encoder.fit_transform(data)
print(x_encoder)
print(encoder.get_feature_names_out())

data=[
    {'age':25,'city':"New York"},
    {'age':30,'city':"BS"},
    {'age':35,'city':"New York"}
]
dict=feature_extraction.DictVectorizer(sparse=False);

x_dict=dict.fit_transform(data)
print(x_dict)
print(dict.get_feature_names_out())

feature_selection.VarianceThreshold(0.5)
feature_selection.SelectKBest()
