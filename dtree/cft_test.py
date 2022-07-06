import os
os.chdir('/home/sangmin')
path = os.getcwd()
os.chdir(path + '/cft/dtree')

# from sklearn.metrics import classification_report, confusion_matrix
# src 다운로드 후 실행해보니 오류가 뜹니다. 아래서 말하는대로 setup.py다운받고자 했지만, 오류가 뜨네요.
# =================================================================
# from scikit_learn.sklearn.model_selection import train_test_split
# from scikit_learn.sklearn.tree import DecisionTreeClassifier
# from scikit_learn.sklearn import tree
# =================================================================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np

import pydotplus

# load example dataset
tennis_data = pd.read_csv('play_tennis.csv')

outlook = list(tennis_data.outlook)
temp = list(tennis_data.temp)
humidity = list(tennis_data.humidity)
wind = list(tennis_data.wind)
play = list(tennis_data.play)

#dictionary
data = { 'outlook' : outlook,
        'temp' : temp,
        'humidity' : humidity,
        'wind' : wind,
        'play' : play}

#d2Dataframe(String)
df = pd.DataFrame(data)

#string2int
df.outlook = df.outlook.replace('Sunny', 0)
df.outlook = df.outlook.replace('Overcast', 1)
df.outlook = df.outlook.replace('Rain', 2)

df.temp = df.temp.replace('Hot', 3)
df.temp = df.temp.replace('Mild', 4)
df.temp = df.temp.replace('Cool', 5)

df.humidity = df.humidity.replace('High', 6)
df.humidity = df.humidity.replace('Normal', 7)

df.wind = df.wind.replace('Strong', 8)
df.wind = df.wind.replace('Weak', 9)

df.play = df.play.replace('No', 10)
df.play = df.play.replace('Yes', 11)

# y=f(X)
X = np.array(pd.DataFrame(df, columns=['outlook','temp', 'humidity', 'wind']))
y = np.array(pd.DataFrame(df, columns=['play']))

X_train, X_test, y_train, y_test = train_test_split(X,y)

# decision tree
dt_clf = DecisionTreeClassifier(criterion="entropy", splitter="best")
dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)

# visuallization
from IPython.display import Image

feature_names = tennis_data.columns.tolist()
feature_names
feature_names = feature_names[1:5]
target_name = np.array(['Play No', 'Play Yes'])

dt_dot_data = tree.export_graphviz(dt_clf, out_file = None,
                                  feature_names = feature_names,
                                  class_names = target_name,
                                  filled = True, rounded = True,
                                  special_characters = True)
dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)

Image(dt_graph.create_png())
dt_graph.write_pdf("test_entropy_best.pdf")