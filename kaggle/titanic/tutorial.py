import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

train_data =  pd.read_csv("./datasets/train.csv")
test_data =  pd.read_csv("./datasets/test.csv")

print(train_data.head())

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


features = ["Pclass", "Sex",  "SibSp", "Parch"]
Y = train_data["Survived"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X,Y)
tree_predict_orig = tree.predict(X)
tree_predict = tree.predict(X_test)

print(accuracy_score(Y,tree_predict_orig))

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,Y)
predictions_orig = model.predict(X)
predictions = model.predict(X_test)

print(accuracy_score(Y,predictions_orig))

output = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived': predictions})
output.to_csv('output/submission.csv', index=False)
print("Your submission was successfully saved!")