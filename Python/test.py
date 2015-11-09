import numpy as np

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, Imputer
from patsy import dmatrices, dmatrix

#Print you can execute arbitrary python code
df_test = pd.read_csv("data_test.csv")
df_train = pd.read_csv("data_train.csv")

print(df_train.describe())
print(type(df_train))
print(df_test.head())

print df_test.dtypes
#df_train.loc[df_train["workclass"] == "?", "workclass"] = np.nanmedian(df_train['workclass'])

#df_train.loc[df_train['class'] == '>50K', 'class'] = 1
#df_test.loc[df_test['class'] == '>50K', 'class'] = 1

#df_train.loc[df_train['class'] == '<=50K', 'class'] = 0
#df_test.loc[df_test['class'] == '<=50K', 'class'] = 0


y_train, X_train = dmatrices('categ ~ age + fnlwgt + capital_gain + capital_loss + hours_per_week + occupation + relationship + race', df_train)
y_test, X_test = dmatrices('categ ~ age + fnlwgt + capital_gain + capital_loss + hours_per_week + occupation + relationship + race', df_test)
#X_test = dmatrix('age + fnlwgt + capital_gain + capital_loss + hours_per_week', df_test)

steps1 = [('poly_features', PolynomialFeatures(3, interaction_only=True)),
          ('logistic', LogisticRegression(C=5555., max_iter=25, penalty='l2'))]
steps2 = [('rforest', RandomForestClassifier(min_samples_split=15, n_estimators=73, criterion='entropy'))]
pipeline1 = Pipeline(steps=steps1)
pipeline2 = Pipeline(steps=steps2)

pipeline1.fit(X_train, y_train.ravel())
print('Accuracy of Training (Logistic Regression-Poly Features (cubic)): {:.4f}'.format(pipeline1.score(X_train, y_train.ravel())))
print('Accuracy of Testing (Logistic Regression-Poly Features (cubic)): {:.4f}'.format(pipeline1.score(X_test, y_test.ravel())))

pipeline2.fit(X_train[:600], y_train[:600].ravel())
calibratedpipe2 = CalibratedClassifierCV(pipeline2, cv=3, method='sigmoid')
calibratedpipe2.fit(X_train[600:], y_train[600:].ravel())
print('Accuracy of Training (Random Forest - Calibration): {:.4f}'.format(calibratedpipe2.score(X_train, y_train.ravel())))
print('Accuracy of Testing (Random Forest - Calibration): {:.4f}'.format(calibratedpipe2.score(X_test, y_test.ravel())))


#output = pd.DataFrame(columns=['age','fnlwgt','capital_gain','capital_loss','hours_per_week','categ', 'Predicted'])
#output['PassengerId'] = df_test['PassengerId']

output = df_test

output1 = pd.DataFrame(columns=['Predicted'])

# Predict the survivors and output csv
output1['Predicted'] = pipeline1.predict(X_test).astype(int)

# Predict the survivors and output csv

#fileh= open("output.csv", "w+")
#fileh.write(pipeline1.predict(X_test).astype(int))
#fileh.close()
#output.to_csv('output.csv', index=False)
#output1.to_csv('output1.csv', index=False)


#print("\n\nSummary statistics of training data")
#print(df_train.describe())

# Age imputation

