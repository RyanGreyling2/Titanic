# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

training_data = pd.read_csv('/kaggle/input/titanic/train.csv')
# Convert categorical labels to numbers
def getLastName(name):
    last_name = name.split(", ")
    first_char = last_name[0][0].lower()
    return ord(first_char) - 96
    
training_data['Last Name'] = training_data['Name'].apply(getLastName)

training_data = training_data.drop(['Name', 'Ticket'], axis=1)


# total = training_data['Last Name'].value_counts()
# survived = training_data.loc[training_data['Survived'] == 1, 'Last Name'].value_counts()
# ratio = survived.divide(total).fillna(0)
# print(ratio)
# plt.scatter(ratio.index, ratio.values)


catToNum = {"Sex": {"male": 1, "female": 2},
           "Embarked": {"S": 1, "C": 2, "Q": 3}}
training_data = training_data.replace(catToNum)

def ageFix(age):
    if str(age) == 'nan':
        return 29.7
    return age

def agePresent(age):
    if str(age) == 'nan':
        return 0
    return 1

training_data['AgePresent'] = training_data['Age'].apply(agePresent)

training_data['Age'] = training_data['Age'].apply(ageFix)

def cabinAvailable(cabin): # Create feature based on whether or not a cabin was listed
    if str(cabin) == "nan":
        return 0
    else:
        return 1

training_data['Cabin'] = training_data['Cabin'].apply(cabinAvailable)

def embarkFix(embark):
    if str(embark) == 'nan':
        return 0
    else:
        return embark

training_data['Embarked'] = training_data['Embarked'].apply(embarkFix)    

# print(training_data[:10])

numpy_data = np.array(training_data)
# cov = np.corrcoef(numpy_data, rowvar=False)

features = numpy_data[::,:-1:].astype('float32')
target = numpy_data[::,-1].astype('int64')

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=16, step=1)
selector = selector.fit(features, target)

unimportant_features = []

for i, importance in enumerate(selector.support_,0):
    if not importance:
        unimportant_features.append(i)
        
np.savetxt("/kaggle/output/unimportant_features.csv", unimportant_features, delimiter=",")

features = np.delete(features, unimportant_features, 1)
columns = np.delete(columns, unimportant_features, 0)

target = np.reshape(target, (-1, 1))
modified_data = np.concatenate((features, target), axis=1)

modified_data = pd.DataFrame(data=modified_data, columns=columns)
print(modified_data)

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session