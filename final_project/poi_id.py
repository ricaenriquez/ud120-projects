#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from tester import test_classifier, dump_classifier_and_data

# Task 1: Explore the dataset
# Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

""" SUPPORT CODE
people = data_dict.keys()
npeople = len(data_dict.keys())
print "There are", npeople, "people in the dataset."
print "There are", len(data_dict[people[0]]), "features per person."
npoi = 0
for person in data_dict:
    if data_dict[person]["poi"]:
        npoi += 1
print "There are", npoi, "POIs in the dataset."
# Make an array of all the features in the data set
features_list = []
features_list.append("poi")
for feat in data_dict[people[0]]:
    if feat != "poi" and feat != "email_address":
        features_list.append(feat)
"""

""" SUPPORT CODE
# Make dictionary of the number of NaNs for each feature for all the people,
# POIs, and non-POIs.
nnan = {}
nnan["general"] = {}
for feat in features_list:
    nnan["general"][feat] = 0
    nnan["poi"][feat] = 0
    nnan["not_poi"][feat] = 0
    for person in data_dict:
        if data_dict[person][feat] == "NaN":
            nnan["general"][feat] += 1
            if data_dict[person]["poi"] == 1:
                nnan["poi"][feat] += 1
            else:
                nnan["not_poi"][feat] += 1
# Write dictionary to CSV file
import csv
with open("nnans.csv", "wb") as csvfile:
    f = csv.writer(csvfile, delimiter="\t")
    f.writerow(["feature", "fraction NANs for all", "fraction NANs for POI",
                "fraction NANs for non-POI"])
    for feat in features:
        f.writerow([feat, nnan["general"][feat]*1.0/npeople,
                    nnan["poi"][feat]*1.0/npoi,
                    nnan["not_poi"][feat]*1.0/(npeople-npoi)])
# """

# Task 2: Select what financial features to use
# """ SUPPORT CODE
features_list = ["poi", "salary", "total_payments", "bonus", "deferred_income",
                 "total_stock_value", "expenses", "exercised_stock_options",
                 "other", "long_term_incentive", "restricted_stock"]
# """


# Task 3: Remove outliers
# for key in data_dict.keys():
#     print key
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
# data_dict["BHATNAGAR SANJAY"]["restricted_stock"] = \
#     -data_dict["BHATNAGAR SANJAY"]["restricted_stock"]

""" SUPPORT CODE
data = featureFormat(data_dict, features_list)
"""

# Plot features vs. salary and color POIs as red dots, non-POIs as blue dots
""" SUPPORT CODE
import matplotlib.pyplot
i = 2
for feat in features_list[2:]:
    for point in data:
        salary = point[1]
        feat_plot = point[i]
        if point[0] == 1:
            matplotlib.pyplot.scatter(salary, feat_plot, color="red")
        else:
            matplotlib.pyplot.scatter(salary, feat_plot, color="blue")
    i += 1
    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel(feat)
    matplotlib.pyplot.show()
# Explore outliers in total_payments, salary, and restricted stock
for key in data_dict:
    if (data_dict[key]["total_payments"] > 1e8) and \
            (data_dict[key]["salary"] > 1e6):
        if (data_dict[key]["total_payments"] != "NaN") and \
                (data_dict[key]["salary"] != "NaN"):
            print key, data_dict[key]["total_payments"], \
                data_dict[key]["salary"], data_dict[key]["poi"]
for key in data_dict:
    if (data_dict[key]["restricted_stock"] < 0):
        if (data_dict[key]["restricted_stock"] != "NaN"):
            print key, data_dict[key]["restricted_stock"], data_dict[key]["poi"]
"""

# Task 4: Create new features
def compute_fraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    """

    if all_messages == 0:
        return 0.
    if all_messages == "NaN":
        return 0.
    if poi_messages == "NaN":
        return 0.
    return poi_messages*1.0/all_messages*1.0

for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
    data_dict[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
    data_dict[name]["fraction_to_poi"] = fraction_to_poi

# Add these new features to the features list
# """ SUPPORT CODE
features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")
# Recreate the arrays with the financial features and new e-mail features
data = featureFormat(data_dict, features_list, sort_keys=True)
# """

# Task 5: Intelligently Select Features
# """SUPPORT CODE
label, features = targetFeatureSplit(data)
# First scale the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
# """

# Use feature selection algorithms to pick the top seven features
"""SUPPORT CODE
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
selector.fit(scaled_features, label)
print selector.get_support()
print selector.scores_

from sklearn.linear_model import RandomizedLasso
selector = RandomizedLasso(alpha=0.0008)
selector.fit(scaled_features, label)
print selector.get_support()
print selector.scores_

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
selector = RFE(estimator=LinearSVC(), n_features_to_select=10)
selector.fit(scaled_features, label)
print selector.support_
print selector.ranking_
"""

features_list = ["poi", "salary", "bonus", "deferred_income", "total_stock_value",
                 "expenses", "exercised_stock_options", "long_term_incentive",
                 "fraction_to_poi"]

# Task 6: Try a varity of classifiers
""" SUPPORT CODE
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
print "AdaBoost"
clf = Pipeline([("scale", MinMaxScaler()),
                ("ada", AdaBoostClassifier(random_state=5))])
test_classifier(clf, data_dict, features_list)
print "RandomForest"
clf = Pipeline([("scale", MinMaxScaler()),
                ("ranfor", RandomForestClassifier(random_state=5))])
test_classifier(clf, data_dict, features_list)
print "Bagging"
clf = Pipeline([("scale", MinMaxScaler()),
                ("bagging", BaggingClassifier(random_state=5))])
test_classifier(clf, data_dict, features_list)
print "ExtraTree"
clf = Pipeline([("scale", MinMaxScaler()),
                ("extra", ExtraTreesClassifier(random_state=5))])
test_classifier(clf, data_dict, features_list)
print "AdaBoost Random"
clf = Pipeline([("scale", MinMaxScaler()),
                ("ada", AdaBoostClassifier(random_state=5, base_estimator=RandomForestClassifier()))])
test_classifier(clf, data_dict, features_list)
print "AdaBoost ExtraTree"
clf = Pipeline([("scale", MinMaxScaler()),
                ("ada", AdaBoostClassifier(random_state=5, base_estimator=ExtraTreesClassifier()))])
test_classifier(clf, data_dict, features_list)
"""

# Task 7: Tune your classifier to achieve better than .3 precision and recall
# using our testing script.
"""SUPPORT CODE
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test,labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=5)
from sklearn.grid_search import GridSearchCV
ada = AdaBoostClassifier(random_state=5)
params = {'algorithm': ('SAMME', "SAMME.R"), 'n_estimators': [40, 50, 60],
          'learning_rate': [.5, .75, 1.0, 1.25]}
clf = GridSearchCV(ada, params, scoring="recall", cv=5)
clf.fit(features_test, labels_test)
print clf.best_estimator_
"""

"""SUPPORT CODE
# Initial loop
# al_var = ["SAMME", "SAMME.R"]
# learn_var = [.3, .4, .5, .6, .7, .8, .9, 1., 1.1, 1.2]
# est_var = [50, 60, 70, 80]
# Second loop
al_var = ["SAMME.R"]
learn_var = [.8, .85, .9, .95, 1.]
est_var = [50, 60, 65, 70, 75, 80]
als = []
learning = []
est = []
prec = []
rec = []
f = []
for i in al_var:
    for j in learn_var:
        for k in est_var:
            print "New CLF"
            als.append(i)
            learning.append(j)
            est.append(k)
            ada = AdaBoostClassifier(random_state=5, algorithm=i, learning_rate=j, n_estimators=k)
            clf = Pipeline([("scale", MinMaxScaler()), ("ada", ada)])
            precision, recall, f1 = test_classifier(clf, data_dict, features_list)
            prec.append(precision)
            rec.append(recall)
            f.append(f1)
import pandas as pd
df = pd.DataFrame({'algorithm': pd.Series(als),
                  'learning': pd.Series(learning),
                  'estimators': pd.Series(est),
                  'precision': pd.Series(prec),
                  'recall': pd.Series(rec),
                  'f1': pd.Series(f)})
print "Max F1"
print df[df["f1"] == max(df[(df["precision"] >= 0.3) & (df["recall"] >= 0.3)]["f1"])]
print "Max recall"
print df[df["recall"] == max(df[df["precision"] >= 0.3]["recall"])]
print "Max precision"
print df[(df["precision"] == max(df[df["recall"] >= 0.3]["precision"]))]
"""

# """SUPPORT CODE
print "Final AdaBoost Classifier"
ada = AdaBoostClassifier(random_state=5, n_estimators=70, learning_rate=0.9)
clf = Pipeline([("scale", MinMaxScaler()),
                ("ada", ada)])
# test_classifier(clf, data_dict, features_list)
# """

# Dump your classifier, dataset, and features_list so
# anyone can run/check your results.
dump_classifier_and_data(clf, data_dict, features_list)


# Code to print out intermediate metrics
"""SUPPORT CODE
print "AdaBoost without fraction_to_poi"
ada = AdaBoostClassifier(random_state=5, n_estimators=70, learning_rate=0.9)
clf = Pipeline([("scale", MinMaxScaler()),
                ("ada", ada)])
test_classifier(clf, data_dict, features_list[0:-1])
# """
