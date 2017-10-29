#!/usr/bin/python

########################################################################################
import sys
import pickle
import pprint
import matplotlib.pyplot as mp
from collections import OrderedDict
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

########################################################################################


def dataExplore(data_dict):
    print "**** Exploratory Data Analysis ****\n____________________________________\n"

    print "=> Total points in Enron Dataset: ", len(data_dict)
    all_features = {sec_key for primary_key in data_dict.values() for sec_key in primary_key}
    print "=> Number of features for each data point: ", len(all_features)
    poi = [(name, feat) for name, feat in data_dict.items() if feat["poi"]==1]
    print "=> Total persons of interest in dataset: ", len(poi)

    nan = {key:0 for key in all_features}
    for person in data_dict.values():
        for key, val in person.items():
            if val == "NaN":
                nan[key] +=1
    print "=> NaN values in whole of dataset: "
    pprint.pprint(nan)

    print "\n**** End of Exploratory Data Analysis ****\n__________________________________________\n"

### Task 2: Remove outliers
def removeOutlier(data_dict, feature_list):
    #Initial Visualisations to spot outliers
    print "\n**** Visualisations and Outlier Removal ****\n____________________________________________\n"
    data_to_plot = featureFormat(data_dict, feature_list, sort_keys = True)

    salary_index = feature_list.index('salary')
    bonus_index = feature_list.index('bonus')
    total_payments_index = feature_list.index('total_payments')
    total_stock_index = feature_list.index('total_stock_value')

    #Plotting
    visualise(data_to_plot, salary_index, bonus_index, 'salary', 'bonus')
    visualise(data_to_plot, total_stock_index, total_payments_index, 'total stock', 'total payments')

    points_to_remove = ['THE TRAVEL AGENCY IN THE PARK']    #Payment not related to employees at Enron

    #For extremely high and unusual value of salary
    for name, val in data_dict.items():
        if val['salary'] > 25000000 and val['salary'] != "NaN" :
            points_to_remove.append(name)


    #For persons with no financial data
    for name, val in data_dict.items():
        if val['total_payments'] == "NaN" and val['total_stock_value'] == "NaN" and val['director_fees'] == "NaN":
            points_to_remove.append(name)

    #Removing outliers
    for key in points_to_remove:
        if key in data_dict.keys():
            data_dict.pop(key, 0)
            print "=> Outlier \""+key+"\" removed"

    #Visualising again to see the difference
    data_to_plot = featureFormat(data_dict, feature_list, sort_keys=True)
    visualise(data_to_plot, salary_index, bonus_index, 'salary', 'bonus')
    visualise(data_to_plot, total_stock_index, total_payments_index, 'total stock', 'total payments')

    print "\n---> Another Outlier is observed in second plot, but it is an actual POI and is relevant."
    print "\n**** End of Outlier Removal and Visualisations ****\n__________________________________________________\n"

def visualise(data, feat1, feat2, x_label, y_label ):
    for each in data:
        x = each[feat1]
        y = each[feat2]
        mp.scatter(x,y)
    mp.xlabel(x_label)
    mp.ylabel(y_label)
    mp.show()


### Task 3: Create new feature(s)

def createFeature(data_dict):

    

### Extract features and labels from dataset for local testing
def featureExtract(my_dataset, features_list):

    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    return labels, features

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def NB():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
def evalMetrics(features, labels):

    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### Load the dictionary containing the dataset
def main():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    dataExplore(data_dict)

    #Step 1: Selecting Features
    features_list = ['poi', 'salary', 'bonus', 'total_payments', 'total_stock_value', 'exercised_stock_options', 'restricted_stock',
                    'director_fees', 'long_term_incentive', 'from_poi_to_this_person', 'from_this_person_to_poi','shared_receipt_with_poi']

    removeOutlier(data_dict, features_list)
    #createFeature(data_dict)
    #features_list = selectFeatures(data_dict)
    ### Store to my_dataset for easy export below.
    #my_dataset = data_dict
    #features, labels = featureExtract(my_dataset, features_list)

    #choose algo here

    #evalMetrics(features, labels)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    #dump_classifier_and_data(clf, my_dataset, features_list)


if __name__ == '__main__':
    main()