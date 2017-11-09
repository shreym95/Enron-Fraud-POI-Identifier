#!/usr/bin/python

########################################################################################################################
import sys
from time import time
import pickle
import pprint
from tester import dump_classifier_and_data, main as tester_main
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import matplotlib.pyplot as mp
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

t0 = time()

########################################################################################################################

def main():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    dataExplore(data_dict)

    # Step 1: Selecting Features
    features_list = ['poi', 'salary', 'bonus', 'total_payments', 'total_stock_value',
                     'exercised_stock_options', 'restricted_stock', 'deferred_income',
                     'long_term_incentive', 'shared_receipt_with_poi']

    # Step 2: Removing Outliers
    removeOutlier(data_dict, features_list)

    # Step 3: Engineering New Features
    data_dict = createFeature(data_dict)
    features_list =  ['poi', 'exercised_stock_options', 'deferred_income',
                      'shared_receipt_with_poi','poi_outgoing', 'poi_incoming',
                      'total_wealth', 'poi_interaction']

    # Getting best features
    features_list = get_best_feats(data_dict, features_list, 6)
    print "\n=> Best features chosen : "
    pprint.pprint(features_list)
    print "\n_____________________________________________"
    # Step 4: Extracting features and labels according to feature list
    my_dataset = data_dict
    labels, features = featureExtract(my_dataset, features_list)

    # Step 5: Choosing and running Untuned Algorithms

    #clf = GaussianNB()
    #clf = DecisionTreeClassifier()
    #clf = AdaBoostClassifier()

    # Step 6 : Tuning Above Classifiers and checking Performance

    #clf_nb, params_nb = tune_NB()
    #clf_dt, params_dt = tune_DT()

    clf_adb, params_adb = tune_ADB()                        # Final chosen algorithm

    # Create pipeline
    scale = MinMaxScaler()
    pca = PCA()                                             # Not used - hampers performance in DT and ADB
    estimators = [('scale', scale), ('clf', clf_adb)]
    pipe = Pipeline(estimators)

    # Create GridSearchCV Instance
    grid = GridSearchCV(pipe, params_adb, scoring='precision')
    grid.fit(features, labels)

    # Final classifier
    clf = grid.best_estimator_

    print '\n=> Chosen parameters :'
    print grid.best_params_


    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    # Calling functions from tester.py for local testing
    #print "--------------- Evaluating Performance! -----------------"
    print "\n===> DUMPING DATASET, FEATURES AND CLASSIFIER TO PKL FILE"
    dump_classifier_and_data(clf, my_dataset, features_list)
    # tester_main()                                    # Uncomment for local testing using tester.py file

########################################################################################################################
# Exploring the data for initial stats
def dataExplore(data_dict) :
    print '\n====> Exploratory Data Analysis!'

    print "=> Total points in Enron Dataset: ", len(data_dict)
    all_features = { sec_key for primary_key in data_dict.values() for sec_key in primary_key }
    print "=> Number of features for each data point: ", len(all_features)
    poi = [(name, feat) for name, feat in data_dict.items() if feat["poi"] == 1]
    print "=> Total persons of interest in dataset: ", len(poi)

    nan = { key : 0 for key in all_features }
    for person in data_dict.values():
        for key, val in person.items():
            if val == "NaN":
                nan[key] +=1
    print "=> NaN values in whole of dataset: "
    pprint.pprint(nan)

    print "\n_____________________________________________"

# Removing unwanted data points
def removeOutlier(data_dict, feature_list) :          # Future: Shift from dict to pandas dataframe from easier handling
    # Initial Visualisations to spot outliers
    print "\n====> Visualisations and Outlier Removal \n"
    data_to_plot = featureFormat(data_dict, feature_list, sort_keys = True)

    salary_index = feature_list.index('salary')
    bonus_index = feature_list.index('bonus')
    total_payments_index = feature_list.index('total_payments')
    total_stock_index = feature_list.index('total_stock_value')

    # Plotting
    visualise(data_to_plot, salary_index, bonus_index, 'salary', 'bonus')
    visualise(data_to_plot, total_stock_index, total_payments_index, 'total stock', 'total payments')

    points_to_remove = ['THE TRAVEL AGENCY IN THE PARK']    # Payment not related to employees at Enron

    # For extremely high and unusual value of salary
    for name, val in data_dict.items() :
        if val['salary'] > 25000000 and val['salary'] != "NaN" :
            points_to_remove.append(name)


    # For persons with no financial data
    for name, val in data_dict.items() :
        if val['total_payments'] == "NaN" and val['total_stock_value'] == "NaN" and val['director_fees'] == "NaN" :
            points_to_remove.append(name)

    # Removing outliers
    for key in points_to_remove :
        if key in data_dict.keys() :
            data_dict.pop(key, 0)
            print "=> Outlier \"" + key + "\" removed"

    # Visualising again to see the difference
    data_to_plot = featureFormat(data_dict, feature_list, sort_keys=True)
    visualise(data_to_plot, salary_index, bonus_index, 'salary', 'bonus')
    visualise(data_to_plot, total_stock_index, total_payments_index, 'total stock', 'total payments')

    print "=> Another Outlier is observed in second plot, but it is an actual POI."
    print "\n_______________________________________________"

# Help function for plotting
def visualise(data, feat1, feat2, x_label, y_label ) :
    for each in data :
        x = each[feat1]
        y = each[feat2]
        mp.scatter(x,y)
    mp.xlabel(x_label)
    mp.ylabel(y_label)
    mp.show()


# Creating new feature(s)

def createFeature(data_dict) :
    print "\n====> Engineering New Features!\n"
    # Distinguishing features into finance and email related
    finance_features = ['total_stock_value', 'total_payments', 'salary', 'bonus', 'deferred_income',
                        'exercised_stock_options', 'long_term_incentive', 'restricted_stock']
    email_features = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
                      'shared_receipt_with_poi']

    # Replacing NaN values with 0
    for person in data_dict.values() :
        for fin in finance_features :
            person[fin] = handleNaN(person[fin], 0)
        for em in email_features :
            person[em] = float(handleNaN(person[em], 0))    # Conversion to float to obtain proportion later
    max_shared_receipt = 0
    for person in data_dict.values():
        if person['shared_receipt_with_poi'] > max_shared_receipt :
            max_shared_receipt = person['shared_receipt_with_poi']

    for person in data_dict.values() :
        # Feature 1
        person['total_wealth'] = person['total_stock_value'] + person['total_payments']
        # Feature 2
        if person['from_messages'] != 0 :
            person['poi_outgoing'] = person['from_this_person_to_poi'] / person['from_messages']
        else:
            person['poi_outgoing'] = 0.0
        # Feature 3
        if person['to_messages'] != 0 :
            person['poi_incoming'] = person['from_poi_to_this_person'] / person['to_messages']
        else:
            person['poi_incoming'] = 0.0
        # Feature 4
        total_poi_interaction = float(person['from_this_person_to_poi'] + person['from_poi_to_this_person'])
        total_interaction = float(person['from_messages'] + person['to_messages'])
        if total_interaction != 0.0 :
            poi_ratio = total_poi_interaction / total_interaction
        else :
            poi_ratio = 0.0

        person['poi_interaction'] = poi_ratio + float(person['shared_receipt_with_poi'] / max_shared_receipt)

    print "_____________________________________________"

    return data_dict

# To handle all NaN values, with desired replacement
def handleNaN(val_to_check, new_val) :
    def_list = ["NaN","nan", "NA","NULL"]
    if val_to_check in def_list :
        return new_val
    else :
        return val_to_check

# Extract features and labels from dataset for local testing
def featureExtract(my_dataset, features_list) :
    print "\n===> Feature And Labels Extraction!"
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    return labels, features

# Tuning Functions for various classifiers
def tune_NB() :
    print "------------- Using Naive Bayes -----------------\n"

    nb_clf = GaussianNB()
    param_grid = {}             # No parameters for tuning

    return nb_clf, param_grid

# Decision Tree Classifier
def tune_DT():
    print "------------- Tuning Decision Tree -----------------\n"
    clf = DecisionTreeClassifier()
    param_grid = {
        'clf__criterion': ['entropy', 'gini'],
        'clf__splitter': ['best', 'random'],
        'clf__min_samples_split': [2, 4, 6]
    }

    return clf, param_grid

# AdaBoost Classifier
def tune_ADB():
    print "\n===> Tuning AdaBoost Ensemble"
    clf = AdaBoostClassifier()
    param_grid = {
        'clf__algorithm' : ['SAMME', 'SAMME.R'],
        'clf__learning_rate': [1, 2],
        'clf__random_state': [42],
        'clf__n_estimators' : [50, 65, 80, 100]
    }

    return clf, param_grid

# Getting K best features using SelectKBest
def get_best_feats(data_dict, features_list, k):

    # Scaling before selecting features
    scaler = MinMaxScaler()
    data = featureFormat(data_dict, features_list)
    labels_train, features_train = targetFeatureSplit(data)
    scaler.fit(features_train)
    features_train_scaled = scaler.transform(features_train)

    skb = SelectKBest(k=k)
    skb.fit(features_train_scaled, labels_train)

    unsorted_list = zip(features_list[1:], skb.scores_)

    sorted_features = sorted(unsorted_list, key=lambda x: x[1], reverse=True)

    print "Feature Scores:\n"
    pprint.pprint(sorted_features)

    k_best_features = dict(sorted_features[:k])
    return ['poi'] + k_best_features.keys()                   # 'poi' needs to be the first key for proper usage later

########################################################################################################################

if __name__ == '__main__':
    main()
    print "\n_____________________________________________\n"
    print "=> Program Completed!"
    print '=> Time taken: ', time()-t0, ' seconds!'