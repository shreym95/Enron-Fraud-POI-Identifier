#!/usr/bin/python

########################################################################################
import sys
import pickle
import pprint
from tester import dump_classifier_and_data, main as my_main
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as mp
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


########################################################################################
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
    features_list = features_list + ['poi_outgoing', 'poi_incoming', 'total_wealth']
    # Getting best features
    features_list = get_k_best(data_dict, features_list, 10)
    # Updated feature list

    print "=> Choice of 10 best features: "
    pprint.pprint(features_list)

    # Step 4: Extracting features and labels according to feature list
    my_dataset = data_dict
    # labels, features = featureExtract(my_dataset, features_list)

    # Step 5: Choosing and running Untuned Algorithms
    # clf = GaussianNB()
    # clf = DecisionTreeClassifier()
    # clf = AdaBoostClassifier()

    # Step 6 : Tuning Above Classifiers and checking Performance

    # clf_nb, params_nb = tune_NB()
    # clf_dt, params_dt = tune_DT()
    clf_adb, params_adb = tune_ADB()
    # clf_svc, params_svc = tune_SVC()

    # create pipeline
    pipeline = createPipe(clf_adb)
    # params['pca__n_components'] = range(3,8)
    # print pipeline.named_steps['pca'].explained_variance_ratio_

    # create GridSearch
    clf = GridSearchCV(pipeline, params_adb)
    # pred, labels_test = gridSearchCV(pipeline, params, features, labels)

    # testing scores after tuning
    # evalMetrics(pred, labels_test)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    print "--------------- Evaluating Performance! -----------------"
    dump_classifier_and_data(clf, my_dataset, features_list)
    my_main()


def dataExplore(data_dict) :
    print "**** Exploratory Data Analysis ****\n____________________________________\n"

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

    print "\n**** End of Exploratory Data Analysis ****\n__________________________________________\n"

### Task 2: Remove outliers
def removeOutlier(data_dict, feature_list) :
    # Initial Visualisations to spot outliers
    print "\n**** Visualisations and Outlier Removal ****\n____________________________________________\n"
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

    print "\n---> Another Outlier is observed in second plot, but it is an actual POI and is relevant."
    print "\n**** End of Outlier Removal and Visualisations ****\n__________________________________________________\n"

def visualise(data, feat1, feat2, x_label, y_label ) :
    for each in data :
        x = each[feat1]
        y = each[feat2]
        mp.scatter(x,y)
    mp.xlabel(x_label)
    mp.ylabel(y_label)
   # mp.show()                  UNCOMMENT LATER BEFORE SUBMISSION


### Task 3: Create new feature(s)

def createFeature(data_dict) :

    # Feature 1 => Total Wealth = Total Stock Value + Total payments
    # Feature 2 => POI Interaction Ratio = Total POI Interaction / Total Interaction
    # Replacing NaN values with 0
    finance_features = ['total_stock_value', 'total_payments', 'salary', 'bonus', 'deferred_income',
                        'exercised_stock_options', 'long_term_incentive', 'restricted_stock']
    email_features = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
                      'shared_receipt_with_poi']

    for person in data_dict.values() :
        for fin in finance_features :
            person[fin] = handleNaN(person[fin], 0)
        for em in email_features :
            person[em] = float(handleNaN(person[em], 0)) # Conversion to float to obtain proportion later

    for person in data_dict.values() :
        # Feature 1
        person['total_wealth'] = person['total_stock_value'] + person['total_payments']
        # Feature 2


        if person['from_messages'] != 0 :
            person['poi_outgoing'] = person['from_this_person_to_poi'] / person['from_messages']
        else:
            person['poi_outgoing'] = 0.0

        if person['to_messages'] != 0 :
            person['poi_incoming'] = person['from_poi_to_this_person'] / person['to_messages']
        else:
            person['poi_incoming'] = 0.0
    return data_dict

# To handle all NaN values, with desired replacement
def handleNaN(val_to_check, new_val) :
    def_list = ["NaN","nan", "NA","NULL"]
    if val_to_check in def_list :
        return new_val
    else :
        return val_to_check

### Extract features and labels from dataset for local testing
def featureExtract(my_dataset, features_list) :

    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    return labels, features

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def tune_NB() :
    print "------------- Using Naive Bayes -----------------\n"

    nb_clf = GaussianNB()
    param_grid = {}

    return nb_clf, param_grid

def tune_SVC() :
    print "------------- Using SVM -----------------\n"

    svc_clf = svm.SVC()
    param_grid = {
        # 'pca__n_components' : range(3,10),
        'clf__C' : [1000],
        'clf__gamma' : [0.1],
        'clf__kernel' : ['rbf', 'poly'],
        'clf__random_state' : [42]
    }

    return svc_clf, param_grid

def tune_DT():
    print "------------- Tuning Decision Tree -----------------\n"
    clf = DecisionTreeClassifier()
    param_grid = {
        'clf__criterion': ['entropy', 'gini'],
        'clf__splitter': ['best', 'random']
    }
    return clf, param_grid

def tune_ADB():
    print "------------- Using AdaBoost Ensemble -----------------\n"

    clf = AdaBoostClassifier()
    param_grid = {
        'clf__algorithm' : ['SAMME', 'SAMME.R'],
        'clf__learning_rate': [0.5, 1, 1.5, 2],
        'clf__n_estimators' : [100, 300, 500, 1000, 5000]
    }

    return clf, param_grid

#####################################################################################

def createPipe(clf):
    print "------------------ Creating Pipeline ---------------\n"
    scaler = MinMaxScaler()
    pca = PCA()

    estimators = [('scale', scaler),('pca', pca),('clf', clf)]
    pipe = Pipeline(estimators)

    print "=> Pipeline Created!"
    return pipe

#######################################################################################

def gridSearchCV(pipeline, parameters, features, labels):
    print "---------------- Tuning Algorithm using GridSearchCV -----------------\n"
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    print parameters, type(parameters)
    grid_search = GridSearchCV(pipeline, parameters, cv = None, scoring='f1')
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)

    print '=> Best score: %0.3f' % grid_search.best_score_
    print '=> Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '--> \t%s: %r' % (param_name, best_parameters[param_name])
    tuned_clf = grid_search.best_estimator_
    return pred, labels_test

#######################################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
def evalMetrics(predictions, test_labels):
    print "---------------- Final Performance -----------------\n"
    print "=> Accuracy Score: ", skm.accuracy_score(test_labels, predictions)
    print "=> Precision Score: ", skm.precision_score(test_labels, predictions)
    print "=> Recall Score: ", skm.recall_score(test_labels, predictions)

### Load the dictionary containing the dataset
def get_k_best(data_dict, features_list, k):

    data = featureFormat(data_dict, features_list)
    labels_train, features_train = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features_train, labels_train)

    unsorted_list = zip(features_list[1:], k_best.scores_)
    print k_best.scores_
    sorted_list = sorted(unsorted_list, key=lambda x: x[1], reverse=True)
    k_best_features = dict(sorted_list[:k])

    return ['poi'] + k_best_features.keys()



if __name__ == '__main__':
    main()