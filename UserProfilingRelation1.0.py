# Python Project Template
# 1. Prepare Problem
# a) Load libraries
# b) Load dataset

# 2. Summarize Data
# a) Descriptive statistics
# b) Data visualizations

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms

# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use




# 1. Prepare Problem ##########################################################
# 1.a) Load libraries

import pandas as pd
import sys as os
import time
import winsound
from matplotlib import pyplot
#from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from pandas import set_option
#from pandas.plotting import scatter_matrix
#from sklearn.preprocessing import StandardScaler

#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#from sklearn.svm import SVC
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import GradientBoostingClassifier

#from sklearn.ensemble import ExtraTreesClassifier


def convert_age_to_class(n):

    if n <= 24:
        return "A"
    elif n <= 34:
        return "B"
    elif n <= 49:
        return "C"
    else:
        return "D"

def main():



    t0 = time.time()

## 1.b) Load and convert datasets                                              start
#
#    #Get cvs paths
#    relation_path = r"C:\\Users\\V\\Desktop\\UW Vidal\\Winter 18\\TCSS455 Introduction to Machine Learning\\Project\\training\\relation\\relation.csv"
#    profile_path = r"C:\\Users\\V\\Desktop\\UW Vidal\\Winter 18\\TCSS455 Introduction to Machine Learning\\Project\\training\\profile\\profile.csv"
#
#    #Convert csv into pandas DataFrame
#    relation_df = pd.read_csv(relation_path)
#    profile_df = pd.read_csv(profile_path)

# 2. Summarize Data ###########################################################
# 2.a) Descriptive statistics
#    print(profile_df.describe())
#    pd.set_option('display.width', 100)
#    pd.set_option('precision', 3)
#    correlations = profile_df.corr(method='pearson')
#    print(correlations)
#    profile_df.hist()
#    pyplot.show()
#
#
#
## 3. Prepare Data #############################################################
## a) Data Cleaning
## b) Feature Selection
## c) Data Transforms
#
#
#    userid_col = relation_df[['userid']]
#    row_counter = 1
#    num_users = 1
#    userid_dict = {}
#
#    #put all userids' in a dictionary
#    for index, row in userid_col.iterrows():
#        l = row.tolist()
#        userid = l[0].strip()
#        if (userid not in userid_dict):
#            userid_dict[userid] = ""
#            num_users += 1
#
#        row_counter += 1
#        if (row_counter < -2000): #15change
#            break
#
#
#    #relation_head = relation_df.head(2000) #25change
#    #profile_head = profile_df.head(2000) #35change
#    #head_profile.to_csv("head.csv", sep=',')
#
#
#    print("Here now")
#
#    #combine all likeids' associated with a userid
#    #make this the value of the userid in the dictionary
#    for index, row in relation_df.iterrows():#45change
#
#        row_list = row.tolist()
#
#        userid = str(row_list[1])
#        user_vals = userid_dict[userid]
#        userid_dict[userid] = user_vals + " " + str(row_list[2])
#
#    t_df = pd.DataFrame.from_dict(userid_dict, orient='index')
#    t_df = t_df.reset_index() ## remember to reassign when calling a function
#    t_df.columns = ["userid", "likes"]
#
#    merge_df = pd.merge(t_df, profile_df, on="userid") #55change                 end
    merge_df = pd.read_csv('merged.csv', sep=',')

#    pd.set_option('display.width', 100)
#    pd.set_option('precision', 3)
#    correlations = merge_df.corr(method='pearson')



# 4. Evaluate Algorithms ######################################################
# a) Split-out validation dataset
    X = merge_df['likes']
    y_gender = merge_df['gender']
    y_age = merge_df['age']

    #age convert
    y_age = y_age.apply(convert_age_to_class)
    y.to_csv("age_classified.csv", sep=',')

    #general algorithm
    valida_size = 0.20
    seed = 7
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=valida_size, random_state=seed)
#
## b) Test options and evaluation metric
#
    count_vect1 = CountVectorizer()
    X_train = count_vect1.fit_transform(X_train) #X_train is sparse.csr_matrix
##    print(type(X_train))
##    print(X_train.shape)
##    print(count_vect1.get_feature_names())
    count_vect2 = CountVectorizer(vocabulary=count_vect1.vocabulary_)
    X_validation = count_vect2.fit_transform(X_validation)
##    print(type(X_validation))
##    print(X_validation.shape)
#
#    kNN = KNeighborsClassifier()
#    bNB_clf = BernoulliNB()
#    dt_clf = DecisionTreeClassifier(random_state=0)
#    lr_clf = LogisticRegression(random_state=seed)
#    sgd_clf = linear_model.SGDClassifier(max_iter=10, learning_rate='optimal', random_state=seed)
#    mNB_clf = MultinomialNB()
#
#    enclf = VotingClassifier(estimators=[('dt', dt_clf),('lr', lr_clf), ('sgd', sgd_clf), ('mNB', mNB_clf)], voting='hard')
#
#    dt_clf.fit(X_train, y_train)
#    bNB_clf.fit(X_train, y_train)
#    lr_clf.fit(X_train, y_train)
#    sgd_clf.fit(X_train, y_train)
#    mNB_clf.fit(X_train, y_train)
#    enclf = enclf.fit(X_train, y_train)
#
#    print("Here")
#
#    results_dt = dt_clf.predict(X_validation)
#    results_bNB = bNB_clf.predict(X_validation)
#    results_lr = lr_clf.predict(X_validation)
#    results_sgd = sgd_clf.predict(X_validation)
#    results_nMB = mNB_clf.predict(X_validation)
#    results_enclf = enclf.predict(X_validation)
#
#    print("results_bNB")
#    print(accuracy_score(y_validation, results_dt))
#    print(confusion_matrix(y_validation, results_dt))
#    print(classification_report(y_validation, results_dt))
#    print("results_bNB")
#    print(accuracy_score(y_validation, results_bNB))
#    print(confusion_matrix(y_validation, results_bNB))
#    print(classification_report(y_validation, results_bNB))
#    print("results_lr")
#    print(accuracy_score(y_validation, results_lr))
#    print(confusion_matrix(y_validation, results_lr))
#    print(classification_report(y_validation, results_lr))
#    print("results_sgd")
#    print(accuracy_score(y_validation, results_sgd))
#    print(confusion_matrix(y_validation, results_sgd))
#    print(classification_report(y_validation, results_sgd))
#    print(results_lr)
#    print(results_sgd)
#    print()
#    print("results_mNB")
#    print(accuracy_score(y_validation, results_nMB))
#    print(confusion_matrix(y_validation, results_nMB))
#    print(classification_report(y_validation, results_nMB))
#
#    print("results_enclf")
#    print(accuracy_score(y_validation, results_enclf))
#    print(confusion_matrix(y_validation, results_enclf))
#    print(classification_report(y_validation, results_enclf))
## c) Spot Check Algorithms
#    models = []
#    models.append(('multiNB', MultinomialNB()))
#    models.append(('bernoulliNB', BernoulliNB()))
#    models.append(('kNN', KNeighborsClassifier()))
#    models.append(('LogReg', LogisticRegression()))
#    models.append(('SGD', linear_model.SGDClassifier(max_iter=10, learning_rate='optimal', random_state=seed)))
##
##
##    print()
#    print('Comparing Algorithms')
#    results = []
#    names = []
#    for name, model in models:
#        kfold = KFold(n_splits=10, random_state = seed)
#        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#        results.append(cv_results)
#        names.append(name)
#        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
#
## d) Compare Algorithms
#    fig = pyplot.figure()
#    fig.suptitle('Algorithm Comparison')
#    ax = fig.add_subplot(111)
#    pyplot.boxplot(results)
#    ax.set_xticklabels(names)
#    pyplot.show()   'learning_rate':('optimal', 'constant', 'invscaling')

    #'learning_rate':('optimal', 'constant', 'invscaling') 'learning_rate':('optimal', 'constant', 'invscaling')
    parameters = { 'max_iter':(1,5,10,20),'shuffle':(True, False) }
    sgd = linear_model.SGDClassifier()
    clf = GridSearchCV(sgd, parameters)
    clf.fit(X_train, y_train)
    print(clf.cv_results_)
#    lR = LogisticRegression()
#    lR.fit(X_train,y_train)
#    count_vect_val = CountVectorizer()
#    print("Here")
##    X_validation = count_vect_val.fit_transform(X_validation)
#    predictions = lR.predict(X_validation)
#    print(accuracy_score(y_validation, predictions))
#    print(confusion_matrix(y_validation, predictions))
#    print(classification_report(y_validation, predictions))
#
#    t1 = time.time()
#    print("\n\n--- %s seconds ---" % (t1-t0))
#    winsound.Beep(500,1000)
    print("Done")
    ###########################################################################
    # Training a Naive Bayes model
#count_vect = CountVectorizer()
#X_train = count_vect.fit_transform(data_train['transcripts'])
#y_train = data_train['gender']
#clf = MultinomialNB()
#clf.fit(X_train, data_train['gender'])

## Testing the Naive Bayes model
#X_test = count_vect.transform(data_test['transcripts'])
#y_test = data_test['gender']
#y_predicted = clf.predict(X_test)
## Reporting on classification performance
#print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
#classes = ['Male','Female']
#cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
#print("Confusion matrix:")
#print(cnf_matrix)

# =============================================================================
#     #variables for user features
#     male = 0asdf
#     female = 0
#
#     sumOfAges = 0
#     _xxTo24 = 0
#     _25To34 = 0
#     _35To49 = 0
#     _50Toxx = 0
#
#     openess = 0.0
#     consci = 0.0
#     extro = 0.0
#     agree = 0.0
#     emot = 0.0
# =============================================================================

    #f = open(argv[2], 'w')
# =============================================================================

#
#def outputXML(user):
#    f = open(user[1] + ".xml", 'w')
#    f.write("<user\n\tid=\"" + user[1] + "\"\n")
#    f.write("age_group=\"" + user[2] + "\"\n")
#    f.write("gender=\"" + user[3] + "\"\n")
#    f.write("extrovert=\"" + user[4] + "\"\n")
#    f.write("neurotic=\"" + user[5] + "\"\n")
#    f.write("agreeable=\"" + user[6] + "\"\n")
#    f.write("conscientious=\"" + user[7] + "\"\n")
#    f.write("open=\"" + user[8] + "\"\n")
#    f.write("/>")
#    f.close()
#
#    return


main()
