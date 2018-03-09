
#TCSS455 User Profiling
#@author: Vidal Sisneros vjsisneros@uw.edu


import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pickle import dump
from pickle import load


def create_xml(profile_id, gender, age, output_path):
    target_file = open(output_path + str(profile_id) + ".xml", "w")
    target_file.write("<user\n" +
        "  id=\"" + str(profile_id) + "\"\n" +
        "  age_group=\"" + str(age) + "\"\n" +
        "  gender=\"" + str(gender) + "\"\n" +
        "  extrovert=\"" + "3.486858" + "\"\n" +
        "  neurotic=\"" + "2.732424" + "\"\n" +
        "  agreeable=\"" + "3.583904" + "\"\n" +
        "  conscientious=\"" + "3.445617" + "\"\n" +
        "  open=\"" + "3.908691" + "\"\n" +
        "/>")

    target_file.close()

def convert_age_to_class(n):
    
    n = int(n)
    if n <= 24:
        return "xx-24"
    elif n <= 34:
        return "25-34"
    elif n <= 49:
        return "35-49"
    else:
        return "50-xx"

def parseArgs():

    input_path = ""
    output_path = ""
    
    for i in range(0, len(sys.argv)):
        if(sys.argv[i] == "-i"):
            input_path = sys.argv[i + 1]
        elif(sys.argv[i] == "-o"):
            output_path = sys.argv[i + 1]

    return input_path, output_path

def predict_gender_with_likes(input_path):

#    print("working")
#    input_path, output_path = parseArgs()

    profile_path = input_path + "profile/profile.csv"
    relation_path = input_path + "relation/relation.csv"
#    print(output_path)

    relation_df = pd.read_csv(relation_path)
    profile_df = pd.read_csv(profile_path)

    userid_col = relation_df[['userid']]
    print()
    row_counter = 1
    num_users = 1
    userid_dict = {}
    
    print("Put all userids' in a dictionary")
    #put all userids' in a dictionary
    for index, row in userid_col.iterrows():
        l = row.tolist()
        userid = l[0].strip()
        if (userid not in userid_dict):
            userid_dict[userid] = ""
            num_users += 1

        row_counter += 1

    print("Combining likeIds with userIDs")
    print()

    #combine all likeids' associated with a userid
    #make this the value of the userid in the dictionary
    for index, row in relation_df.iterrows():#45change

        row_list = row.tolist()
        userid = str(row_list[1])
        user_vals = userid_dict[userid]
        userid_dict[userid] = user_vals + " " + str(row_list[2])

#    print(len(userid_dict))

    t_df = pd.DataFrame.from_dict(userid_dict, orient='index')
    t_df = t_df.reset_index() ## remember to reassign when calling a function
    t_df.columns = ["userid", "likes"]

    merge_df = pd.merge(t_df, profile_df, on="userid") #55change
    
#    merge_df.to_csv("merging.csv", sep=',')
#    merge_df = pd.read_csv('merged_test.csv', sep=',')
#    merge_df.to_csv("merging.csv", sep=',')
    print()
    print("Predicting with likes...")
    X = merge_df['likes']
#    print(type(merge_df[['userid']]))
    userid_df = merge_df[['userid']]
    y_gender = merge_df['gender']
    y_age = merge_df['age']
    y_age = y_age.apply(convert_age_to_class)

    count_vectorizer = load(open('count_vectorizer.sav', 'rb'))
    X = count_vectorizer.fit_transform(X)

    filename_gender = 'gender_ensemble_clf.sav'
    ensemble_gender = load(open(filename_gender, 'rb'))

    filename_age = 'age_ensemble_clf.sav'
    ensemble_age = load(open(filename_age, 'rb'))

    predictions_gender = ensemble_gender.predict(X)
    predictions_age = ensemble_age.predict(X)
    
#    print(type(predictions_gender))
#    print(len(predictions_gender))
    p_gender_df = pd.DataFrame(predictions_gender)
    
    joined_df  = userid_df.join(p_gender_df)
    joined_df = joined_df.rename(columns={0: 'gender_likes'})
    print(type(joined_df))
    print(joined_df)
    
    return(joined_df)
    
#    count = 0
#    for index, row in merge_df.iterrows():
#
#        user_id = row[1]
#        gender_num = predictions_gender[count]
#        gender = ""
#        if(gender_num == 0.0):
#            gender = "male"
#        else:
#            gender = "female"
#
#        age = ""
#        age_char = predictions_age[count]
#        if age_char == "A":
#            age = "xx-24"
#        elif age_char == "B":
#            age = "25-34"
#        elif age_char == "C":
#            age = "35-49"
#        else:
#            age = "50-xx"

#        create_xml(user_id, gender, age, output_path)
#        count += 1
#        if(count > 5):
#            break

    print("Done") 
predict_gender_with_likes('C:/Users/admin/Desktop/UserProfilingRelation1.0/')
