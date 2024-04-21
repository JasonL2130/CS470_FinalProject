import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import numpy as np


############################# OCD MODEL AND RESULTS GO HERE ##############################
def run_OCD_model(OCD_train, OCD_test):

    OCD_xTrain = OCD_train[:, :-1]
    OCD_yTrain = OCD_train[:, -1]
    OCD_xTest = OCD_test[:, :-1]
    OCD_yTest = OCD_test[:, -1]

    OCD_rf = RandomForestClassifier(max_depth=19, min_samples_leaf=7, min_samples_split=5, n_estimators=15)

    OCD_rf.fit(OCD_xTrain, OCD_yTrain)
    OCD_yPred = OCD_rf.predict(OCD_xTest)

    print("accuracy score for OCD: ", accuracy_score(OCD_yTest, OCD_yPred))
    print("precision score for OCD: ", precision_score(OCD_yTest, OCD_yPred, average='weighted'))
    print("recall score for OCD: ", recall_score(OCD_yTest, OCD_yPred, average='weighted'))
    print("mean squared error for OCD: ", mean_squared_error(OCD_yTest, OCD_yPred))
    

    return accuracy_score(OCD_yTest, OCD_yPred)




######################################################


############################### INSOMNIA MODEL AND RESULTS GO HERE ########################
def run_insomnia_model(insomnia_train, insomnia_test):

    insomnia_xTrain = insomnia_train[:, :-1]
    insomnia_yTrain = insomnia_train[:, -1]
    insomnia_xTest = insomnia_test[:, :-1]
    insomnia_yTest = insomnia_test[:, -1]

    insomnia_rf = RandomForestClassifier(max_depth=27, min_samples_leaf=8, min_samples_split=9, n_estimators=29)

    insomnia_rf.fit(insomnia_xTrain, insomnia_yTrain)
    insomnia_yPred = insomnia_rf.predict(insomnia_xTest)

    print("accuracy score for insomnia: ", accuracy_score(insomnia_yTest, insomnia_yPred))
    print("precision score for insomnia: ", precision_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("recall score for insomnia: ", recall_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("mean squared error for insomnia: ", mean_squared_error(insomnia_yTest, insomnia_yPred))

    return accuracy_score(insomnia_yTest, insomnia_yPred)

##############################################################


################################ ANXIETY MODEL AND RESULTS GO HERE #############################
def run_anxiety_model(anxiety_train, anxiety_test):

    # anxiety_data = pd.read_csv("anxiety_data_name_")

    anxiety_xTrain = anxiety_train[:, :-1]
    anxiety_yTrain = anxiety_train[:, -1]
    anxiety_xTest = anxiety_test[:, :-1]
    anxiety_yTest = anxiety_test[:, -1]

    anxiety_rf = RandomForestClassifier(max_depth=33, min_samples_leaf=9, min_samples_split=7, n_estimators=15)

    anxiety_rf.fit(anxiety_xTrain, anxiety_yTrain)
    anxiety_yPred = anxiety_rf.predict(anxiety_xTest)

    print("accuracy score for anxiety: ", accuracy_score(anxiety_yTest, anxiety_yPred))
    print("precision score for anxiety: ", precision_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("recall score for anxiety: ", recall_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("mean squared error for anxiety: ", mean_squared_error(anxiety_yTest, anxiety_yPred))

    return accuracy_score(anxiety_yTest, anxiety_yPred)

##############################################################


################################## DEPRESSION MODEL AND RESULTS GO HERE ###############################
def run_depression_model(depression_train, depression_test):

    # depression_data = pd.read_csv("axitety_data_name_")

    depression_xTrain = depression_train[:, :-1]
    depression_yTrain = depression_train[:, -1]
    depression_xTest = depression_test[:, :-1]
    depression_yTest = depression_test[:, -1]

    depression_rf = RandomForestClassifier(max_depth=27, min_samples_leaf=9, min_samples_split=2, n_estimators=53) # made wih best parameters

    depression_rf.fit(depression_xTrain, depression_yTrain)
    depression_yPred = depression_rf.predict(depression_xTest)

    print("accuracy score for depression: ", accuracy_score(depression_yTest, depression_yPred))
    print("precision score for depression: ", precision_score(depression_yTest, depression_yPred, average='weighted'))
    print("recall score for depression: ", recall_score(depression_yTest, depression_yPred, average='weighted'))
    print("mean squared error for depression: ", mean_squared_error(depression_yTest, depression_yPred))

    return accuracy_score(depression_yTest, depression_yPred)

#######################################################

def gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain):
    param_grid = {
    'n_estimators': list(range(1, 55, 2)),
    'max_depth': list(range(1, 35, 2)),
    'min_samples_split': list(range(2, 10, 1)),
    'min_samples_leaf': list(range(1, 10, 1))
}

    OCD_rf = RandomForestClassifier(n_estimators=15)
    insomnia_rf = RandomForestClassifier(n_estimators=15)
    anxiety_rf = RandomForestClassifier(n_estimators=15)
    depression_rf = RandomForestClassifier(n_estimators=15)

    grid_search_OCD = GridSearchCV(OCD_rf, param_grid, cv=5, n_jobs=-1)
    grid_search_insomnia = GridSearchCV(insomnia_rf, param_grid, cv=5, n_jobs=-1)
    grid_search_anxiety = GridSearchCV(anxiety_rf, param_grid, cv=5, n_jobs=-1)
    grid_search_depression = GridSearchCV(depression_rf, param_grid, cv=5, n_jobs=-1)

    grid_search_OCD.fit(OCD_xTrain, OCD_yTrain)
    grid_search_insomnia.fit(insomnia_xTrain, insomnia_yTrain)
    grid_search_anxiety.fit(anxiety_xTrain, anxiety_yTrain)
    grid_search_depression.fit(depression_xTrain, depression_yTrain)


    print("Best OCD parameters:", grid_search_OCD.best_params_)
    # print("Best OCD estimator: ", grid_search_OCD.best_estimator_)
    print("Best OCD score:", grid_search_OCD.best_score_)

    print("Best insomnia parameters:", grid_search_insomnia.best_params_)
    # print("Best insomnia estimator: ", grid_search_insomnia.best_estimator_)
    print("Best insomnia score:", grid_search_insomnia.best_score_)

    print("Best anxiety parameters:", grid_search_anxiety.best_params_)
    # print("Best anxiety estimator: ", grid_search_anxiety.best_estimator_)
    print("Best anxiety score:", grid_search_anxiety.best_score_)

    print("Best depression parameters:", grid_search_depression.best_params_)
    # print("Best depression estimator: ", grid_search_depression.best_estimator_)
    print("Best depression score:", grid_search_depression.best_score_)

def main():

    OCD_train = np.array(pd.read_csv("ocd_train_final.csv"))
    OCD_test = np.array(pd.read_csv("ocd_test_final.csv"))

    OCD_xTrain = OCD_train[:, :-1]
    OCD_yTrain = OCD_train[:, -1]
    OCD_xTest = OCD_test[:, :-1]
    OCD_yTest = OCD_test[:, -1]

    run_OCD_model(OCD_train, OCD_test)

    # accuracy score for OCD:  0.47586206896551725
    # precision score for OCD:  0.3205159048382785
    # recall score for OCD:  0.47586206896551725
    # mean squared error for OCD:  1.7310344827586206

    insomnia_train = np.array(pd.read_csv("insomnia_train_final.csv"))
    insomnia_test = np.array(pd.read_csv("insomnia_test_final.csv"))

    insomnia_xTrain = insomnia_train[:, :-1]
    insomnia_yTrain = insomnia_train[:, -1]
    insomnia_xTest = insomnia_test[:, :-1]
    insomnia_yTest = insomnia_test[:, -1]

    run_insomnia_model(insomnia_train, insomnia_test)

    # accuracy score for insomnia:  0.23448275862068965
    # precision score for insomnia:  0.20484100207215344
    # recall score for insomnia:  0.23448275862068965
    # mean squared error for insomnia:  1.9724137931034482

    anxiety_train = np.array(pd.read_csv("anxiety_train_final.csv"))
    anxiety_test = np.array(pd.read_csv("anxiety_test_final.csv"))

    anxiety_xTrain = anxiety_train[:, :-1]
    anxiety_yTrain = anxiety_train[:, -1]
    anxiety_xTest = anxiety_test[:, :-1]
    anxiety_yTest = anxiety_test[:, -1]

    run_anxiety_model(anxiety_train, anxiety_test)

    # accuracy score for anxiety:  0.33793103448275863
    # precision score for anxiety:  0.30454859948666446
    # recall score for anxiety:  0.33793103448275863
    # mean squared error for anxiety:  1.3793103448275863

    depression_train = np.array(pd.read_csv("depression_train_final.csv"))
    depression_test = np.array(pd.read_csv("depression_test_final.csv"))

    depression_xTrain = depression_train[:, :-1]
    depression_yTrain = depression_train[:, -1]
    depression_xTest = depression_test[:, :-1]
    depression_yTest = depression_test[:, -1]

    run_depression_model(depression_train, depression_test)

    # accuracy score for depression:  0.31724137931034485
    # precision score for depression:  0.45913369227850104
    # recall score for depression:  0.31724137931034485
    # mean squared error for depression:  1.193103448275862


    # gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain)

if __name__ == "__main__":
    main()