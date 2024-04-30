import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, roc_curve, roc_auc_score, confusion_matrix
import numpy as np


############################# OCD MODEL AND RESULTS GO HERE ##############################
def run_OCD_model(OCD_xTrain, OCD_xTest, OCD_yTrain, OCD_yTest):

    # OCD_xTrain = OCD_train[:, :-1]
    # OCD_yTrain = OCD_train[:, -1]
    # OCD_xTest = OCD_test[:, :-1]
    # OCD_yTest = OCD_test[:, -1]
    OCD_xTrain = OCD_xTrain
    OCD_xTest = OCD_xTest
    OCD_yTrain = OCD_yTrain
    OCD_yTest = OCD_yTest

    OCD_dt = DecisionTreeClassifier(criterion='gini', max_depth=50, max_features="log2", min_samples_leaf=1, min_samples_split=2)

    OCD_dt.fit(OCD_xTrain, OCD_yTrain)
    OCD_yPred = OCD_dt.predict(OCD_xTest)

    print("accuracy score for OCD: ", accuracy_score(OCD_yTest, OCD_yPred))
    print("precision score for OCD: ", precision_score(OCD_yTest, OCD_yPred, average='weighted'))
    print("recall score for OCD: ", recall_score(OCD_yTest, OCD_yPred, average='weighted'))
    print("mean squared error for OCD: ", mean_squared_error(OCD_yTest, OCD_yPred))
    

    return OCD_yPred, accuracy_score(OCD_yTest, OCD_yPred)




######################################################


############################### INSOMNIA MODEL AND RESULTS GO HERE ########################
def run_insomnia_model(insomnia_xTrain, insomnia_xTest, insomnia_yTrain, insomnia_yTest):

    # insomnia_xTrain = insomnia_train[:, :-1]
    # insomnia_yTrain = insomnia_train[:, -1]
    # insomnia_xTest = insomnia_test[:, :-1]
    # insomnia_yTest = insomnia_test[:, -1]

    insomnia_xTrain = insomnia_xTrain
    insomnia_xTest = insomnia_xTest
    insomnia_yTrain = insomnia_yTrain
    insomnia_yTest = insomnia_yTest

    insomnia_dt = DecisionTreeClassifier(criterion="entropy", max_depth=20, max_features="sqrt", min_samples_leaf=2, min_samples_split=5)

    insomnia_dt.fit(insomnia_xTrain, insomnia_yTrain)
    insomnia_yPred = insomnia_dt.predict(insomnia_xTest)

    print("accuracy score for insomnia: ", accuracy_score(insomnia_yTest, insomnia_yPred))
    print("precision score for insomnia: ", precision_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("recall score for insomnia: ", recall_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("mean squared error for insomnia: ", mean_squared_error(insomnia_yTest, insomnia_yPred))

    return insomnia_yPred, accuracy_score(insomnia_yTest, insomnia_yPred)

##############################################################


################################ ANXIETY MODEL AND RESULTS GO HERE #############################
def run_anxiety_model(anxiety_xTrain, anxiety_xTest, anxiety_yTrain, anxiety_yTest):

    # anxiety_data = pd.read_csv("anxiety_data_name_")

    # anxiety_xTrain = anxiety_train[:, :-1]
    # anxiety_yTrain = anxiety_train[:, -1]
    # anxiety_xTest = anxiety_test[:, :-1]
    # anxiety_yTest = anxiety_test[:, -1]

    anxiety_xTrain = anxiety_xTrain
    anxiety_yTrain = anxiety_yTrain
    anxiety_xTest = anxiety_xTest
    anxiety_yTest = anxiety_yTest

    anxiety_dt = DecisionTreeClassifier(criterion="entropy", max_depth=40, max_features="log2", min_samples_leaf=1, min_samples_split=2)

    anxiety_dt.fit(anxiety_xTrain, anxiety_yTrain)
    anxiety_yPred = anxiety_dt.predict(anxiety_xTest)

    print("accuracy score for anxiety: ", accuracy_score(anxiety_yTest, anxiety_yPred))
    print("precision score for anxiety: ", precision_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("recall score for anxiety: ", recall_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("mean squared error for anxiety: ", mean_squared_error(anxiety_yTest, anxiety_yPred))

    return anxiety_yPred, accuracy_score(anxiety_yTest, anxiety_yPred)

##############################################################


################################## DEPRESSION MODEL AND RESULTS GO HERE ###############################
def run_depression_model(depression_xTrain, depression_xTest, depression_yTrain, depression_yTest):

    # depression_data = pd.read_csv("axitety_data_name_")

    # depression_xTrain = depression_train[:, :-1]
    # depression_yTrain = depression_train[:, -1]
    # depression_xTest = depression_test[:, :-1]
    # depression_yTest = depression_test[:, -1]

    depression_xTrain = depression_xTrain
    depression_xTest = depression_xTest
    depression_yTrain = depression_yTrain
    depression_yTest = depression_yTest

    depression_dt = DecisionTreeClassifier(criterion="entropy", max_depth=40, max_features="sqrt", min_samples_leaf=1, min_samples_split=2)

    depression_dt.fit(depression_xTrain, depression_yTrain)
    depression_yPred = depression_dt.predict(depression_xTest)

    print("accuracy score for depression: ", accuracy_score(depression_yTest, depression_yPred))
    print("precision score for depression: ", precision_score(depression_yTest, depression_yPred, average='weighted'))
    print("recall score for depression: ", recall_score(depression_yTest, depression_yPred, average='weighted'))
    print("mean squared error for depression: ", mean_squared_error(depression_yTest, depression_yPred))

    return depression_yPred, accuracy_score(depression_yTest, depression_yPred)

#######################################################

def gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain):
    param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }

    OCD_dt = DecisionTreeClassifier()
    insomnia_dt = DecisionTreeClassifier()
    anxiety_dt = DecisionTreeClassifier()
    depression_dt = DecisionTreeClassifier()

    grid_search_OCD = GridSearchCV(OCD_dt, param_grid, cv=5, n_jobs=-1)
    grid_search_insomnia = GridSearchCV(insomnia_dt, param_grid, cv=5, n_jobs=-1)
    grid_search_anxiety = GridSearchCV(anxiety_dt, param_grid, cv=5, n_jobs=-1)
    grid_search_depression = GridSearchCV(depression_dt, param_grid, cv=5, n_jobs=-1)

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

    OCD_xTrain = np.array(pd.read_csv("binary_ocd_train_xFeat.csv"))
    OCD_yTrain = np.array(pd.read_csv("binary_ocd_train_y.csv"))
    OCD_xTest = np.array(pd.read_csv("binary_ocd_test_xFeat.csv"))
    OCD_yTest = np.array(pd.read_csv("binary_ocd_test_y.csv"))

    OCD_yPred, _ = run_OCD_model(OCD_xTrain, OCD_xTest, OCD_yTrain, OCD_yTest)

    OCD_fpr, OCD_tpr, _ = roc_curve(OCD_yTest, OCD_yPred)
    OCD_roc_auc = roc_auc_score(OCD_yTest, OCD_yPred)

    # print("ocd roc: ", OCD_roc_auc)
    # df = pd.DataFrame({'FPR': OCD_fpr, 'TPR': OCD_tpr})

    # # Write the DataFrame to a CSV file
    # df.to_csv('dt_OCD_graph_data.csv', index=False)

    matrix = confusion_matrix(OCD_yTest, OCD_yPred)

    print("OCD DT Confusion Matrix:")
    print(matrix)


    insomnia_train = np.array(pd.read_csv("insomnia_train_final.csv"))
    insomnia_test = np.array(pd.read_csv("insomnia_test_final.csv"))

    insomnia_xTrain = insomnia_train[:, :-1]
    insomnia_yTrain = insomnia_train[:, -1]
    insomnia_xTest = insomnia_test[:, :-1]
    insomnia_yTest = insomnia_test[:, -1]

    insomnia_xTrain = np.array(pd.read_csv("binary_insomnia_train_xFeat.csv"))
    insomnia_yTrain = np.array(pd.read_csv("binary_insomnia_train_y.csv"))
    insomnia_xTest = np.array(pd.read_csv("binary_insomnia_test_xFeat.csv"))
    insomnia_yTest = np.array(pd.read_csv("binary_insomnia_test_y.csv"))

    insomnia_yPred, _ = run_insomnia_model(insomnia_xTrain, insomnia_xTest, insomnia_yTrain, insomnia_yTest)

    insomnia_fpr, insomnia_tpr, _ = roc_curve(insomnia_yTest, insomnia_yPred)
    insomnia_roc_auc = roc_auc_score(insomnia_yTest, insomnia_yPred)

    df = pd.DataFrame({'FPR': insomnia_fpr, 'TPR': insomnia_tpr})

    # # Write the DataFrame to a CSV file
    # df.to_csv('dt_insomnia_graph_data.csv', index=False)

    # print("dt insomnia roc score", insomnia_roc_auc)

    matrix = confusion_matrix(insomnia_yTest, insomnia_yPred)

    print("Insomnia DT Confusion Matrix:")
    print(matrix)


    anxiety_train = np.array(pd.read_csv("anxiety_train_final.csv"))
    anxiety_test = np.array(pd.read_csv("anxiety_test_final.csv"))

    anxiety_xTrain = anxiety_train[:, :-1]
    anxiety_yTrain = anxiety_train[:, -1]
    anxiety_xTest = anxiety_test[:, :-1]
    anxiety_yTest = anxiety_test[:, -1]

    anxiety_xTrain = np.array(pd.read_csv("binary_anxiety_train_xFeat.csv"))
    anxiety_yTrain = np.array(pd.read_csv("binary_anxiety_train_y.csv"))
    anxiety_xTest = np.array(pd.read_csv("binary_anxiety_test_xFeat.csv"))
    anxiety_yTest = np.array(pd.read_csv("binary_anxiety_test_y.csv"))

    anxiety_yPred, _ = run_anxiety_model(anxiety_xTrain, anxiety_xTest, anxiety_yTrain, anxiety_yTest)

    anxiety_fpr, anxiety_tpr, _ = roc_curve(anxiety_yTest, anxiety_yPred)
    anxiety_roc_auc = roc_auc_score(anxiety_yTest, anxiety_yPred)

    # df = pd.DataFrame({'FPR': anxiety_fpr, 'TPR': anxiety_tpr})

    # # Write the DataFrame to a CSV file
    # df.to_csv('dt_anxiety_graph_data.csv', index=False)

    # print("dt anxiety roc score", anxiety_roc_auc)

    matrix = confusion_matrix(anxiety_yTest, anxiety_yPred)

    print("anxiety DT Confusion Matrix:")
    print(matrix)

    depression_train = np.array(pd.read_csv("depression_train_final.csv"))
    depression_test = np.array(pd.read_csv("depression_test_final.csv"))

    depression_xTrain = depression_train[:, :-1]
    depression_yTrain = depression_train[:, -1]
    depression_xTest = depression_test[:, :-1]
    depression_yTest = depression_test[:, -1]

    depression_xTrain = np.array(pd.read_csv("binary_depression_train_xFeat.csv"))
    depression_yTrain = np.array(pd.read_csv("binary_depression_train_y.csv"))
    depression_xTest = np.array(pd.read_csv("binary_depression_test_xFeat.csv"))
    depression_yTest = np.array(pd.read_csv("binary_depression_test_y.csv"))

    depression_yPred, _ = run_depression_model(depression_xTrain, depression_xTest, depression_yTrain, depression_yTest)

    depression_fpr, depression_tpr, _ = roc_curve(depression_yTest, depression_yPred)
    depression_roc_auc = roc_auc_score(depression_yTest, depression_yPred)

    df = pd.DataFrame({'FPR': depression_fpr, 'TPR': depression_tpr})

    # # Write the DataFrame to a CSV file
    # df.to_csv('dt_depression_graph_data.csv', index=False)

    # print("dt depression roc score", depression_roc_auc)

    matrix = confusion_matrix(depression_yTest, depression_yPred)

    print("Depression DT Confusion Matrix:")
    print(matrix)

    # gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain)

if __name__ == "__main__":
    main()