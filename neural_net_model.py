import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, roc_auc_score
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


############################# OCD MODEL AND RESULTS GO HERE ##############################
def run_OCD_model(OCD_train, OCD_test):

    OCD_xTrain = OCD_train[:, :-1]
    OCD_yTrain = OCD_train[:, -1]
    OCD_xTest = OCD_test[:, :-1]
    OCD_yTest = OCD_test[:, -1]

    OCD_nn = MLPClassifier(solver="sgd", learning_rate="adaptive", hidden_layer_sizes=(25, 25, 25), alpha=0.0001, activation="relu", max_iter=500, random_state=42, early_stopping=True)

    OCD_nn.fit(OCD_xTrain, OCD_yTrain)
    OCD_yPred = OCD_nn.predict(OCD_xTest)

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

    insomnia_nn = MLPClassifier(solver="sgd", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.0001, activation="tanh", max_iter=500, random_state=42, early_stopping=True)

    insomnia_nn.fit(insomnia_xTrain, insomnia_yTrain)
    insomnia_yPred = insomnia_nn.predict(insomnia_xTest)

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

    anxiety_nn = MLPClassifier(solver="sgd", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.0001, activation="relu", max_iter=500, random_state=42, early_stopping=True)

    anxiety_nn.fit(anxiety_xTrain, anxiety_yTrain)
    anxiety_yPred = anxiety_nn.predict(anxiety_xTest)

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

    depression_nn = MLPClassifier(solver="adam", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.001, activation="relu", max_iter=500, random_state=42, early_stopping=True)

    depression_nn.fit(depression_xTrain, depression_yTrain)
    depression_yPred = depression_nn.predict(depression_xTest)

    print("accuracy score for depression: ", accuracy_score(depression_yTest, depression_yPred))
    print("precision score for depression: ", precision_score(depression_yTest, depression_yPred, average='weighted'))
    print("recall score for depression: ", recall_score(depression_yTest, depression_yPred, average='weighted'))
    print("mean squared error for depression: ", mean_squared_error(depression_yTest, depression_yPred))

    return accuracy_score(depression_yTest, depression_yPred)

#######################################################

def gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain):
    param_dist = {
    'hidden_layer_sizes': [(100,), (50, 50), (25, 25, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}

    OCD_nn = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)
    insomnia_nn = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)
    anxiety_nn = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)
    depression_nn = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)

    grid_search_OCD = RandomizedSearchCV(estimator=OCD_nn, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
    grid_search_insomnia = RandomizedSearchCV(estimator=insomnia_nn, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
    grid_search_anxiety = RandomizedSearchCV(estimator=anxiety_nn, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
    grid_search_depression = RandomizedSearchCV(estimator=depression_nn, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)

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

    # accuracy score for OCD:  0.3724137931034483
    # precision score for OCD:  0.34326431163371735
    # recall score for OCD:  0.3724137931034483
    # mean squared error for OCD:  1.7241379310344827

    insomnia_train = np.array(pd.read_csv("insomnia_train_final.csv"))
    insomnia_test = np.array(pd.read_csv("insomnia_test_final.csv"))

    insomnia_xTrain = insomnia_train[:, :-1]
    insomnia_yTrain = insomnia_train[:, -1]
    insomnia_xTest = insomnia_test[:, :-1]
    insomnia_yTest = insomnia_test[:, -1]

    run_insomnia_model(insomnia_train, insomnia_test)

    # accuracy score for insomnia:  0.2896551724137931
    # precision score for insomnia:  0.2883202319812738
    # recall score for insomnia:  0.2896551724137931
    # mean squared error for insomnia:  2.503448275862069


    anxiety_train = np.array(pd.read_csv("anxiety_train_final.csv"))
    anxiety_test = np.array(pd.read_csv("anxiety_test_final.csv"))

    anxiety_xTrain = anxiety_train[:, :-1]
    anxiety_yTrain = anxiety_train[:, -1]
    anxiety_xTest = anxiety_test[:, :-1]
    anxiety_yTest = anxiety_test[:, -1]

    run_anxiety_model(anxiety_train, anxiety_test)

    # accuracy score for anxiety:  0.3103448275862069
    # precision score for anxiety:  0.2978311663840211
    # recall score for anxiety:  0.3103448275862069
    # mean squared error for anxiety:  1.8482758620689654

    depression_train = np.array(pd.read_csv("depression_train_final.csv"))
    depression_test = np.array(pd.read_csv("depression_test_final.csv"))

    depression_xTrain = depression_train[:, :-1]
    depression_yTrain = depression_train[:, -1]
    depression_xTest = depression_test[:, :-1]
    depression_yTest = depression_test[:, -1]

    run_depression_model(depression_train, depression_test)

    # accuracy score for depression:  0.32413793103448274
    # precision score for depression:  0.3345493919706813
    # recall score for depression:  0.3103448275862069
    # mean squared error for depression:  1.6758620689655173

    # gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain)

    models = ['KNN', 'Decision Tree', 'Random Forest', 'Neural Network', 'Random Guess']

    # List of accuracy scores for each model
    accuracy_scores = [0.296551724137931, 0.3568965517241379, 0.31724137931034485, 0.32413793103448274, 0.25]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracy_scores, palette='viridis')
    plt.ylim(0.0, 0.7)  # Set the y-axis limits to better visualize differences
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Classification Models Predicting Depression Severity')
    plt.show()



if __name__ == "__main__":
    main()