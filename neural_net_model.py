import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, roc_auc_score, roc_curve
import numpy as np
# from knn_model import OCD_fpr, OCD_tpr, OCD_roc_auc

import matplotlib.pyplot as plt
# import seaborn as sns


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

    # optimal for label data (not binary)
    # OCD_nn = MLPClassifier(solver="sgd", learning_rate="adaptive", hidden_layer_sizes=(25, 25, 25), alpha=0.0001, activation="relu", max_iter=500, random_state=42, early_stopping=True)
    
    # optimal for binary data
    OCD_nn = MLPClassifier(solver="adam", learning_rate="constant", hidden_layer_sizes=(100,), alpha=0.001, activation="relu", random_state=42)

    OCD_nn.fit(OCD_xTrain, OCD_yTrain)
    OCD_yPred = OCD_nn.predict(OCD_xTest)

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

    # optimal for label data (not binary)
    # insomnia_nn = MLPClassifier(solver="sgd", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.0001, activation="tanh", max_iter=500, random_state=42, early_stopping=True)
    
    # optimal for binary data
    insomnia_nn = MLPClassifier(solver="adam", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.0001, activation="relu", random_state=42)

    insomnia_nn.fit(insomnia_xTrain, insomnia_yTrain)
    insomnia_yPred = insomnia_nn.predict(insomnia_xTest)

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

    # optimal for label data (not binary)
    # anxiety_nn = MLPClassifier(solver="sgd", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.0001, activation="relu", max_iter=500, random_state=42, early_stopping=True)

    # optimal for binary
    anxiety_nn = MLPClassifier(solver="adam", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.001, activation="relu", random_state=42)

    anxiety_nn.fit(anxiety_xTrain, anxiety_yTrain)
    anxiety_yPred = anxiety_nn.predict(anxiety_xTest)

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

    # optimal for label data (not binary)
    # depression_nn = MLPClassifier(solver="adam", learning_rate="constant", hidden_layer_sizes=(50, 50), alpha=0.001, activation="relu", max_iter=500, random_state=42, early_stopping=True)

    depression_nn = MLPClassifier(solver="adam", learning_rate="adaptive", hidden_layer_sizes=(100,), alpha=0.0001, activation="relu", random_state=42)

    depression_nn.fit(depression_xTrain, depression_yTrain)
    depression_yPred = depression_nn.predict(depression_xTest)

    print("accuracy score for depression: ", accuracy_score(depression_yTest, depression_yPred))
    print("precision score for depression: ", precision_score(depression_yTest, depression_yPred, average='weighted'))
    print("recall score for depression: ", recall_score(depression_yTest, depression_yPred, average='weighted'))
    print("mean squared error for depression: ", mean_squared_error(depression_yTest, depression_yPred))

    return depression_yPred, accuracy_score(depression_yTest, depression_yPred)

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

    OCD_xTrain = np.array(pd.read_csv("binary_ocd_train_xFeat.csv"))
    OCD_yTrain = np.array(pd.read_csv("binary_ocd_train_y.csv"))
    OCD_xTest = np.array(pd.read_csv("binary_ocd_test_xFeat.csv"))
    OCD_yTest = np.array(pd.read_csv("binary_ocd_test_y.csv"))
    

    OCD_yPred, _ = run_OCD_model(OCD_xTrain, OCD_xTest, OCD_yTrain, OCD_yTest)

    OCD_nn_fpr, OCD_nn_tpr, _ = roc_curve(OCD_yTest, OCD_yPred)
    OCD_nn_roc_auc = roc_auc_score(OCD_yTest, OCD_yPred)

    OCD_knn_roc_auc = 0.5127889818002952
    OCD_knn_fpr_tpr = pd.read_csv('knn_OCD_graph_data.csv')
    OCD_knn_fpr = OCD_knn_fpr_tpr.iloc[:, 0]
    OCD_knn_tpr = OCD_knn_fpr_tpr.iloc[:, 1]

    OCD_dt_roc_auc = 0.48413674372848003
    OCD_dt_fpr_tpr = pd.read_csv('dt_OCD_graph_data.csv')
    OCD_dt_fpr = OCD_dt_fpr_tpr.iloc[:, 0]
    OCD_dt_tpr = OCD_dt_fpr_tpr.iloc[:, 1]
    
    OCD_rf_roc_auc = 0.4757747171667487
    OCD_rf_fpr_tpr = pd.read_csv('random_forest_OCD_graph_data.csv')
    OCD_rf_fpr = OCD_rf_fpr_tpr.iloc[:, 0]
    OCD_rf_tpr = OCD_rf_fpr_tpr.iloc[:, 1]




    print(OCD_knn_fpr_tpr)
    print(OCD_knn_fpr)

    

    # print(OCD_roc_auc)

    plt.figure()
    plt.plot(OCD_nn_fpr, OCD_nn_tpr, color='darkorange', lw=2, label='NN ROC curve (area = %0.2f)' % OCD_nn_roc_auc)
    plt.plot(OCD_knn_fpr, OCD_knn_tpr, color='blue', lw=2, label='KNN ROC curve (area = %0.2f)' % OCD_knn_roc_auc)
    plt.plot(OCD_dt_fpr, OCD_dt_tpr, color='brown', lw=2, label='DT ROC curve (area = %0.2f)' % OCD_dt_roc_auc)
    plt.plot(OCD_rf_fpr, OCD_rf_tpr, color='brown', lw=2, label='Random Forest ROC curve (area = %0.2f)' % OCD_rf_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves for OCD')
    plt.legend(loc="lower right")
    plt.show()



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

    insomnia_xTrain = np.array(pd.read_csv("binary_insomnia_train_xFeat.csv"))
    insomnia_yTrain = np.array(pd.read_csv("binary_insomnia_train_y.csv"))
    insomnia_xTest = np.array(pd.read_csv("binary_insomnia_test_xFeat.csv"))
    insomnia_yTest = np.array(pd.read_csv("binary_insomnia_test_y.csv"))
    

    insomnia_yPred, _ = run_insomnia_model(insomnia_xTrain, insomnia_xTest, insomnia_yTrain, insomnia_yTest)
    
    fpr, tpr, _ = roc_curve(insomnia_yTest, insomnia_yPred)
    roc_auc = roc_auc_score(insomnia_yTest, insomnia_yPred)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for insomnia NN')
    plt.legend(loc="lower right")
    plt.show()


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

    anxiety_xTrain = np.array(pd.read_csv("binary_anxiety_train_xFeat.csv"))
    anxiety_yTrain = np.array(pd.read_csv("binary_anxiety_train_y.csv"))
    anxiety_xTest = np.array(pd.read_csv("binary_anxiety_test_xFeat.csv"))
    anxiety_yTest = np.array(pd.read_csv("binary_anxiety_test_y.csv"))

    anxiety_yPred, _ = run_anxiety_model(anxiety_xTrain, anxiety_xTest, anxiety_yTrain, anxiety_yTest)

    fpr, tpr, _ = roc_curve(anxiety_yTest, anxiety_yPred)
    roc_auc = roc_auc_score(anxiety_yTest, anxiety_yPred)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for anxiety NN')
    plt.legend(loc="lower right")
    plt.show()

    # accuracy score for anxiety:  0.3103448275862069
    # precision score for anxiety:  0.2978311663840211
    # recall score for anxiety:  0.3103448275862069
    # mean squared error for anxiety:  1.8482758620689654

    # depression_train = np.array(pd.read_csv("depression_train_final.csv"))
    # depression_test = np.array(pd.read_csv("depression_test_final.csv"))

    # depression_xTrain = depression_train[:, :-1]
    # depression_yTrain = depression_train[:, -1]
    # depression_xTest = depression_test[:, :-1]
    # depression_yTest = depression_test[:, -1]

    depression_xTrain = np.array(pd.read_csv("binary_depression_train_xFeat.csv"))
    depression_yTrain = np.array(pd.read_csv("binary_depression_train_y.csv"))
    depression_xTest = np.array(pd.read_csv("binary_depression_test_xFeat.csv"))
    depression_yTest = np.array(pd.read_csv("binary_depression_test_y.csv"))

    depression_yPred, _ = run_depression_model(depression_xTrain, depression_xTest, depression_yTrain, depression_yTest)

    fpr, tpr, _ = roc_curve(depression_yTest, depression_yPred)
    roc_auc = roc_auc_score(depression_yTest, depression_yPred)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for depression NN')
    plt.legend(loc="lower right")
    plt.show()

    # accuracy score for depression:  0.32413793103448274
    # precision score for depression:  0.3345493919706813
    # recall score for depression:  0.3103448275862069
    # mean squared error for depression:  1.6758620689655173

    

    # gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain)

    models = ['KNN', 'Decision Tree', 'Random Forest', 'Neural Network']

    # List of accuracy scores for each model
    OCD_accuracy_scores = [0.496551724137931, 0.43275862068965515, 0.48793103448275865, 0.4637931034482759]

    insomnia_accuracy_scores = [0.3448275862068966, 0.3362068965517241, 0.3931034482758621, 0.3017241379310345]

    anxiety_accuracy_scores = [0.41206896551724137, 0.37413793103448284, 0.43103448275862066, 0.3844827586206897]

    depression_accuracy_scores = [0.3431034482758621, 0.3362068965517241, 0.3931034482758621, 0.31551724137931036]

    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=models, y=OCD_accuracy_scores, palette='viridis')
    # plt.ylim(0.0, 0.7)  # Set the y-axis limits to better visualize differences
    # plt.axhline(y=0.25, color='r', linestyle='--', label='Random Guess')
    # # plt.text(3.5, 0.26, 'Random Guess', color='red', fontsize=12)
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy of Classification Models Predicting OCD Severity')
    # plt.show()


    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=models, y=insomnia_accuracy_scores, palette='viridis')
    # plt.ylim(0.0, 0.7)  # Set the y-axis limits to better visualize differences
    # plt.axhline(y=0.25, color='r', linestyle='--', label='Random Guess')
    # # plt.text(-0.3, 0.22, 'Random Guess', color='red', fontsize=12)
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy of Classification Models Predicting Insomnia Severity')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=models, y=anxiety_accuracy_scores, palette='viridis')
    # plt.ylim(0.0, 0.7)  # Set the y-axis limits to better visualize differences
    # plt.axhline(y=0.25, color='r', linestyle='--', label='Random Guess')
    # # plt.text(-0.3, 0.22, 'Random Guess', color='red', fontsize=12)
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy of Classification Models Predicting Anxiety Severity')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=models, y=depression_accuracy_scores, palette='viridis')
    # plt.ylim(0.0, 0.7)  # Set the y-axis limits to better visualize differences
    # plt.axhline(y=0.25, color='r', linestyle='--', label='Random Guess')
    # # plt.text(-0.3, 0.22, 'Random Guess', color='red', fontsize=12)
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy of Classification Models Predicting Depression Severity')
    # plt.show()



if __name__ == "__main__":
    main()