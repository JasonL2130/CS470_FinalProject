import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt



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

    # best for multi-label, not binary
    # OCD_rf = RandomForestClassifier(max_depth=19, min_samples_leaf=7, min_samples_split=5, n_estimators=15)
    OCD_rf = RandomForestClassifier(max_depth=31, min_samples_leaf=2, min_samples_split=6, n_estimators=53)


    OCD_rf.fit(OCD_xTrain, OCD_yTrain)
    OCD_yPred = OCD_rf.predict(OCD_xTest)
    print("length of yPred: ", len(OCD_yPred))
    OCD_yRand = np.random.randint(0, 4, size=145)

    # plot_model(OCD_yTest, OCD_yPred, OCD_yRand)


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

    # best for multi-label, not binary
    # insomnia_rf = RandomForestClassifier(max_depth=27, min_samples_leaf=8, min_samples_split=9, n_estimators=29)
    insomnia_rf = RandomForestClassifier(max_depth=13, min_samples_leaf=6, min_samples_split=7, n_estimators=27)

    insomnia_rf.fit(insomnia_xTrain, insomnia_yTrain)
    insomnia_yPred = insomnia_rf.predict(insomnia_xTest)

    print("accuracy score for insomnia: ", accuracy_score(insomnia_yTest, insomnia_yPred))
    print("precision score for insomnia: ", precision_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("recall score for insomnia: ", recall_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("mean squared error for insomnia: ", mean_squared_error(insomnia_yTest, insomnia_yPred))

    return accuracy_score(insomnia_yTest, insomnia_yPred)

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

    # best for multi-label, not binary
    # anxiety_rf = RandomForestClassifier(max_depth=33, min_samples_leaf=9, min_samples_split=7, n_estimators=15)
    anxiety_rf = RandomForestClassifier(max_depth=11, min_samples_leaf=3, min_samples_split=7, n_estimators=39)

    anxiety_rf.fit(anxiety_xTrain, anxiety_yTrain)
    anxiety_yPred = anxiety_rf.predict(anxiety_xTest)

    print("accuracy score for anxiety: ", accuracy_score(anxiety_yTest, anxiety_yPred))
    print("precision score for anxiety: ", precision_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("recall score for anxiety: ", recall_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("mean squared error for anxiety: ", mean_squared_error(anxiety_yTest, anxiety_yPred))

    return accuracy_score(anxiety_yTest, anxiety_yPred)

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

    # best for multi-label
    # depression_rf = RandomForestClassifier(max_depth=27, min_samples_leaf=9, min_samples_split=2, n_estimators=53)

    depression_rf = RandomForestClassifier(max_depth=25, min_samples_leaf=7, min_samples_split=9, n_estimators=39)

    depression_rf.fit(depression_xTrain, depression_yTrain)
    depression_yPred = depression_rf.predict(depression_xTest)

    print("accuracy score for depression: ", accuracy_score(depression_yTest, depression_yPred))
    print("precision score for depression: ", precision_score(depression_yTest, depression_yPred, average='weighted'))
    print("recall score for depression: ", recall_score(depression_yTest, depression_yPred, average='weighted'))
    print("mean squared error for depression: ", mean_squared_error(depression_yTest, depression_yPred))

    return accuracy_score(depression_yTest, depression_yPred)


############################################################

def plot_model(yTest, yPred, yRand):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr_list = []
    tpr_list = []
    roc_auc_list = []


    # mean_fpr = np.linspace(0, 1, 100)  # 100 points for the ROC curve
    # mean_tpr = 0.0

    # for i in range(4):  # Assuming there are 4 classes
    #     fpr[i], tpr[i], _ = roc_curve(yTest == i, yPred == i)
    #     print("fpr[i]: ", fpr[i])
    #     mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])
    #     print(mean_tpr)

    # mean_tpr /= 4  # Divide by the number of classes to get the mean

    # plt.plot(mean_fpr, mean_tpr, label='Mean ROC')

    # mean_fpr = np.linspace(0, 1, 100)  # 100 points for the ROC curve
    # mean_tpr = 0.0

    # for i in range(4):  # Assuming there are 4 classes
    #     fpr[i], tpr[i], _ = roc_curve(yTest == i, yRand == i)
    #     print("fpr[i]: ", fpr[i])
    #     mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])
    #     print(mean_tpr)

    # mean_tpr /= 4  # Divide by the number of classes to get the mean

    # plt.plot(mean_fpr, mean_tpr, label='Random ROC')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Mean ROC Curve')
    # plt.legend()
    # plt.show()

    fpr1, tpr1, _ = roc_curve(yTest, yPred, pos_label=0)

    # Calculate AUC
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(yTest, yPred, pos_label=1)
    roc_auc2 = auc(fpr2, tpr2)

    fpr3, tpr3, _ = roc_curve(yTest, yPred, pos_label=2)
    roc_auc3 = auc(fpr3, tpr3)

    fpr4, tpr4, _ = roc_curve(yTest, yPred, pos_label=3)
    roc_auc4 = auc(fpr4, tpr4)

    fpr5, tpr5, _ = roc_curve(yTest, yRand, pos_label=0)
    roc_auc5 = auc(fpr5, tpr5)

    fpr6, tpr6, _ = roc_curve(yTest, yRand, pos_label=1)
    roc_auc6 = auc(fpr6, tpr6)

    fpr7, tpr7, _ = roc_curve(yTest, yRand, pos_label=2)
    roc_auc7 = auc(fpr7, tpr7)

    fpr8, tpr8, _ = roc_curve(yTest, yRand, pos_label=3)
    roc_auc8 = auc(fpr8, tpr8)

    

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc2)
    plt.plot(fpr3, tpr3, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc3)
    plt.plot(fpr4, tpr4, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc4)
    plt.plot(fpr5, tpr5, color='grey', lw=2, label='ROC curve (area = %0.2f)' % roc_auc5)
    plt.plot(fpr6, tpr6, color='grey', lw=2, label='ROC curve (area = %0.2f)' % roc_auc6)
    plt.plot(fpr7, tpr7, color='grey', lw=2, label='ROC curve (area = %0.2f)' % roc_auc7)
    plt.plot(fpr4, tpr4, color='grey', lw=2, label='ROC curve (area = %0.2f)' % roc_auc8)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Class %d' % 2)
    plt.legend(loc="lower right")
    plt.show()





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

    OCD_xTrain = np.array(pd.read_csv("binary_ocd_train_xFeat.csv"))
    OCD_yTrain = np.array(pd.read_csv("binary_ocd_train_y.csv"))
    OCD_xTest = np.array(pd.read_csv("binary_ocd_test_xFeat.csv"))
    OCD_yTest = np.array(pd.read_csv("binary_ocd_test_y.csv"))
    insomnia_xTrain = np.array(pd.read_csv("binary_insomnia_train_xFeat.csv"))
    insomnia_yTrain = np.array(pd.read_csv("binary_insomnia_train_y.csv"))
    insomnia_xTest = np.array(pd.read_csv("binary_insomnia_test_xFeat.csv"))
    insomnia_yTest = np.array(pd.read_csv("binary_insomnia_test_y.csv"))
    anxiety_xTrain = np.array(pd.read_csv("binary_anxiety_train_xFeat.csv"))
    anxiety_yTrain = np.array(pd.read_csv("binary_anxiety_train_y.csv"))
    anxiety_xTest = np.array(pd.read_csv("binary_anxiety_test_xFeat.csv"))
    anxiety_yTest = np.array(pd.read_csv("binary_anxiety_test_y.csv"))
    depression_xTrain = np.array(pd.read_csv("binary_depression_train_xFeat.csv"))
    depression_yTrain = np.array(pd.read_csv("binary_depression_train_y.csv"))
    depression_xTest = np.array(pd.read_csv("binary_depression_test_xFeat.csv"))
    depression_yTest = np.array(pd.read_csv("binary_depression_test_y.csv"))

    OCD_yPred, _ = run_OCD_model(OCD_xTrain, OCD_xTest, OCD_yTrain, OCD_yTest)

    OCD_fpr, OCD_tpr, _ = roc_curve(OCD_yTest, OCD_yPred)
    OCD_roc_auc = roc_auc_score(OCD_yTest, OCD_yPred)

    df = pd.DataFrame({'FPR': OCD_fpr, 'TPR': OCD_tpr})

    # Write the DataFrame to a CSV file
    df.to_csv('random_forest_OCD_graph_data.csv', index=False)
    print("random forest ocd roc auc: ", OCD_roc_auc)

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

    # run_insomnia_model(insomnia_xTrain, insomnia_xTest, insomnia_yTrain, insomnia_yTest)

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

    # run_anxiety_model(anxiety_xTrain, anxiety_xTest, anxiety_yTrain, anxiety_yTest)

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

    # run_depression_model(depression_xTrain, depression_xTest, depression_yTrain, depression_yTest)

    # accuracy score for depression:  0.31724137931034485
    # precision score for depression:  0.45913369227850104
    # recall score for depression:  0.31724137931034485
    # mean squared error for depression:  1.193103448275862


    # gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain)

if __name__ == "__main__":
    main()