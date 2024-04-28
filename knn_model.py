import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



############################# OCD MODEL AND RESULTS GO HERE ##############################
def run_OCD_model(OCD_xTrain, OCD_xTest, OCD_yTrain, OCD_yTest, k, w):

    # OCD_xTrain = OCD_train[:, :-1]
    # OCD_yTrain = OCD_train[:, -1]
    # OCD_xTest = OCD_test[:, :-1]
    # OCD_yTest = OCD_test[:, -1]

    OCD_xTrain = OCD_xTrain
    OCD_xTest = OCD_xTest
    OCD_yTrain = OCD_yTrain
    OCD_yTest = OCD_yTest
    print(len(OCD_yTest))



    OCD_knn = KNeighborsClassifier(n_neighbors=k, weights=w)

    OCD_knn.fit(OCD_xTrain, OCD_yTrain)
    OCD_yPred = OCD_knn.predict(OCD_xTest)

    print("accuracy score for OCD: ", accuracy_score(OCD_yTest, OCD_yPred))
    print("precision score for OCD: ", precision_score(OCD_yTest, OCD_yPred, average='weighted'))
    print("recall score for OCD: ", recall_score(OCD_yTest, OCD_yPred, average='weighted'))
    print("mean squared error for OCD: ", mean_squared_error(OCD_yTest, OCD_yPred))
    

    return OCD_yPred, accuracy_score(OCD_yTest, OCD_yPred)



######################################################


############################### INSOMNIA MODEL AND RESULTS GO HERE ########################
def run_insomnia_model(insomnia_xTrain, insomnia_xTest, insomnia_yTrain, insomnia_yTest, k, w):

    # insomnia_xTrain = insomnia_train[:, :-1]
    # insomnia_yTrain = insomnia_train[:, -1]
    # insomnia_xTest = insomnia_test[:, :-1]
    # insomnia_yTest = insomnia_test[:, -1]

    insomnia_xTrain = insomnia_xTrain
    insomnia_xTest = insomnia_xTest
    insomnia_yTrain = insomnia_yTrain
    insomnia_yTest = insomnia_yTest
    

    insomnia_knn = KNeighborsClassifier(n_neighbors=k, weights=w)

    insomnia_knn.fit(insomnia_xTrain, insomnia_yTrain)
    insomnia_yPred = insomnia_knn.predict(insomnia_xTest)

    print("accuracy score for insomnia: ", accuracy_score(insomnia_yTest, insomnia_yPred))
    print("precision score for insomnia: ", precision_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("recall score for insomnia: ", recall_score(insomnia_yTest, insomnia_yPred, average='weighted'))
    print("mean squared error for insomnia: ", mean_squared_error(insomnia_yTest, insomnia_yPred))

    return insomnia_yPred, accuracy_score(insomnia_yTest, insomnia_yPred)

##############################################################


################################ ANXIETY MODEL AND RESULTS GO HERE #############################
def run_anxiety_model(anxiety_xTrain, anxiety_xTest, anxiety_yTrain, anxiety_yTest, k, w):

    # anxiety_data = pd.read_csv("anxiety_data_name_")

    # anxiety_xTrain = anxiety_train[:, :-1]
    # anxiety_yTrain = anxiety_train[:, -1]
    # anxiety_xTest = anxiety_test[:, :-1]
    # anxiety_yTest = anxiety_test[:, -1]

    anxiety_xTrain = anxiety_xTrain
    anxiety_yTrain = anxiety_yTrain
    anxiety_xTest = anxiety_xTest
    anxiety_yTest = anxiety_yTest

    anxiety_knn = KNeighborsClassifier(n_neighbors=k, weights=w)

    anxiety_knn.fit(anxiety_xTrain, anxiety_yTrain)
    anxiety_yPred = anxiety_knn.predict(anxiety_xTest)

    print("accuracy score for anxiety: ", accuracy_score(anxiety_yTest, anxiety_yPred))
    print("precision score for anxiety: ", precision_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("recall score for anxiety: ", recall_score(anxiety_yTest, anxiety_yPred, average='weighted'))
    print("mean squared error for anxiety: ", mean_squared_error(anxiety_yTest, anxiety_yPred))

    return accuracy_score(anxiety_yTest, anxiety_yPred)

##############################################################


################################## DEPRESSION MODEL AND RESULTS GO HERE ###############################
def run_depression_model(depression_xTrain, depression_xTest, depression_yTrain, depression_yTest, k, w):

    # depression_data = pd.read_csv("axitety_data_name_")

    # depression_xTrain = depression_train[:, :-1]
    # depression_yTrain = depression_train[:, -1]
    # depression_xTest = depression_test[:, :-1]
    # depression_yTest = depression_test[:, -1]

    depression_xTrain = depression_xTrain
    depression_xTest = depression_xTest
    depression_yTrain = depression_yTrain
    depression_yTest = depression_yTest

    depression_knn = KNeighborsClassifier(n_neighbors=k, weights=w)

    depression_knn.fit(depression_xTrain, depression_yTrain)
    depression_yPred = depression_knn.predict(depression_xTest)

    print("accuracy score for depression: ", accuracy_score(depression_yTest, depression_yPred))
    print("precision score for depression: ", precision_score(depression_yTest, depression_yPred, average='weighted'))
    print("recall score for depression: ", recall_score(depression_yTest, depression_yPred, average='weighted'))
    print("mean squared error for depression: ", mean_squared_error(depression_yTest, depression_yPred))

    return accuracy_score(depression_yTest, depression_yPred)

#######################################################


#################################### FINDING OPTIMAL NUMBER OF NEIGHBORS IS DONE HERE ###################################

def optimal_k_value(OCD_train, OCD_test, insomnia_train, insomnia_test, anxiety_train, anxiety_test, depression_train, depression_test):

    accuracy_score_list = []
    OCD_accuracy_score_list = []
    insomnia_accuracy_score_list = []
    anxiety_accuracy_score_list = []
    depression_accuracy_score_list = []
    for k in range(1,100):
        OCD_accuracy_score = run_OCD_model(OCD_train, OCD_test, k)
        OCD_accuracy_score_list.append(OCD_accuracy_score)
        insomnia_accuracy_score = run_insomnia_model(insomnia_train, insomnia_test, k)
        insomnia_accuracy_score_list.append(insomnia_accuracy_score)
        anxiety_accuracy_score = run_anxiety_model(anxiety_train, anxiety_test, k)
        anxiety_accuracy_score_list.append(anxiety_accuracy_score)
        depression_accuracy_score = run_depression_model(depression_train, depression_test, k)
        depression_accuracy_score_list.append(depression_accuracy_score)

        accuracy_score_list.append(np.mean([OCD_accuracy_score, insomnia_accuracy_score, anxiety_accuracy_score, depression_accuracy_score]))
    
    for i in range(len(accuracy_score_list)):
        if accuracy_score_list[i] == max(accuracy_score_list):
            print("The best k value is ", i, " with a mean accuracy of ", accuracy_score_list[i])
        if OCD_accuracy_score_list[i] == max(OCD_accuracy_score_list):
            print("The best k value for OCD is ", i, " with a mean accuracy of ", OCD_accuracy_score_list[i])
        if insomnia_accuracy_score_list[i] == max(insomnia_accuracy_score_list):
            print("The best k value for anxiety is ", i, " with a mean accuracy of ", accuracy_score_list[i])

        

        
def gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain):
    param_grid = {
    'n_neighbors': np.arange(2, 150, 1),
    'weights': ['uniform', 'distance']
    }
     
    OCD_knn = KNeighborsClassifier()
    insomnia_knn = KNeighborsClassifier()
    anxiety_knn = KNeighborsClassifier()
    depression_knn = KNeighborsClassifier()
     
    grid_search_OCD = GridSearchCV(OCD_knn, param_grid, cv=5, scoring='accuracy')
    grid_search_insomnia = GridSearchCV(insomnia_knn, param_grid, cv=5, scoring='accuracy')
    grid_search_anxiety = GridSearchCV(anxiety_knn, param_grid, cv=5, scoring='accuracy')
    grid_search_depression = GridSearchCV(depression_knn, param_grid, cv=5, scoring='accuracy')

    grid_search_OCD.fit(OCD_xTrain, OCD_yTrain)
    grid_search_insomnia.fit(insomnia_xTrain, insomnia_yTrain)
    grid_search_anxiety.fit(anxiety_xTrain, anxiety_yTrain)
    grid_search_depression.fit(depression_xTrain, depression_yTrain)


    results = []

    for gs, target in [(grid_search_OCD, 'OCD'), (grid_search_insomnia, 'insomnia'), (grid_search_anxiety, 'anxiety'), (grid_search_depression, 'depression')]:
        best_params = gs.best_params_
        cv_results = gs.cv_results_

        for weight in ['uniform', 'distance']:
            weight_mask = cv_results['param_weights'] == weight
            accuracies = cv_results['mean_test_score'][weight_mask]
            ks = cv_results['param_n_neighbors'][weight_mask]

            results.extend([{'target': target, 'k': k, 'weight': weight, 'accuracy': accuracy} for k, accuracy in zip(ks, accuracies)])

    print("Best OCD parameters:", grid_search_OCD.best_params_)
    print("Best OCD score:", grid_search_OCD.best_score_)

    print("Best insomnia parameters:", grid_search_insomnia.best_params_)
    print("Best insomnia score:", grid_search_insomnia.best_score_)

    print("Best anxiety parameters:", grid_search_anxiety.best_params_)
    print("Best anxiety score:", grid_search_anxiety.best_score_)

    print("Best depression parameters:", grid_search_depression.best_params_)
    print("Best depression score:", grid_search_depression.best_score_)

    return results



############################## K-FOLD CROSS VALIDATION IS PERFORMED HERE ###################################
def k_fold_validation():
     
    for train_index, test_index in kf.split(ODC_x):
                OCD_xTrain, OCD_xTest = OCD_x[train_index], OCD_x[test_index]
                OCD_yTrain, OCD_yTest = OCD_y[train_index], OCD_y[test_index]
                OCD_knn.fit(OCD_xTrain, OCD_yTrain)
                OCD_knn.predict(OCD_xTest)

                insomnia_xTrain, insomnia_xTest = insomnia_x[train_index], insomnia_x[test_index]
                insomnia_yTrain, insomnia_yTest = insomnia_y[train_index], insomnia_y[test_index]
                insomnia_knn.fit(insomnia_xTrain, insomnia_yTrain)
                insomnia_knn.predict(insomnia_xTest)

                anxiety_xTrain, anxiety_xTest = anxiety_x[train_index], anxiety_x[test_index]
                anxiety_yTrain, anxiety_yTest = anxiety_y[train_index], anxiety_y[test_index]
                anxiety_knn.fit(anxiety_xTrain, anxiety_yTrain)
                anxiety_knn.predict(anxiety_xTest)

def plot_knn(X, y, knn_model):
    # compare frequency rock and frequency jazz indicies 12 and 20
    x_min, x_max = X[:, 12].min() - 1, X[:, 12].max() + 1
    y_min, y_max = X[:, 20].min() - 1, X[:, 20].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("KNN Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# will get 12 datasets, 4 are the full datasets and 8 are train/test split

# build loops to find the best k value for the four models

# once you get the best k value across all 4 models using the mean, do k-fold cross validation on k-folds

def main():

    OCD_train = np.array(pd.read_csv("ocd_train_final.csv"))
    OCD_test = np.array(pd.read_csv("ocd_test_final.csv"))

    OCD_train = np.array(pd.read_csv("ocd_train_final.csv"))
    # OCD_test

    OCD_xTrain = OCD_train[:, :-1]
    OCD_yTrain = OCD_train[:, -1]
    OCD_xTest = OCD_test[:, :-1]
    OCD_yTest = OCD_test[:, -1]

    OCD_xTrain = np.array(pd.read_csv("binary_ocd_train_xFeat.csv"))
    OCD_yTrain = np.array(pd.read_csv("binary_ocd_train_y.csv"))
    OCD_xTest = np.array(pd.read_csv("binary_ocd_test_xFeat.csv"))
    OCD_yTest = np.array(pd.read_csv("binary_ocd_test_y.csv"))

    OCD_yPred, _ = run_OCD_model(OCD_xTrain, OCD_xTest, OCD_yTrain, OCD_yTest, 94, 'uniform') # based on best parameters from gridsearch for all these models




    OCD_fpr, OCD_tpr, _ = roc_curve(OCD_yTest, OCD_yPred)
    OCD_roc_auc = roc_auc_score(OCD_yTest, OCD_yPred)

    df = pd.DataFrame({'FPR': OCD_fpr, 'TPR': OCD_tpr})

    # Write the DataFrame to a CSV file
    df.to_csv('knn_OCD_graph_data.csv', index=False)
    print("knn ocd roc auc: ", OCD_roc_auc)
    # knn_OCD_graph_data.to_csv('knn_OCD_graph_data.csv')



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

    # run_insomnia_model(insomnia_xTrain, insomnia_xTest, insomnia_yTrain, insomnia_yTest, 65, 'distance')


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

    # run_anxiety_model(anxiety_xTrain, anxiety_xTest, anxiety_yTrain, anxiety_yTest, 30, 'uniform')
    

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

    # run_depression_model(depression_xTrain, depression_xTest, depression_yTrain, depression_yTest, 140, 'distance')

    # optimal_k_value(OCD_train, OCD_test, insomnia_train, insomnia_test, anxiety_train, anxiety_test, depression_train, depression_test)

    # results = gridSearch(OCD_xTrain, OCD_yTrain, insomnia_xTrain, insomnia_yTrain, anxiety_xTrain, anxiety_yTrain, depression_xTrain, depression_yTrain)
    

    # targets = ['OCD', 'insomnia', 'anxiety', 'depression']
    # results_by_target = {target: [result for result in results if result['target'] == target] for target in targets}

    # Plot for each target variable
    # for target in targets:
    #     target_results = results_by_target[target]

    #     plt.figure(figsize=(8, 6))

    #     ks_uniform = [result['k'] for result in target_results if result['weight'] == 'uniform']
    #     accuracies_uniform = [result['accuracy'] for result in target_results if result['weight'] == 'uniform']
    #     plt.plot(ks_uniform, accuracies_uniform, label='Uniform Weight')

    #     ks_distance = [result['k'] for result in target_results if result['weight'] == 'distance']
    #     accuracies_distance = [result['accuracy'] for result in target_results if result['weight'] == 'distance']
    #     plt.plot(ks_distance, accuracies_distance, label='Distance Weight')

    #     plt.axvline(x=140, color='r', linestyle='--')

    #     plt.xlabel('Number of Neighbors (k)')
    #     plt.ylabel('Accuracy')
    #     plt.title(f'{target}')
    #     plt.legend()
    #     plt.xlim(1, 150)
    #     plt.show()

    # OCD_knn = KNeighborsClassifier(n_neighbors=94, weights='uniform')
    # OCD_knn.fit(OCD_xTrain, OCD_yTrain)
    # plot_knn(OCD_xTrain, OCD_yTrain, OCD_knn)


if __name__ == "__main__":
    main()