import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


############################# OCD MODEL AND RESULTS GO HERE ##############################

OCD_train = pd.read_csv("ocd_train_data_name_")
OCD_test = pd.read_csv("ocd_test_data_name_")
OCD_xTrain = OCD_train.iloc[:, :-1]
OCD_yTrain = OCD_train.iloc[:, -1]
OCD_xTest = OCD_test.iloc[:, :-1]
OCD_yTest = OCD_test.iloc[:, -1]

OCD_knn = KNeighborsClassifier()

OCD_knn.fit(OCD_xTrain, OCD_yTrain)
OCD_knn.predict(OCD_xTest)




######################################################


############################### INSOMNIA MODEL AND RESULTS GO HERE ########################

insomnia_train = pd.read_csv("insomnia_data_name_")
insomnia_test = pd.read_csv("insomnia test data ehre")

insomnia_xTrain = insomnia_train.iloc[:, :-1]
insomnia_yTrain = insomnia_train.iloc[:, -1]
insomnia_xTest = insomnia_test.iloc[:, :-1]
insomnia_yTest = insomnia_test.iloc[:, -1]

insomnia_knn = KNeighborsClassifier()

insomnia_knn.fit(insomnia_xTrain, insomnia_yTrain)
insomnia_knn.predict(insomnia_xTest)

##############################################################


################################ ANXIETY MODEL AND RESULTS GO HERE #############################


anxiety_data = pd.read_csv("anxiety_data_name_")


anxiety_train = pd.read_csv("anxiety_train data here")
anxiety_test = pd.read_csv("anxiety test data ehre")

anxiety_xTrain = anxiety_train.iloc[:, :-1]
anxiety_yTrain = anxiety_train.iloc[:, -1]
anxiety_xTest = anxiety_test.iloc[:, :-1]
anxiety_yTest = anxiety_test.iloc[:, -1]

anxiety_knn = KNeighborsClassifier()

anxiety_knn.fit(anxiety_xTrain, anxiety_yTrain)
anxiety_knn.predict(anxiety_xTest)

##############################################################


################################## DEPRESSION MODEL AND RESULTS GO HERE ###############################

depression_data = pd.read_csv("axitety_data_name_")


depression_train = pd.read_csv("depression_train data here")
depression_test = pd.read_csv("depression test data ehre")

depression_xTrain = depression_train.iloc[:, :-1]
depression_yTrain = depression_train.iloc[:, -1]
depression_xTest = depression_test.iloc[:, :-1]
depression_yTest = depression_test.iloc[:, -1]

depression_knn = KNeighborsClassifier()

depression_knn.fit(depression_xTrain, depression_yTrain)
depression_knn.predict(depression_xTest)

#######################################################


#################################### FINDING OPTIMAL NUMBER OF NEIGHBORS IS DONE HERE ###################################

for k in range(1, 21):


############################## K-FOLD CROSS VALIDATION IS PERFORMED HERE ###################################




# will get 12 datasets, 4 are the full datasets and 8 are train/test split

# build loops to find the best k value for the four models

# once you get the best k value across all 4 models using the mean, do k-fold cross validation on k-folds