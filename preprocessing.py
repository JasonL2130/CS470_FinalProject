import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Drop Irrelevant Features - [‘Permissions’, ‘Music effects’, ‘Timestamp’]
def drop_irreleveant(df, cols):
    df.drop(columns = cols, inplace=True)
    return df

# One Hot Encoding Categorical Features
def one_hot_encode(df, cols):
    encoded_df = pd.get_dummies(df, columns=cols, dtype=int)
    return encoded_df

# Label Encode Categorical Features
def label_encode(df, cols):
    label_encoder = LabelEncoder()
    df[cols] = df[cols].apply(label_encoder.fit_transform)

    return df

# Converting Target Features to Categorical
def convert_ranges(df, features):
    range_vals = [float('-inf'), 1.0, 4.0, 7.0, 10.0]
    associated_labels = ["Symptoms_None", "Symptoms_Mild", "Symptoms_Moderate", "Symptoms_Severe"]

    for feature in features:
        df[feature] = pd.cut(df[feature],
                                    bins = range_vals,
                                    right = True,
                                    labels = associated_labels)
    return df

# Moving Label Features to End of DF
def label_end(df, cols):
    for feature in cols:
        label_col = df.pop(feature)
        df[feature] = label_col

    return df

# Min-Max Scaling --> Only for Training Data
def scale(df, cols):
    scaler = MinMaxScaler()

    return None

# Pearson Correlation Matrix
def pearson_matrix(df):
    corrDF = df.corr(method='pearson')
    return corrDF


