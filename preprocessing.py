import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Drop Irrelevant Features - [‘Permissions’, ‘Music effects’, ‘Timestamp’]
def drop_irreleveant(df, cols):
    df.drop(columns = cols, inplace=True)
    return df

# One Hot Encoding Categorical Features
def one_hot_encode(df, cols):
    encoded_df = pd.get_dummies(df, columns=cols, dtype=int)
    return encoded_df

# Label Encode Categorical Features
def label_encode(df, cols, options):
    label_encoder = LabelEncoder()
    label_encoder.fit(options)
    for feature in cols:
        df[feature] = label_encoder.transform(df[feature])

    return df

# Converting Target Features to Categorical
def convert_ranges(df, features):
    range_vals = [float('-inf'), 1.0, 4.0, 7.0, 10.0]
    associated_labels = ["Symptoms_Asymptomatic", "Symptoms_Mild", "Symptoms_Moderate", "Symptoms_Severe"]

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
def min_max_scale(train_df, test_df, cols):
    scaler = MinMaxScaler()
    scaler.fit(train_df[cols]) 
    
    train_df[cols] = scaler.transform(train_df[cols]) 
    test_df[cols] = scaler.transform(test_df[cols]) 

    return train_df, test_df

# Train-Test Split of Data
def data_split(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test

# Pearson Correlation Matrix
def pearson_matrix(df):
    corrDF = df.corr(method='pearson')
    return corrDF

# Pearson Correlation --> Delta Feature & Remover
def pearson_drop_delta(orig_df, corr_df, delta):
    def delta_search(correlation_matrix, delta_val):
        delta_list = [] 

        val_len = len(correlation_matrix)

        for row_val in range(val_len-1): 
            for col_val in range(row_val+1, val_len): 
                if np.abs(correlation_matrix.iloc[row_val, col_val]) > delta_val:
                    options = [correlation_matrix.index[row_val], correlation_matrix.columns[col_val]] 
                    delta_list.append(np.random.choice(options)) 

        return delta_list

    delta_list = delta_search(corr_df, delta) 
    final_drop_list = list(set(delta_list)) 
    print(final_drop_list)
    final_DF = orig_df.drop(final_drop_list, axis=1)

    return final_DF

# Pearson Correlation --> Gamma Feature & Remover
def pearson_drop_gamma(orig_df, corr_df, gamma):
    def gamma_search(correlation_matrix, gamma_val):
        gamma_list = [] 

        target_col = corr_df.columns[-1] # Gets DF w/ only Target Column to Focus on... converts into Series Pandas Object (items())

            # Iterates through every Cell from the Target Col
        for row_val, corr_value in corr_df[target_col].items():
            if np.abs(corr_value) < gamma: # Threshold Check
                gamma_list.append(row_val) # Add Val Feature to 'gamma_list'

        return gamma_list

    gamma_list = gamma_search(corr_df, gamma) 
    final_drop_list = list(set(gamma_list)) 
    print(final_drop_list)
    
    return final_drop_list 

# Update Sample Name --> Remove 'Symptoms_'
def remove_prefix(df, col_name):
    df[col_name] = df[col_name].str.replace('Symptoms_', '')
    return df

