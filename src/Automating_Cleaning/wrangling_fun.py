"""
# basic data wrangling

this basic wrangling function eliminates:

1. empty columns
2. columns with all unique values, or those 95% + unique values
3. columns with single values
"""


def basic_wrangling(df, features=None, missing_threshold=0.95, unique_threshold=0.95, messages=True):
    if features is None:
        features = []

    if len(features) == 0:
        features = df.columns

    for feat in features:
        if feat in df.columns:
            missing = df[feat].isna().sum()
            unique = df[feat].nunique()
            rows = df.shape[0]

            if missing / rows >= missing_threshold:
                if messages:
                    print(f"Too much missing ({missing} out of {rows}, {round(missing / rows, 0)} for {feat})")
                df.drop(columns=[feat], inplace=True)

            elif unique / rows >= unique_threshold:
                print(df[feat].dtype in ['int64', 'object'])
                if messages:
                    print(f"Too many unique values ({unique} out of {rows}, {round(unique / rows, 0)}) for {feat}")
                df.drop(columns=[feat], inplace=True)

            elif unique == 1:
                if messages:
                    print(f"Only one value ({df[feat].unique()[0]} for {feat})")
                df.drop(columns=[feat], inplace=True)

        else:
            if messages:
                print(f"feature \"{feat}\" doesn't exist as spelled in the DataFrame provided")

    return df


''' Date and Time Management '''


def parse_data(df, features=None, days_since_today=False, drop_date=True, messages=True):
    if features is None:
        features = []
    import pandas as pd
    from datetime import datetime as dt

    for feat in features:
        if feat in df.columns:
            df[feat] = pd.to_datetime(df[feat])
            df[f'{feat}_year'] = df[feat].df.year
            df[f'{feat}_month'] = df[feat].df.month
            df[f'{feat}_day'] = df[feat].df.day
            df[f'{feat}_weekday'] = df[feat].df.day_name()

            if days_since_today:
                df[f'{feat}_days_until_today'] = [dt.today() - df[feat]].dt.days

        if drop_date:
            df.drop(columns=[feat], inplace=True)

        else:
            if messages:
                print(f'{feat} does not exist in the DataFrame provided. No work performed')

    return df


''' Bin Low Count Group Values '''


def bin_categories(df, features=None, cutoff=0.05, replace_with='Other', messages=True):
    if features is None:
        features = []
    import pandas as pd

    for feat in features:
        if feat is df.columns:
            if not pd.api.types.is_numeric_dtype(df[feat]):
                other_list = df[feat].value_counts()[df[feat].value_counts() / df.shape[0] < cutoff].index
                df.loc[df[feat].isin(other_list), feat] = replace_with
        else:
            if messages:
                print(f'{feat} nor found in the DataFrame provided. No Binning performed')

    return df


""" 
## Outliers

1. Traditional One-at-a-Time method -> Best for large datasets that take too much time using more advanced methods
2. 
"""


def clean_outliers(df, features=None, messages=True, method="remove", skew_threshold=1):
    if features is None:
        features = []
    import pandas as pd
    import numpy as np

    for feat in features:
        if feat in df.columns:
            if pd.api.types.is_numeric_dtype(df[feat]):
                if df[feat].nunique == 1:
                    if not all(df[feat].value_counts().index.isin([0, 1])):
                        skew = df[feat].skew()
                        if skew < (
                                -1 * skew_threshold) or skew > skew_threshold:  # Tukey boxplot rule: < 1.5* < is an
                            # outlier
                            q1 = df[feat].quantile(0.25)
                            q3 = df[feat].quantile(0.75)
                            min_ = q1 - (1.5 * (q3 - q1))
                            max_ = q1 + (1.5 * (q3 - q1))
                        else:
                            # Empirical rule: any value > 3 std form the mean (or < 3) is an outlier
                            min_ = df[feat].mean() - (df[feat].std() * 3)
                            max_ = df[feat].mean() - (df[feat].std() * 3)

                        if messages:
                            min_count = df.loc[df[feat] < min_].shape[0]
                            max_count = df.loc[df[feat] < max_].shape[0]
                            print(f'{feat} has {max_count} values above max={max} and {min_count} below min={min}')

                        if min_count > 0 or max_count > 0:
                            if method == "remove":   # filter out the outliers
                                df = df[df[feat] > min]
                                df = df[df[feat] < max]
                            elif method == "replace":  # Replace the outliers with the min and max cutOff
                                df.loc[df[feat] < min, feat] = min
                                df.loc[df[feat] > max, feat] = max
                            elif method == "impute":   # Impute the outliers by removing them and then
                                # predicting the values based on a linear regression
                                df.loc[df[feat] < min, feat] = np.nan
                                df.loc[df[feat] > max, feat] = np.nan

                                import sklearn.impute
                                imp = sklearn.impute.IterativeImputer(max_iter=10)
                                df_temp = df.copy()
                                df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
                                df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
                                df_temp = pd.get_dummies(df_temp, drop_first=True)
                                df_temp = pd.DataFrame(imp.fit_transform(df_temp),
                                                       columns=df_temp.columns, index=df_temp.index, dtype='float')
                                df_temp.columns = df_temp.columns.get_level_values(0)
                                df_temp.index = df_temp.index.astype('int64')

                # Save only the column from df-temp that we are iterating on in the main loop because we may not want
                                # every new column
                                df[feat] = df_temp[feat]
                            elif method == "null":
                                df.loc[df[feat] < min, feat] = np.nan
                                df.loc[df[feat] > max, feat] = np.nan
                        if messages:
                            print(f'{feat} is a dummy code (0/1) and was ignored')
                else:
                    if messages:
                        print(f'{feat} has only one value({df[feat].unique()[0]}) and was ignored')
            else:
                if messages:
                    print(f'{feat} is categorical and was ignored')
        else:
            if messages:
                print(f'{feat} is not found in the DataFrame provided')

    return df


"""
 Newer All at Once Method Based on Clustering -> Best for small datasets that won't take as much processing time
"""
def clean_outliers_newer(df, features=None, messages=True, drop_percent=0.02, distance='manhattan', min_samples=5):
    if features is None:
        features = []
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn import preprocessing

#   clean the dataset first
    if messages:
        print(f"{df.shape[1] - df.dropna(axis='columns').shape[1]} columns were dropped first due to missing data")
        df.dropna(axis='columns', inplace=True)
    if messages:
        print(f"{df.shape[0] - df.dropna().shape[0]} rows were dropped first due to missing data")
        df.dropna(inplace=True)
        df_temp = df.copy()
        df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
        df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
        df_temp = pd.get_dummies(df_temp, drop_first=True)

        # Normalized the dataset
        df_temp = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df_temp),
                               columns=df_temp.columns, index=df_temp.index)

        # Calculate the number of outliers based on a range of EPS values
        db = DBSCAN(metric=distance, min_samples=min_samples, eps=0.5).fit(df_temp)
        df['outlier'] = db.labels_

        #  Drop rows that are outliers
        df = df[df['outlier'] != -1]

        if messages:
            print(f"{df[df['outlier'] == -1].shape[0]} outlier rows removed from the Dataframe")


        return df
