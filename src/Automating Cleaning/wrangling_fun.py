"""
# basic data wrangling

this basic wrangling function eliminates:

1. empty columns
2. columns with all unique values, or those 95% + unique values
3. columns with single values
"""


def basic_wrangling(df, features=None, missing_threshold=0.95, unique_threshold=0.95, messages=True):
    import pandas as pd

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


# Date and Time Management

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
