def skew_correct(df, feature, max_power=100, message=True):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from src.Automating_Cleaning.wrangling_fun import basic_wrangling

    if not pd.api.types.is_numeric_dtype(df[feature]):
        if message:
            print(f"{feature} is not a numeric feature. No transformation performed")
        return df

    """ Address missing data """
    # clean the dataset first
    df = basic_wrangling(df, messages=False)
    if message:
        print(f"{df.shape[0] - df.dropna().shape[0]} rows were dropped first due to missing data")
        df.dropna(inplace=True)

    """ in case the dataset is too big, we can reduce the subsample """
    df_temp = df.copy()
    if df_temp.memory_usage().sum > 100000:
        df_temp = df.sample(frac=round(5000 / df.shape[0], 2))

    """Identify the proper transformation (i)"""
    i = 1
    skew = df_temp[feature].skew()
    if message:
        print(f"Starting Skew: \t{round(skew, 5)}")

    while round(skew, 2) != 0 and i <= max_power:
        i += 0.01
        if skew > 0:
            skew = np.power(df_temp[feature], 1 / i).skew()
        else:
            skew = np.power(df_temp[feature], i).skew()
    if message:
        print(f"Final skew: \t{round(skew, 3)} based on raising to {round(i, 2)}")

    """ Make the transformed version of the feature in the df Dataframe"""
    if -0.1 < skew < 0.1:
        if skew > 0:
            corrected = np.power(df[feature], 1 / round(i, 3))
            name = f"{feature}_1/{round(i, 3)}"
        else:
            corrected = np.power(df[feature], round(i, 3))
            name = f"{feature}_{round(i, 3)}"
        df[name] = corrected  # Add the corrected version of the feature back into the original df
    else:
        name = f"{feature}_binary"
        df[name] = df[feature]
        if skew > 0:
            df.loc[df[name] == df[name].value_counts().index[0], name] = 0
            df.loc[df[name] != df[name].value_counts().index[0], name] = 1
        else:
            df.loc[df[name] == df[name].value_counts().index[0], name] = 1
            df.loc[df[name] == df[name].value_counts().index[0], name] = 0
        if message:
            print(f"The feature {feature} could not be transformed into a normal distribution.")
            print(f"Instead, it has been converted to a binary (0/1")

    if message:
        f, axes = plt.subplots(1, 2, figsize=[7, 3.5])
        sns.despine(left=True)
        sns.histplot(df_temp[feature], color='b', ax=axes[0])

        if -0.1 < skew < 0.1:
            if skew > 0:
                corrected = np.power(df_temp[feature], 1 / round(i, 3))
            else:
                corrected = np.power(df_temp[feature], round(i, 3))
            df_temp['corrected'] = corrected
            sns.histplot(df_temp.corrected, color='g', ax=axes[1], kde=True)
        else:
            df_temp['corrected'] = df[feature]
            if skew > 0:
                df_temp.loc[df_temp['corrected'] == df_temp['corrected'].min(), 'corrected'] = 0
                df_temp.loc[df_temp['corrected'] > df_temp['corrected'].min(), 'corrected'] = 1
            else:
                df_temp.loc[df_temp['corrected'] == df_temp['corrected'].max(), 'corrected'] = 1
                df_temp.loc[df_temp['corrected'] < df_temp['corrected'].max(), 'corrected'] = 0
            sns.countplot(data=df_temp, x='corrected', color='g', ax=axes[1])
        plt.setp(axes, yticks=[])
        plt.tight_layout()
        plt.show()

        return df
