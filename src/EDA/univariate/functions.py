def univariate(df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    output_df = pd.DataFrame(columns=['feature', 'type', 'count', 'missing', 'unique', 'mode', 'min', 'q1', 'median',
                                      'q3', 'max', 'mean', 'std', 'skew', 'kurt'])
    output_df.set_index('feature', inplace=True)

    for col in df:
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]

        if pd.api.types.is_numeric_dtype(df[col]):
            min_ = df[col].min()
            q1 = df[col].quantile(.25)
            median = df[col].median()
            q3 = df[col].quantile(.75)
            max_ = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurt()

            output_df.loc[col] = [dtype, count, missing, unique, mode, min_, q1, median,
                                  q3, max_, mean, std, skew, kurt]
            sns.histplot(data=df, x=col)
        else:
            output_df.loc[col] = [dtype, count, missing, unique, '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
            sns.countplot(data=df, x=col)

        plt.show()
    return output_df
