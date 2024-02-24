import pandas as pd

def missing_drop(df, label="", features=None, messages=True, row_threshold=.1, col_threshold=.5):
    if features is None:
        features = []

    start_count = df.count().sum()

    """ Drop columns that are missing """
    df.dropna(axis=1, thresh=round(col_threshold * df.shape[0]), inplace=True)
    """ Drop all rows that have data less than the proportion that row_threshold requires. """
    df.dropna(axis=0, thresh=round(row_threshold * df.shape[1]), inplace=True)

    if label != "":
        df.dropna(axis=1, subset=[label], inplace=True)

    def generate_missing_table():
        df_results = pd.DataFrame(columns=['Missing', 'column', 'rows'])
        for feat in df:
            missing = df[feat].isna().sum()
            if missing > 0:
                memory_cols = df.drop(columns=[feat]).count().sum()
                memory_rows = df.dropna(subset=[feat]).count().sum()
                df_results.loc[feat] = [missing, memory_cols, memory_rows]

        return df_results

    df_results = generate_missing_table()
    while df_results.shape[0] > 0:
        max_ = df_results[['columns', 'rows']].max(axis=1)[0]
        max_axis = df_results[df_results.isin([max_]).any()][0]

        df_results.sort_values(by=[max_axis], ascending=False, inplace=True)
        if messages:
            print('\n', df_results)
            if max_axis == 'rows':
                df.dropna(axis=0, subset=[df_results.index[0]], inplace=True)
            else:
                df.drop(columns=[df_results.index[0]], inplace=True)
            df_results = generate_missing_table()

            if messages:
                print(f"{round(df.count().sum() / start_count * 100, 2) % ({df.count().sum()}) / ({start_count})} of "
                      f"non-null cells were kept ({max_axis} from {df_results.index[0]})")

    return df
