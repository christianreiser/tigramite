# Imports

## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import numpy as np
import pandas as pd


def remove_nan_seq_from_top_and_bot(df):
    for column in df:
        # reset index
        df = df.set_index('Date')
        df = df.reset_index()

        # array with indices of NaN entries
        indices_of_nans = df.loc[pd.isna(df[column]), :].index.values

        # remove unbroken sequence of nans from beginning of list
        sequence_number = -1
        for i in indices_of_nans:
            sequence_number += 1
            if i == sequence_number:
                df = df.drop([i], axis=0)
            else:
                break

        # remove unbroken sequence of nans from end of list

        # reset index
        df = df.set_index('Date')
        df = df.reset_index()

        indices_of_nans = df.loc[pd.isna(df[column]), :].index.values
        indices_of_nans = np.flip(indices_of_nans)
        len_df = len(df)
        sequence_number = len_df
        for i in indices_of_nans:
            sequence_number -= 1
            if i == sequence_number:
                df = df.drop([i], axis=0)
            else:
                break

        # print nans in middle
        print('remaining Nans: ',df.loc[pd.isna(df[column]), :].index.values)

    return df


