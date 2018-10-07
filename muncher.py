import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




###############################################################################
###############################################################################
######    TODO:                                                     ###########
###############################################################################
######    * fix columns_name_map functionality (constructor)        ###########
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



class DataMuncher(object):
    def __init__(self, df = None, columns_name_map = None):
        '''
        Constructor for DataMuncher class objects.
        Args
            - df (string | pandas.DataFrame) - name of file to be loaded or
                dataframe.
            - columns_name_map (dict[string]: string) - translation dictionary
                for data's column names.
        '''
        # if df is None, we simply initialize an empty DataFrame
        if df is None:
            self.df = pd.DataFrame()
        # for strings, we try to initialize an empty dataframe with the
        # indicated file's data
        elif type(df) == str:
            self.df = pd.read_csv(df)
        # if it's a data frame, we just assign it
        elif type(df) == pd.DataFrame:
            self.df = df
        # else, value error
        else:
            raise ValueError("df should either be a str or pd.DataFrame")
        # columns translation. if no dict is passed, we do nothing
        if columns_name_map is not None:
            # create an array to keep notice of which columns are not in the
            # translation dict
            drop_cols = []
            for c in data.columns:
                if c not in columns_name_map.keys():
                    drop_cols.append(c)
            # if there are missing columns in the dictionary, we just drop them
            if len(drop_cols) > 0:
                self.df = self.df.drop(columns = drop_cols)
            # then apply the translation
            self.df.columns = [
                columns_name_map[c] for c in self.df.columns]

    def standardize(self, label, df = None):
        '''
        standardizes a series with name ``label'' within the pd.DataFrame
        ``df''.
        taken from:
        https://github.com/pandas-dev/pandas/issues/18028
        Returns:
            - standardized pd.Series relative to the label's column
        '''
        if df is None:
            df = self.df.copy(deep=True)
        series = df.loc[:, label]
        avg = series.mean()
        stdv = series.std()
        series_standardized = (series - avg) / stdv
        return series_standardized

    def plot_many_dep(self, dep, n_cols, kind, df = None):
        '''
        Method for plotting all independent variables vs the dependent one at once
        Args:
            df (pd.DataFrame) - respective dataframe. Passing it as an
                argument here because we might need this data to be encoded
                (so we can have methods for simple encoding and one hot
                encode)
            dep (string) - dependent variable
            n_cols (int) - number of columns in the plot
            kind (string) - kind of plot to be passed into pd.DataFrame.plot()
        '''
        if df is None:
            df = self.df.copy(deep=True)
        # number of rows should be the total number of columns divided by the
        # number of columns plotted by row (passed by the n_cols parameter)
        # we subtract one because we're not plotting 'dep ~ dep'
        n_rows = math.ceil((len(df.columns)-1)/n_cols)
        # create the subplot areas
        _, axs = plt.subplots(n_rows, n_cols)
        col = row = 0
        for c in self.df.columns:
            # if it's the dependent column, we skip
            if c == dep:
                continue
            # if in the last column, we reset col counter and move to the next
            # line
            if col == n_cols:
                col = 0
                row = row + 1
            # plot
            # ideally, this method should be passed as a paramenter if we want
            # better code reutilisation. but for that we would need wrappers..
            df[[c,dep]].plot(x = c,
                             y = dep,
                             kind = kind,
                             ax=axs[row][col],
                             label = '{} ~ {}'.format(dep, c))
            col = col + 1
        plt.show()

    def plot_many(self, n_cols, kind, df = None):
        '''
        Method for plotting many variables at once.
        Args:
            df (pd.DataFrame) - respective dataframe. Passing it as an
                argument here because we might need this data to be encoded
                (so we can have methods for simple encoding and one hot
                encode)
            n_cols (int) - number of columns in the plot
            kind (string) - kind of plot to be passed into pd.DataFrame.plot()
        '''
        if df is None:
            df = self.df.copy(deep=True)
        len_data = len(df.columns)
        n_rows = math.ceil(len_data/n_cols)
        _, axs = plt.subplots(n_rows, n_cols)
        col = row = 0
        print('len(df.columns): {0}'.format(len(df.columns)))
        print('n_cols: {0}\nn_rows: {1}'.format(n_cols, n_rows))
        for c in df.columns:
            print('row: {0}\ncol: {1}'.format(row, col))
            if col == n_cols:
                col = 0
                row = row + 1
            df[[c]].plot(kind = kind,
                         ax=axs[row][col])
            col = col + 1
        plt.show()

    def encode_target_simple(self, target_column, df = None):
        """Add column to df with integers for the target.
        http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
        Args
        ----
        df -- pandas DataFrame.
        target_column -- column to map to int, producing
                         new Target column.

        Returns
        -------
        df_mod -- modified DataFrame.
        targets -- list of target names.
        """
        if df is None:
            df = self.df.copy(deep = True)
        targets = df[target_column].unique()
        map_to_int = {name: n for n, name in enumerate(targets)}
        df[target_column] = df[target_column].replace(map_to_int)
        return (df, targets)

    def encode_all_simple(self, df = None):
        '''
        Runs `encode_target_simple` for all dataframe columns that have a numpy
        object type (`np.dtype('O')`).
        '''
        if df is None:
            df = self.df.copy(deep = True)
        for c, c_type in df.dtypes.to_dict().items():
            if c_type == np.dtype('O'):
                df, _ = self.encode_target_simple(c, df = df)
        return df

    def plot_scatter_all_simple_encode(self, dependent, n_cols = 4, df = None):
        '''
        Scatter plots all variables against the dependent indicated
        Args:
            dependent (string) - name of the dependent variable to plot against
            n_cols (int) - number of columns to be plotted in the chart area.
        '''
        if df is None:
            df = self.df.copy(deep = True)
        self.plot_many_dep(dependent,
                           n_cols,
                           'scatter',
                           df = self.encode_all_simple(df = df))

    def plot_box_all_simple_encode(self, n_cols = 4, df = None):
        if df is None:
            df = self.df.copy(deep = True)
        self.plot_many(n_cols,
                       kind = 'box',
                       df = self.encode_all_simple(df = df))

    def plot_density_all_simple_encode(self, n_cols = 4, df = None):
        if df is None:
            df = self.df.copy(deep = True)
        self.plot_many(n_cols,
                       kind = 'density',
                       df = self.encode_all_simple(df = df))


