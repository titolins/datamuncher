import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DataMuncher(object):
    def __init__(self, df = None, columns_name_map = None):
        '''
        Constructor for DataMuncher class objects.
        Args
        ----
            df (string | pandas.DataFrame) - name of file to be loaded or
                dataframe.
            columns_name_map (dict[string]: string) - translation dictionary
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
            for c in self.df.columns:
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
        Args
        ----
            label (string) - label of column to be standardized
            df (pd.DataFrame) - if not to use dm own's initialized df
        Returns
        ----
            new DataMuncher with standardized pd.Series relative to the label's
            column
        '''
        if df is None:
            df = self.df.copy(deep=True)
        series = df.loc[:, label]
        avg = series.mean()
        stdv = series.std()
        series_standardized = (series - avg) / stdv
        df[label] = series_standardized
        return DataMuncher(df = df)

    def plot_all_x_y(self, dep, n_cols, kind, df = None):
        '''
        Method for plotting all independent variables vs the dependent one at once
        Args
        ----
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
            self.plot_one_x_y(c, dep, kind, ax = axs[row][col], df = df)
            col = col + 1
        plt.show()

    def plot_one_x_y(self, x, y, kind, ax = None, df = None):
        '''
        Plots an x, y chart
        Args
        ----
            x (string) - x-axis column label.
            y (string) - y-axis column label.
            kind (string) - plot type.
            ax (matplotlib.AxesSubplot) - axis in which the plot should go.
            df (pandas.DataFrame) - if using a df that's not the one dm got
                instantiated with.
        '''
        if df is None:
            df = self.df.copy(deep=True)
        if ax is None:
            df[[x,y]].plot(x = x,
                           y = y,
                           kind = kind,
                           label = '{} ~ {}'.format(y, x))
            plt.show()
        else:
            df[[x,y]].plot(x = x,
                           y = y,
                           kind = kind,
                           ax = ax,
                           label = '{} ~ {}'.format(y, x))

    def plot_all_x(self, n_cols, kind, df = None):
        '''
        Method for plotting many variables at once.
        Args
        ----
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
        for c in df.columns:
            if col == n_cols:
                col = 0
                row = row + 1
            self.plot_one_x(c, kind, ax = axs[row][col], df = df)
            col = col + 1
        plt.show()

    def plot_one_x(self, x, kind, ax = None, df = None):
        if df is None:
            df = self.df.copy(deep=True)
        if ax is not None:
            df[[x]].plot(kind = kind, ax = ax)
        else:
            df[[x]].plot(kind = kind)
            plt.show()

    def encode_target_simple(self, target_column, df = None):
        '''
        Add column to df with integers for the target.
        http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
        Args
        ----
        df - pandas DataFrame.
        target_column -- column to map to int, producing
                         new Target column.
        Returns
        ----
        new DataMuncher, with respective dataframe with encoded target col
        '''
        if df is None:
            df = self.df.copy(deep = True)
        targets = df[target_column].unique()
        map_to_int = {name: n for n, name in enumerate(targets)}
        df[target_column] = df[target_column].replace(map_to_int)
        return DataMuncher(df = df)

    def encode_all_simple(self, df = None):
        '''
        Runs `encode_target_simple` for all dataframe columns that have a numpy
        object type (`np.dtype('O')`).
        Args
        ----
            df (pd.DataFrame) - pandas data frame containing data to be encoded
        Returns
        ----
            DataMuncher - so we chain methods! Yay
        '''
        if df is None:
            df = self.df.copy(deep = True)
        # iterate cols and cols_types
        for c, c_type in df.dtypes.to_dict().items():
            # check it if it's a numpy object type
            if c_type == np.dtype('O'):
                # if so, encode it
                df = self.encode_target_simple(c, df = df).df
        return DataMuncher(df = df)

    def plot_all_x_simple_encode(self, kind, n_cols = 4, df = None):
        '''
        Plots every variable in the dataframe.
        Args
        ----
            kind (string) - kind of chart to be plotted.
            n_cols (int) - number of columns to divide the plot area into.
            df (pd.DataFrame) - if using another dataframe.
        '''
        if df is None:
            df = self.df.copy(deep=True)
        self.encode_all_simple(df = df).plot_all_x(n_cols, kind, df = df)

    def plot_scatter_dep_simple_encode(self, dependent, n_cols = 4, df = None):
        '''
        Scatter plots all variables against the dependent indicated
        Args
        ----
            dependent (string) - name of the dependent variable to plot against
            n_cols (int) - number of columns to be plotted in the chart area.
        '''
        if df is None:
            df = self.df.copy(deep = True)
        self.plot_all_x_y(dependent,
                          n_cols,
                          'scatter',
                          df = self.encode_all_simple(df = df).df)

    def plot_box_all_simple_encode(self, n_cols = 4, df = None):
        '''
        Method for plotting boxes for all variables after simple encode
        Args
        ----
            n_cols (int) - number of columns to divide plot area into
            df (pd.DataFrame) - if using dm as a parser to other df
        '''
        if df is None:
            df = self.df.copy(deep = True)
        self.plot_all_x(n_cols,
                        kind = 'box',
                        df = self.encode_all_simple(df = df).df)

    def plot_density_all_simple_encode(self, n_cols = 4, df = None):
        '''
        Method for plotting the density for all variables after simple encode
        Args
        ----
            n_cols (int) - number of columns to divide plot area into
            df (pd.DataFrame) - if using dm as a parser to other df
        '''
        if df is None:
            df = self.df.copy(deep = True)
        self.plot_all_x(n_cols,
                        kind = 'density',
                        df = self.encode_all_simple(df = df).df)


