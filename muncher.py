import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

class DataMuncher(object):
    def __init__(self,
                 df = None,
                 columns_name_map = None,
                 auto_parse_cols = False,
                 drop_cols = None,
                 sep = ',',
                 decimal = b'.'):
        '''
        Constructor for DataMuncher class objects.
        Args
        ----
            df (string | pandas.DataFrame) - name of file to be loaded or
                dataframe.
            columns_name_map (dict[string]: string) - translation dictionary
                for data's column names.
            sep (string) - separator character to be passed to pd.read_csv
            decimal (byte string) - decimal character also for pd.read_csv
        '''
        # if df is None, we simply initialize an empty DataFrame
        if df is None:
            self.df = pd.DataFrame()
        # for strings, we try to initialize an empty dataframe with the
        # indicated file's data
        elif type(df) == str:
            self.df = pd.read_csv(df, sep = sep, decimal = decimal)
        # if it's a data frame, we just assign it
        elif type(df) == pd.DataFrame:
            self.df = df
        # else, value error
        else:
            raise ValueError("df should either be a str or pd.DataFrame")
        # columns translation
        # if ``auto_parse_cols`` is True, we run ``auto_parse_cols``
        if auto_parse_cols == True:
            self.df = self.auto_parse_cols_names().df
        # else, if ``columns_name_map`` is passed, we apply it's translation
        elif columns_name_map is not None:
            self.df = self.parse_cols(columns_name_map).df
        if drop_cols is not None:
            self.df = self.df.drop(columns = drop_cols)

    def _parse_string(self, s):
        '''
        Method for simple parsing of strings. Should be used with caution.
        It basically splits the string by anything that's not a regular
        character or number. Then it checks if the first character in the
        string is a number, and if it is, adds an 'x'.
        Args
        -----
            s (string) - string to be parsed
        '''
        # if we have an empty string, it should be filled with 'NA'
        if len(s) < 1:
            return 'NA'
        parsed_s = '_'.join([w2 for w2 in
                             [w.lower() for w in re.split('[^A-Za-z0-9]', s)]
                             if len(w2) >= 1])
        # then we check to see if the first character is a number.
        # if it is, we should put an x in front of it
        try:
            int(parsed_s[0])
            parsed_s = 'x' + parsed_s
        # if it isn't, though, we should keep it as it is
        # except clause exists here for the sole intention of making it
        # explicit that is our intention to capture specifically the
        # ``ValueError`` and ignore it
        except ValueError:
            pass
        return parsed_s

    def auto_parse_cols_names(self, df = None):
        '''
        Tries to auto parse the columns names instead of using the
        ``cols_names_map`` argument passed to the constructor. Should be used
        with caution. Uses the ``_parse_string`` method internally.

        Args
        -----
            df (pd.DataFrame) - if not to use dm own's initialized df
        '''
        if df is None:
            df = self.df.copy(deep=True)
        df.columns = [self._parse_string(c) for c in df.columns]
        return DataMuncher(df = df)

    def drop_cols(self, cols, df = None):
        '''
        Simple drop columns method.

        Args
        ----
            cols (list(str)) - labels of the columns to be dropped
        '''
        if df is None:
            df = self.df.copy(deep=True)
        if len(cols) < 1:
            raise ValueError('You should indicate at least one column')
        return DataMuncher(df = df.drop(columns = cols))

    def parse_cols(self, cols_map, df = None):
        '''
        Applies the ``cols_map`` translation passed to the constructor

        Args
        ----
            cols_map (dict) - Dictionary containing cols names and translation
                values.
            df (pd.DataFrame) - DataFrame, if not to use it's own

        '''
        if df is None:
            df = self.df.copy(deep=True)
        drop_cols = []
        for c in df.columns:
            if c not in cols_map.keys():
                drop_cols.append(c)
        # if there are missing columns in the dictionary, we just drop them
        if len(drop_cols) > 0:
            df = df.drop(columns = drop_cols)
        # then apply the translation
        df.columns = [cols_map[c] for c in df.columns]
        return DataMuncher(df = df)

    def standardize(self, labels, df = None):
        '''
        standardizes a series with name ``label`` within the pd.DataFrame
        ``df``.
        taken from:
        https://github.com/pandas-dev/pandas/issues/18028
        Args
        ----
            labels (list(string)) - labels of columns to be standardized
            df (pd.DataFrame) - if not to use dm own's initialized df
        Returns
        ----
            new DataMuncher with standardized pd.Series relative to the label's
            column
        '''
        if df is None:
            df = self.df.copy(deep=True)
        for l in labels:
            series = df.loc[:, l]
            if np.issubdtype(series.dtype, np.number):
                avg = series.mean()
                stdv = series.std()
                series_standardized = (series - avg) / stdv
                df[l] = series_standardized
            else:
                print("WARNING: Column '{}' not numeric, skipping".format(l))
        return DataMuncher(df = df)

    def standardize_all(self, df = None):
        '''
        standardizes all of the dataframe's columns

        Args
        ----
            df (pd.DataFrame) - Dataframe to be used if not it's own.
        '''
        if df is None:
            df = self.df.copy(deep=True)
        return self.standardize(df.columns, df = df)

    def plot_all_x_y(self, dep, n_cols, kind, df = None):
        '''
        Method for plotting all independent variables vs the dependent one at
        once.

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


