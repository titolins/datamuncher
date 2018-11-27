import math
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sklearn import tree, svm
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import (
    make_scorer,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score)

DEFAULT_METRICS = ( r2_score, explained_variance_score,
                    mean_absolute_error, mean_squared_error,
                    median_absolute_error )
SUPPORTED_ALGS = [
    ('knn_regressor', KNeighborsRegressor),
    ('knn_classifier', KNeighborsClassifier),
    ('gbt_regressor', GradientBoostingRegressor),
    ('lasso2', Lasso),
    ('random_forest_regressor', RandomForestRegressor),
    ('support_vector_regressor', svm.SVR),
    ('linear_regression', LinearRegression),
    ('decision_tree_regressor', tree.DecisionTreeRegressor)
]

class MetaMuncher(type):
    '''
    Meta class for DataMuncher. It basically adds methods for running each of
    the algorithms specified in the SUPPORTED_ALGS list
    '''
    def __init__(self, name, bases, d):
        x = type.__init__(self, name, bases, d)
        def alg_wrapper(alg_func):
            def model_method(self, dep, test_size = .33, seed = 123, df = None,
                             test_set = None, metrics = DEFAULT_METRICS,
                             **kwargs):
                df = self._get_df(df)
                model, ys = self._run_model(alg_func, dep, test_size, seed, df,
                                            metrics, **kwargs)
                if test_set is not None:
                    # if dep variable in the test set, drop it
                    if dep in test_set:
                        test_set = test_set[
                            [c for c in test_set.columns if c != dep]]
                    # append the preiction results
                    test_set['{}_pred'.format(dep)] = model.predict(test_set)
                    return (test_set, ys)
                return (model, ys)
            return model_method

        for alg_name, alg_func in SUPPORTED_ALGS:
            setattr(self, alg_name, alg_wrapper(alg_func))

class DataMuncher(object, metaclass = MetaMuncher):
    def __init__(self, df = None, columns_name_map = None, convert_cols = None,
                 auto_parse_cols_names = False, drop_cols = None, **kwargs):
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
            self.df = pd.read_csv(df, **kwargs)
        # if it's a data frame, we just assign it
        elif type(df) == pd.DataFrame:
            self.df = df
        # else, value error
        else:
            raise ValueError("df should either be a str or pd.DataFrame")
        if convert_cols is not None:
            for c in convert_cols:
                ''' please pay attention to the coerce errors here
                we are not dealing with na's. That's the reason for not using
                pandas dtype or converters parameter in the constructor
                I guess we could use converters, but would have to build a
                custom function calling the pd.converter with errors='coerce'
                '''
                self.df[c[0]] = getattr(pd,c[1])(self.df[c[0]],errors='coerce')
        # columns translation
        # if ``auto_parse_cols`` is True, we run ``auto_parse_cols``
        if auto_parse_cols_names == True:
            self.df = self.auto_parse_cols_names().df
        # else, if ``columns_name_map`` is passed, we apply it's translation
        elif columns_name_map is not None:
            self.df = self.parse_cols(columns_name_map).df
        if drop_cols is not None:
            self.df = self.df.drop(columns = drop_cols)

    def _get_df(self, df):
        '''
        '''
        return df if df is not None else self.df.copy(deep = True)

    def parse_string_(self, s):
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
        with caution. Uses the ``parse_string_`` method internally.

        Args
        -----
            df (pd.DataFrame) - if not to use dm own's initialized df
        '''
        df = self._get_df(df)
        df.columns = [self.parse_string_(c) for c in df.columns]
        return DataMuncher(df = df)

    def drop_cols(self, cols, df = None):
        '''
        Simple drop columns method.

        Args
        ----
            cols (list(str)) - labels of the columns to be dropped
        '''
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
        self.encode_all_simple(df = df).plot_all_x(n_cols, kind, df = df)

    def plot_scatter_dep_simple_encode(self, dependent, n_cols = 4, df = None):
        '''
        Scatter plots all variables against the dependent indicated
        Args
        ----
            dependent (string) - name of the dependent variable to plot against
            n_cols (int) - number of columns to be plotted in the chart area.
        '''
        df = self._get_df(df)
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
        df = self._get_df(df)
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
        df = self._get_df(df)
        self.plot_all_x(n_cols,
                        kind = 'density',
                        df = self.encode_all_simple(df = df).df)

    def get_outliers_z_score(self, threshold = 3.0, df = None):
        '''
        Method for getting the outliers of the entire dataframe. It should be
        noted, however, that if using our ``standardize`` implementation, you
        will already have the z-scores, so this may be redundant.

        Args
        ----
            threshold (float) - z-score threshold for defining outliers
            df (pd.DataFrame) - Dataframe to apply to, if not using it's own

        Returns
        ----
            dictionary (col_name: list(int)) - dictionary with column names as
                keys and a list of integers indicating the indexes of the
                outliers.

        '''
        df = self._get_df(df)
        return {
            c: [ i for i in range(0, len(df[c]))
                     if np.abs(stats.zscore(df[c]))[i] >= threshold ]
               for c in df.columns if c not in
                   [
                       c for c in df.columns if not np.issubdtype(df[c],
                                                                  np.number)
                   ]
        }

    def remove_outliers_iqr(self, df = None):
        '''
        Removes the outliers by means of iqr.

        Args
        ----
            df (pd.DataFrame) - Dataframe to use if not self's one.

        Returns
        ----
            DataMuncher with removed outliers
        '''
        df = self._get_df(df)
        q1 = df.quantile(.25)
        q3 = df.quantile(.75)
        iqr = q3 - q1
        return DataMuncher(
            df = df[~ ((df < (q1 - 1.5 * iqr)) |
                       (df > (q3 + 1.5 * iqr))).any(axis=1)]
        )

    def remove_outliers_z_score(self, df = None, threshold = 3.0):
        '''
        Simply remove outliers above the established z_score threshold from
        every column. Raises ValueError if there are na values within the
        dataframe. We do this because if we don't deal with na values before
        this, it just returns an empty dataframe.

        Args
        ----
            df (pd.DataFrame) - Dataframe to use, if not it's own
            threshold (float) - z_score threshold above and below which
                outliers should be removed
        Returns
        ----
            DataMuncher
        '''
        df = self._get_df(df)
        for na in df.isna().any():
            if na == True:
                raise ValueError("You should first deal with na values")
        return DataMuncher(
            df = df[(np.abs(stats.zscore(df)) < threshold).all(axis=1)])

    def _fill_na(self, df, col, replace_by):
        '''
        '''
        return df[col].fillna(getattr(df[col], replace_by)())

    def fill_na_mean(self, labels, df = None):
        '''
        Fills a given set of columns missing values with the respective means.

        Args
        ----
            label (list(string)) - label for the column
            df (pd.DataFrame) - dataframe to use if not it's own

        Returns
        ----
            DataMuncher with updated df
        '''
        df = self._get_df(df)
        for l in labels:
            df[l] = self._fill_na(df, l, 'mean')
        return DataMuncher(df = df)

    def fill_na_median(self, labels, df = None):
        '''
        '''
        df = self._get_df(df)
        for l in labels:
            df[l] = self._fill_na(df, l, 'median')
        return DataMuncher(df = df)

    def split_data_(self, df, dep, test_size, seed):
        '''
        Helper method to split the train and data sets.

        Args
        ----
            df (pd.DataFrame) - the dataframe to be split
            dep (string) - the dependent variable
            test_size (float) - the size of the test set
            seed (int) - the seed for the random number generator
        '''
        return train_test_split(
            df[[c for c in df.columns if c != dep]],
            df[dep],
            test_size = test_size,
            random_state = seed)

    def _get_metrics(self, alg, metrics, ys):
        '''
        '''
        print()
        print('Printing stats for {}'.format(alg))
        print('=================================================')
        print()
        print('Metrics')
        print('-------------------------------------------------')
        for m in metrics:
            print('{}: {}'.format(m.__name__, m(ys[0], ys[1])))
        print()

    def _run_model(self, m_func, dep, test_size, seed, df, metrics, **kwargs):
        '''
        Helper method for runinng different methods.
        Args
        ----
            m_func (function) - the model constructor. The class should have a
                ``fit`` function, in line with sklearn models.
            dep (string) - dependent variable we are trying to predict.
            test_size (float) - size of the test set to be created from the df.
            seed (int) - seed for the random number generator.
            df (pd.DataFrame) - dataframe to be used if not it's own.
            metrics (list(function)) - list with metric functions

        Returns
        ----
            (model, ys) ((sklearn model) tuple(original_y_vals, y_predictions))

        commented out simple validation set, implementing cross validation
        X_train, X_test, y_train, y_test = self.split_data_(df, dep, test_size,
                                                            seed)
        '''

        # we need to have those parameters later on..
        kf = KFold(n_splits=5, shuffle = True, random_state=seed)
        X = df[[c for c in df.columns if c != dep]]
        y = df[[dep]]
        model = m_func(**kwargs)
        print()
        print('Cross validation scores')
        print('-----------------------')
        for k, (train_is, test_is) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_is], X.iloc[test_is]
            y_train, y_test = y.iloc[train_is], y.iloc[test_is]
            model = model.fit(X_train, y_train)
            print('[fold {0}]: score: {1:.5f}'.
                  format(k, model.score(X_test, y_test)))
        y_pred = model.predict(X_test)
        ys = (y_test, y_pred)
        self._get_metrics(m_func.__name__, metrics, ys)
        return (model, ys)

    def compare_algs(self, dep, df = None):
        '''
        '''
        df = self._get_df(df)
        X, y = df[[c for c in df.columns if c != dep]], df[dep]
        #scoring = 'r2'
        scoring = {
            s.__name__: make_scorer(s) for s in DEFAULT_METRICS
        }

        results = []
        names = []
        for name, model in SUPPORTED_ALGS:
            kf = KFold(n_splits = 10, shuffle = True, random_state = 123)
            #cv_results = cross_val_score(model(), X, y, cv=kf, scoring=scoring)
            cv_results = cross_validate(model(), X, y, cv=kf, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            '''
            print('{}: {} ({})'.
                  format(name, cv_results.mean(), cv_results.std()))
            '''
        for name, result in zip(names, results):
            print('{}: {}'.format(name, result))
        print('')
        # boxplot algorithm comparison
        '''
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()
        '''

    def get_tree_importance(self, dep, df = None, **kwargs):
        '''
        '''
        df = self._get_df(df)
        model, _ = self.decision_tree_regressor(dep, **kwargs)
        print('Printing Tree\'s Feature Importance')
        print('=================================================')
        for (name, importance) in sorted(
                list(zip(df.columns, model.feature_importances_)),
                key = lambda x: x[1]):
            print('{}: {}'.format(name, importance))
        print()

    def set_plot_bg_color(self, color):
        '''
        Set's the background color of the plot
        Args
        ----
            color (string) - background color to be set
        '''
        bg_color_attrs = [
            'axes.facecolor',
            'axes.edgecolor',
            'figure.facecolor',
            'figure.edgecolor',
            'savefig.facecolor',
            'savefig.edgecolor'
        ]
        for attr in bg_color_attrs:
            plt.rcParams[attr] = color

    def set_spines_color(self, ax, spines, color):
        '''
        Sets the color for the plot spines
        Args
        ----
            ax (matplotlib.Axis) - axis object to apply the color to
            spines (string[]) - list of spines (bottom, top, left, right)
            color (string) - color to be applied
        '''
        for s in spines:
            ax.spines[s].set_color(color)

    def set_axis_labels_ticks_color(self, ax, color):
        '''
        Sets both ticklines and ticklabels colors for the x-axis and y-axis
        Args
        ----
            ax (matplotlib.Axis) - axis object to apply the colors to
            color (string) - color to be applied
        '''
        [t.set_color(color) for t in ax.xaxis.get_ticklines()]
        [l.set_color(color) for l in ax.xaxis.get_ticklabels()]
        [t.set_color(color) for t in ax.yaxis.get_ticklines()]
        [l.set_color(color) for l in ax.yaxis.get_ticklabels()]

    def set_spines_labels_ticks_color(self, ax, spines, color):
        '''
        Helper method to change spines and labels and ticks all to the same
        color
        Args
        ----
            ax (matplotlib.Axis) - axis object to apply the color to
            spines (string[]) - list of spines (bottom, top, left, right)
            color (string) - color to be applied
        '''
        self.set_spines_color(ax, spines, color)
        self.set_axis_labels_ticks_color(ax, color)
