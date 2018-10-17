# Data Muncher

## About

* Data muncher is just a python class for basic data analysis. It was written with the purpose of helping first time analysis of data. By using it, we can easily do basic preprossing of the data, as well as some visualizations.

* Please note that this is at an early development stage. 

* Contributions are more than welcome! ;)

## Usage

### Basics

* DataMuncher can be instantiated by pointing to a file, passing a pandas.DataFrame or just by itself.
```
# passing a string to the constructor will be interpreted as a filename
dm = DataMuncher('data.csv')

# passing a pandas.DataFrame
import pandas as pd

df = pd.read_csv('data.csv')
dm = DataMuncher(df)

# no data parameter
dm = DataMuncher()

# anything else raises an ValueError
dm = DataMuncher(df = 1)
ValueError: df should either be a str or pd.DataFrame
```

* If a filename or pandas.DataFrame was passed to the constructor, it can be later acessed by `dm.df`.

* The second parameter to the constructor is a map containing the header names as keys and the translation as values. Eg.:
```
pd.read_csv('example_data.csv').head()
   ID    Gender  CelPhoneNumber HouseAddress age
0  7590  Female   0176 99856912  some street  25
1  5575    Male   0176 12385619  abcdefg, 12  80
2  3668    Male   0174 12359500  kblstr,  37  52
3  7795    Male   0172 12394857  another one  30
4  9237  Female   0176 29506463   nice place  22

cols = {
    'ID':               'id',
    'Gender':           'gender',
    'CelPhoneNumber':   'cel_phone_number',
    'HouseAddress':     'house_address',
}

dm = DataMuncher(df = 'example_data.csv', columns_name_map = cols)

# then, accessing the internal pd.DataFrame we can see it's been translated
dm.df.head()
   id    gender   cel_phone_number house_address age
0  7590  Female      0176 99856912   some street  25
1  5575    Male      0176 12385619   abcdefg, 12  80
2  3668    Male      0174 12359500   kblstr,  37  52
3  7795    Male      0172 12394857   another one  30
4  9237  Female      0176 29506463    nice place  22

# It should be noted, though, that if you leave a column out, it will be dropped. Eg.:

cols = {
    #'ID':               'id', <- comment ID out
    'Gender':           'gender',
    'CelPhoneNumber':   'cel_phone_number',
    'HouseAddress':     'house_address',
}

dm = DataMuncher(df = 'example_data.csv', columns_name_map = cols)
dm.df.head()
   gender   cel_phone_number house_address age
0  Female      0176 99856912   some street  25
1  Male        0176 12385619   abcdefg, 12  80
2  Male        0174 12359500   kblstr,  37  52
3  Male        0172 12394857   another one  30
4  Female      0176 29506463    nice place  22
```

### Encoding
* Encoding any or all attributes should be pretty straight forward (we are saying simple encoding here, basically assigning numbers to the categories/labels).
```
# after instantiating.. encoding one attribute
dm = dm.encode_target_simple('gender')

# encoding all attributes
dm = dm.encode_all_simple()
```

### Plotting
* Plotting one variable charts:
```
# after instantiating.. plotting a one variable chart by itself
dm.encode_target_simple('gender').plot_one_x('gender', kind = 'hist')

# what we're doing here is first enconding the gender column,
# considering that we need a numerical value in this case
# The returned value is another DataMuncher, so we can chain the plot method

# plotting all attributes at once
dm.plot_all_x_simple_encode('hist', n_cols = 4)

# the plot_all_x_simple_encode first uses the encode_all_simple method
# and then divides the plot area in 4 columns and as many rows it needs,
# and plot every variable in the dataframe.
```
* Plot two variable charts:
```
# one chart only
dm.encode_target_simple('gender').plot_one_x_y('gender', 'age', 'scatter')

# plotting all variables using a single one as the y for every chart
dm.plot_all_x_y('age', 4, 'scatter')

```

## TODO's
* implement one hot encoding / dummifying
* implement auto conversion of object types to numeric

#### Author
Tito Lins</br>
07.10.2018</br>
Berlin

