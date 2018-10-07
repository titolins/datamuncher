# Data Muncher

## About

* Data muncher is just a python class for basic data analysis. It was written with the purpose of helping first time analysis of data. By using it, we can easily do basic preprossing of the data, as well as some visualizations.

* Please note that this is at an early development stage. 

* Contributions are more than welcome! ;)

## Usage

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
```

* If a filename or pandas.DataFrame was passed to the constructor, it can be later acessed by `dm.df`.

* The second parameter to the constructor is a map containing the header names as keys and the new values as values. Eg.:
```
pd.read_csv('example_data.csv').head()
   ID    Gender  CelPhoneNumber HouseAddress
0  7590  Female   0176 99856912  some street
1  5575    Male   0176 12385619  abcdefg, 12
2  3668    Male   0174 12359500  kblstr,  37
3  7795    Male   0172 12394857  another one
4  9237  Female   0176 29506463   nice place

cols = {
    'ID':               'id',
    'Gender':           'gender',
    'CelPhoneNumber':   'cel_phone_number',
    'HouseAddress':     'house_address',
}

dm = DataMuncher(df = 'example_data.csv', columns_name_map = cols)

# then, accessing the internal pd.DataFrame we can see it's been translated
dm.df.head()
   id    gender   cel_phone_number house_address
0  7590  Female      0176 99856912   some street
1  5575    Male      0176 12385619   abcdefg, 12
2  3668    Male      0174 12359500   kblstr,  37
3  7795    Male      0172 12394857   another one
4  9237  Female      0176 29506463    nice place

# It should be noted, though, that if you leave a column out, it will be dropped. Eg.:

cols = {
    #'ID':               'id', <- comment ID out
    'Gender':           'gender',
    'CelPhoneNumber':   'cel_phone_number',
    'HouseAddress':     'house_address',
}

dm = DataMuncher(df = 'example_data.csv', columns_name_map = cols)
dm.df.head()
   gender   cel_phone_number house_address
0  Female      0176 99856912   some street
1  Male        0176 12385619   abcdefg, 12
2  Male        0174 12359500   kblstr,  37
3  Male        0172 12394857   another one
4  Female      0176 29506463    nice place
```


## TODO's
* all data modification methods should return DataMuncher's, so we can chain methods.





#### Author
Tito Lins</br>
07.10.2018</br>
Berlin

