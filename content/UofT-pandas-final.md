Title: UofT Scientific Coders - Intro to pandas
Date: 2016-05-04 11:29
Tags: tutorial, python, pandas
Authors: Derek Howard


This is a notebook from a presentation I recently gave to the [UofT scientific coding group](https://uoftcoders.github.io/studyGroup/). There was a good turnout and it gave me good experience for this sort of coding walkthrough as a teaching experience.

You can check out the screencast on [youtube](https://youtu.be/SihxkCtRPBs).





## Objectives
- Read tabular data into an IPython notebook
- Access columns of the data
- Isolate subsets of the data
- Generate plots based on subsetted data

## Resources
Pandas has lots of great documentation, tutorials and walkthroughs.

This tutorial was based largely off of a SWC inspired lesson by [Nancy Soontiens](https://nsoontie.github.io/2015-03-05-ubc/novice/python/Pandas-Lesson.html).

I adapted other parts from [a great tutorial](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/) by Greg Reda.

More can also be found in the pandas [documentation](http://pandas.pydata.org/pandas-docs/stable/).


A great [youtube walkthrough](https://www.youtube.com/watch?v=5JnMutdy6Fw) from PyCon 2015:


I've also found a recent set of helpful [blogposts](https://tomaugspurger.github.io/modern-1.html) for intermediate and advanced users.


## Working with dataframes

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

pandas introduces two new data structures to Python - **Series** and **DataFrame**, both of which are built on top of NumPy.

We can load in a tabular data set as a dataframe in a number of different ways.


```python
df = pd.read_table('./gapminderDataFiveYear.txt')
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1952</td>
      <td>8425333</td>
      <td>Asia</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1957</td>
      <td>9240934</td>
      <td>Asia</td>
      <td>30.332</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>1962</td>
      <td>10267083</td>
      <td>Asia</td>
      <td>31.997</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>1967</td>
      <td>11537966</td>
      <td>Asia</td>
      <td>34.020</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>1972</td>
      <td>13079460</td>
      <td>Asia</td>
      <td>36.088</td>
      <td>739.981106</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Afghanistan</td>
      <td>1977</td>
      <td>14880372</td>
      <td>Asia</td>
      <td>38.438</td>
      <td>786.113360</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Afghanistan</td>
      <td>1982</td>
      <td>12881816</td>
      <td>Asia</td>
      <td>39.854</td>
      <td>978.011439</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Afghanistan</td>
      <td>1987</td>
      <td>13867957</td>
      <td>Asia</td>
      <td>40.822</td>
      <td>852.395945</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Afghanistan</td>
      <td>1992</td>
      <td>16317921</td>
      <td>Asia</td>
      <td>41.674</td>
      <td>649.341395</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Afghanistan</td>
      <td>1997</td>
      <td>22227415</td>
      <td>Asia</td>
      <td>41.763</td>
      <td>635.341351</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Afghanistan</td>
      <td>2002</td>
      <td>25268405</td>
      <td>Asia</td>
      <td>42.129</td>
      <td>726.734055</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>31889923</td>
      <td>Asia</td>
      <td>43.828</td>
      <td>974.580338</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Albania</td>
      <td>1952</td>
      <td>1282697</td>
      <td>Europe</td>
      <td>55.230</td>
      <td>1601.056136</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Albania</td>
      <td>1957</td>
      <td>1476505</td>
      <td>Europe</td>
      <td>59.280</td>
      <td>1942.284244</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Albania</td>
      <td>1962</td>
      <td>1728137</td>
      <td>Europe</td>
      <td>64.820</td>
      <td>2312.888958</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Albania</td>
      <td>1967</td>
      <td>1984060</td>
      <td>Europe</td>
      <td>66.220</td>
      <td>2760.196931</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Albania</td>
      <td>1972</td>
      <td>2263554</td>
      <td>Europe</td>
      <td>67.690</td>
      <td>3313.422188</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Albania</td>
      <td>1977</td>
      <td>2509048</td>
      <td>Europe</td>
      <td>68.930</td>
      <td>3533.003910</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Albania</td>
      <td>1982</td>
      <td>2780097</td>
      <td>Europe</td>
      <td>70.420</td>
      <td>3630.880722</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Albania</td>
      <td>1987</td>
      <td>3075321</td>
      <td>Europe</td>
      <td>72.000</td>
      <td>3738.932735</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Albania</td>
      <td>1992</td>
      <td>3326498</td>
      <td>Europe</td>
      <td>71.581</td>
      <td>2497.437901</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Albania</td>
      <td>1997</td>
      <td>3428038</td>
      <td>Europe</td>
      <td>72.950</td>
      <td>3193.054604</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Albania</td>
      <td>2002</td>
      <td>3508512</td>
      <td>Europe</td>
      <td>75.651</td>
      <td>4604.211737</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Albania</td>
      <td>2007</td>
      <td>3600523</td>
      <td>Europe</td>
      <td>76.423</td>
      <td>5937.029526</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Algeria</td>
      <td>1952</td>
      <td>9279525</td>
      <td>Africa</td>
      <td>43.077</td>
      <td>2449.008185</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Algeria</td>
      <td>1957</td>
      <td>10270856</td>
      <td>Africa</td>
      <td>45.685</td>
      <td>3013.976023</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Algeria</td>
      <td>1962</td>
      <td>11000948</td>
      <td>Africa</td>
      <td>48.303</td>
      <td>2550.816880</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Algeria</td>
      <td>1967</td>
      <td>12760499</td>
      <td>Africa</td>
      <td>51.407</td>
      <td>3246.991771</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Algeria</td>
      <td>1972</td>
      <td>14760787</td>
      <td>Africa</td>
      <td>54.518</td>
      <td>4182.663766</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Algeria</td>
      <td>1977</td>
      <td>17152804</td>
      <td>Africa</td>
      <td>58.014</td>
      <td>4910.416756</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>Yemen, Rep.</td>
      <td>1982</td>
      <td>9657618</td>
      <td>Asia</td>
      <td>49.113</td>
      <td>1977.557010</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>Yemen, Rep.</td>
      <td>1987</td>
      <td>11219340</td>
      <td>Asia</td>
      <td>52.922</td>
      <td>1971.741538</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>Yemen, Rep.</td>
      <td>1992</td>
      <td>13367997</td>
      <td>Asia</td>
      <td>55.599</td>
      <td>1879.496673</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>Yemen, Rep.</td>
      <td>1997</td>
      <td>15826497</td>
      <td>Asia</td>
      <td>58.020</td>
      <td>2117.484526</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>Yemen, Rep.</td>
      <td>2002</td>
      <td>18701257</td>
      <td>Asia</td>
      <td>60.308</td>
      <td>2234.820827</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>Yemen, Rep.</td>
      <td>2007</td>
      <td>22211743</td>
      <td>Asia</td>
      <td>62.698</td>
      <td>2280.769906</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>Zambia</td>
      <td>1952</td>
      <td>2672000</td>
      <td>Africa</td>
      <td>42.038</td>
      <td>1147.388831</td>
    </tr>
    <tr>
      <th>1681</th>
      <td>Zambia</td>
      <td>1957</td>
      <td>3016000</td>
      <td>Africa</td>
      <td>44.077</td>
      <td>1311.956766</td>
    </tr>
    <tr>
      <th>1682</th>
      <td>Zambia</td>
      <td>1962</td>
      <td>3421000</td>
      <td>Africa</td>
      <td>46.023</td>
      <td>1452.725766</td>
    </tr>
    <tr>
      <th>1683</th>
      <td>Zambia</td>
      <td>1967</td>
      <td>3900000</td>
      <td>Africa</td>
      <td>47.768</td>
      <td>1777.077318</td>
    </tr>
    <tr>
      <th>1684</th>
      <td>Zambia</td>
      <td>1972</td>
      <td>4506497</td>
      <td>Africa</td>
      <td>50.107</td>
      <td>1773.498265</td>
    </tr>
    <tr>
      <th>1685</th>
      <td>Zambia</td>
      <td>1977</td>
      <td>5216550</td>
      <td>Africa</td>
      <td>51.386</td>
      <td>1588.688299</td>
    </tr>
    <tr>
      <th>1686</th>
      <td>Zambia</td>
      <td>1982</td>
      <td>6100407</td>
      <td>Africa</td>
      <td>51.821</td>
      <td>1408.678565</td>
    </tr>
    <tr>
      <th>1687</th>
      <td>Zambia</td>
      <td>1987</td>
      <td>7272406</td>
      <td>Africa</td>
      <td>50.821</td>
      <td>1213.315116</td>
    </tr>
    <tr>
      <th>1688</th>
      <td>Zambia</td>
      <td>1992</td>
      <td>8381163</td>
      <td>Africa</td>
      <td>46.100</td>
      <td>1210.884633</td>
    </tr>
    <tr>
      <th>1689</th>
      <td>Zambia</td>
      <td>1997</td>
      <td>9417789</td>
      <td>Africa</td>
      <td>40.238</td>
      <td>1071.353818</td>
    </tr>
    <tr>
      <th>1690</th>
      <td>Zambia</td>
      <td>2002</td>
      <td>10595811</td>
      <td>Africa</td>
      <td>39.193</td>
      <td>1071.613938</td>
    </tr>
    <tr>
      <th>1691</th>
      <td>Zambia</td>
      <td>2007</td>
      <td>11746035</td>
      <td>Africa</td>
      <td>42.384</td>
      <td>1271.211593</td>
    </tr>
    <tr>
      <th>1692</th>
      <td>Zimbabwe</td>
      <td>1952</td>
      <td>3080907</td>
      <td>Africa</td>
      <td>48.451</td>
      <td>406.884115</td>
    </tr>
    <tr>
      <th>1693</th>
      <td>Zimbabwe</td>
      <td>1957</td>
      <td>3646340</td>
      <td>Africa</td>
      <td>50.469</td>
      <td>518.764268</td>
    </tr>
    <tr>
      <th>1694</th>
      <td>Zimbabwe</td>
      <td>1962</td>
      <td>4277736</td>
      <td>Africa</td>
      <td>52.358</td>
      <td>527.272182</td>
    </tr>
    <tr>
      <th>1695</th>
      <td>Zimbabwe</td>
      <td>1967</td>
      <td>4995432</td>
      <td>Africa</td>
      <td>53.995</td>
      <td>569.795071</td>
    </tr>
    <tr>
      <th>1696</th>
      <td>Zimbabwe</td>
      <td>1972</td>
      <td>5861135</td>
      <td>Africa</td>
      <td>55.635</td>
      <td>799.362176</td>
    </tr>
    <tr>
      <th>1697</th>
      <td>Zimbabwe</td>
      <td>1977</td>
      <td>6642107</td>
      <td>Africa</td>
      <td>57.674</td>
      <td>685.587682</td>
    </tr>
    <tr>
      <th>1698</th>
      <td>Zimbabwe</td>
      <td>1982</td>
      <td>7636524</td>
      <td>Africa</td>
      <td>60.363</td>
      <td>788.855041</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>1987</td>
      <td>9216418</td>
      <td>Africa</td>
      <td>62.351</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>1992</td>
      <td>10704340</td>
      <td>Africa</td>
      <td>60.377</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>1997</td>
      <td>11404948</td>
      <td>Africa</td>
      <td>46.809</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>2002</td>
      <td>11926563</td>
      <td>Africa</td>
      <td>39.989</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>2007</td>
      <td>12311143</td>
      <td>Africa</td>
      <td>43.487</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 6 columns</p>
</div>




```python
type(df)
```




    pandas.core.frame.DataFrame




```python
df.shape
```




    (1704, 6)




```python
df.columns
```




    Index(['country', 'year', 'pop', 'continent', 'lifeExp', 'gdpPercap'], dtype='object')




```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1952</td>
      <td>8425333</td>
      <td>Asia</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1957</td>
      <td>9240934</td>
      <td>Asia</td>
      <td>30.332</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>1962</td>
      <td>10267083</td>
      <td>Asia</td>
      <td>31.997</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>1967</td>
      <td>11537966</td>
      <td>Asia</td>
      <td>34.020</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>1972</td>
      <td>13079460</td>
      <td>Asia</td>
      <td>36.088</td>
      <td>739.981106</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(6)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1952</td>
      <td>8425333</td>
      <td>Asia</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1957</td>
      <td>9240934</td>
      <td>Asia</td>
      <td>30.332</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>1962</td>
      <td>10267083</td>
      <td>Asia</td>
      <td>31.997</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>1967</td>
      <td>11537966</td>
      <td>Asia</td>
      <td>34.020</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>1972</td>
      <td>13079460</td>
      <td>Asia</td>
      <td>36.088</td>
      <td>739.981106</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Afghanistan</td>
      <td>1977</td>
      <td>14880372</td>
      <td>Asia</td>
      <td>38.438</td>
      <td>786.113360</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>1987</td>
      <td>9216418</td>
      <td>Africa</td>
      <td>62.351</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>1992</td>
      <td>10704340</td>
      <td>Africa</td>
      <td>60.377</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>1997</td>
      <td>11404948</td>
      <td>Africa</td>
      <td>46.809</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>2002</td>
      <td>11926563</td>
      <td>Africa</td>
      <td>39.989</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>2007</td>
      <td>12311143</td>
      <td>Africa</td>
      <td>43.487</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1704 entries, 0 to 1703
    Data columns (total 6 columns):
    country      1704 non-null object
    year         1704 non-null int64
    pop          1704 non-null float64
    continent    1704 non-null object
    lifeExp      1704 non-null float64
    gdpPercap    1704 non-null float64
    dtypes: float64(3), int64(1), object(2)
    memory usage: 93.2+ KB



```python
df.dtypes
```




    country       object
    year           int64
    pop          float64
    continent     object
    lifeExp      float64
    gdpPercap    float64
    dtype: object



Get summary statistics for the numeric columns with the describe() method


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>pop</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1704.00000</td>
      <td>1.704000e+03</td>
      <td>1704.000000</td>
      <td>1704.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1979.50000</td>
      <td>2.960121e+07</td>
      <td>59.474439</td>
      <td>7215.327081</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17.26533</td>
      <td>1.061579e+08</td>
      <td>12.917107</td>
      <td>9857.454543</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1952.00000</td>
      <td>6.001100e+04</td>
      <td>23.599000</td>
      <td>241.165877</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1965.75000</td>
      <td>2.793664e+06</td>
      <td>48.198000</td>
      <td>1202.060309</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1979.50000</td>
      <td>7.023596e+06</td>
      <td>60.712500</td>
      <td>3531.846989</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1993.25000</td>
      <td>1.958522e+07</td>
      <td>70.845500</td>
      <td>9325.462346</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2007.00000</td>
      <td>1.318683e+09</td>
      <td>82.603000</td>
      <td>113523.132900</td>
    </tr>
  </tbody>
</table>
</div>



## Data selection
Sometimes we need to look at only parts of the data. For example, we might want to look at the data for a particular country or in a particular year.

### Selecting columns


```python
#select multiple columns with a list of column names
col_list = ['year','lifeExp', 'country']
df[col_list]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>lifeExp</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1952</td>
      <td>28.801</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>30.332</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1962</td>
      <td>31.997</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1967</td>
      <td>34.020</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1972</td>
      <td>36.088</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1977</td>
      <td>38.438</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>39.854</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1987</td>
      <td>40.822</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1992</td>
      <td>41.674</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>41.763</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2002</td>
      <td>42.129</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2007</td>
      <td>43.828</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1952</td>
      <td>55.230</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1957</td>
      <td>59.280</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1962</td>
      <td>64.820</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1967</td>
      <td>66.220</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1972</td>
      <td>67.690</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1977</td>
      <td>68.930</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1982</td>
      <td>70.420</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1987</td>
      <td>72.000</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1992</td>
      <td>71.581</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1997</td>
      <td>72.950</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2002</td>
      <td>75.651</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2007</td>
      <td>76.423</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1952</td>
      <td>43.077</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1957</td>
      <td>45.685</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1962</td>
      <td>48.303</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1967</td>
      <td>51.407</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1972</td>
      <td>54.518</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1977</td>
      <td>58.014</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1674</th>
      <td>1982</td>
      <td>49.113</td>
      <td>Yemen, Rep.</td>
    </tr>
    <tr>
      <th>1675</th>
      <td>1987</td>
      <td>52.922</td>
      <td>Yemen, Rep.</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>1992</td>
      <td>55.599</td>
      <td>Yemen, Rep.</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>1997</td>
      <td>58.020</td>
      <td>Yemen, Rep.</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>2002</td>
      <td>60.308</td>
      <td>Yemen, Rep.</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>2007</td>
      <td>62.698</td>
      <td>Yemen, Rep.</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>1952</td>
      <td>42.038</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1681</th>
      <td>1957</td>
      <td>44.077</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1682</th>
      <td>1962</td>
      <td>46.023</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1683</th>
      <td>1967</td>
      <td>47.768</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1684</th>
      <td>1972</td>
      <td>50.107</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1685</th>
      <td>1977</td>
      <td>51.386</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1686</th>
      <td>1982</td>
      <td>51.821</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1687</th>
      <td>1987</td>
      <td>50.821</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1688</th>
      <td>1992</td>
      <td>46.100</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1689</th>
      <td>1997</td>
      <td>40.238</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1690</th>
      <td>2002</td>
      <td>39.193</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1691</th>
      <td>2007</td>
      <td>42.384</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>1692</th>
      <td>1952</td>
      <td>48.451</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1693</th>
      <td>1957</td>
      <td>50.469</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1694</th>
      <td>1962</td>
      <td>52.358</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1695</th>
      <td>1967</td>
      <td>53.995</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1696</th>
      <td>1972</td>
      <td>55.635</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1697</th>
      <td>1977</td>
      <td>57.674</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1698</th>
      <td>1982</td>
      <td>60.363</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>1987</td>
      <td>62.351</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>1992</td>
      <td>60.377</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>1997</td>
      <td>46.809</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>2002</td>
      <td>39.989</td>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>2007</td>
      <td>43.487</td>
      <td>Zimbabwe</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 3 columns</p>
</div>




```python
#alternative selection with dot notation won't work if column names have spaces, uncommon characters or leading numbers
df.lifeExp
```




    0       28.801
    1       30.332
    2       31.997
    3       34.020
    4       36.088
    5       38.438
    6       39.854
    7       40.822
    8       41.674
    9       41.763
    10      42.129
    11      43.828
    12      55.230
    13      59.280
    14      64.820
    15      66.220
    16      67.690
    17      68.930
    18      70.420
    19      72.000
    20      71.581
    21      72.950
    22      75.651
    23      76.423
    24      43.077
    25      45.685
    26      48.303
    27      51.407
    28      54.518
    29      58.014
             ...
    1674    49.113
    1675    52.922
    1676    55.599
    1677    58.020
    1678    60.308
    1679    62.698
    1680    42.038
    1681    44.077
    1682    46.023
    1683    47.768
    1684    50.107
    1685    51.386
    1686    51.821
    1687    50.821
    1688    46.100
    1689    40.238
    1690    39.193
    1691    42.384
    1692    48.451
    1693    50.469
    1694    52.358
    1695    53.995
    1696    55.635
    1697    57.674
    1698    60.363
    1699    62.351
    1700    60.377
    1701    46.809
    1702    39.989
    1703    43.487
    Name: lifeExp, dtype: float64




```python
#using describe() on a categorical column
df[['country', 'continent']].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1704</td>
      <td>1704</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>142</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Guatemala</td>
      <td>Africa</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>12</td>
      <td>624</td>
    </tr>
  </tbody>
</table>
</div>



### Selecting rows


```python
#by index location
df.iloc[[1000]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <td>Mongolia</td>
      <td>1972</td>
      <td>1320500</td>
      <td>Asia</td>
      <td>53.754</td>
      <td>1421.741975</td>
    </tr>
  </tbody>
</table>
</div>




```python
#you can provide a list of index values to select
df.iloc[[0,5,10]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1952</td>
      <td>8425333</td>
      <td>Asia</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Afghanistan</td>
      <td>1977</td>
      <td>14880372</td>
      <td>Asia</td>
      <td>38.438</td>
      <td>786.113360</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Afghanistan</td>
      <td>2002</td>
      <td>25268405</td>
      <td>Asia</td>
      <td>42.129</td>
      <td>726.734055</td>
    </tr>
  </tbody>
</table>
</div>




```python
#or select with the slice notation
df[0:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1952</td>
      <td>8425333</td>
      <td>Asia</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1957</td>
      <td>9240934</td>
      <td>Asia</td>
      <td>30.332</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>1962</td>
      <td>10267083</td>
      <td>Asia</td>
      <td>31.997</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>1967</td>
      <td>11537966</td>
      <td>Asia</td>
      <td>34.020</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>1972</td>
      <td>13079460</td>
      <td>Asia</td>
      <td>36.088</td>
      <td>739.981106</td>
    </tr>
  </tbody>
</table>
</div>




```python
#select by index label
#would require named index
country_index = df.set_index(['continent','country'])
```


```python
country_index
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>year</th>
      <th>pop</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>continent</th>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="12" valign="top">Asia</th>
      <th>Afghanistan</th>
      <td>1952</td>
      <td>8425333</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1957</td>
      <td>9240934</td>
      <td>30.332</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1962</td>
      <td>10267083</td>
      <td>31.997</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1967</td>
      <td>11537966</td>
      <td>34.020</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1972</td>
      <td>13079460</td>
      <td>36.088</td>
      <td>739.981106</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1977</td>
      <td>14880372</td>
      <td>38.438</td>
      <td>786.113360</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1982</td>
      <td>12881816</td>
      <td>39.854</td>
      <td>978.011439</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1987</td>
      <td>13867957</td>
      <td>40.822</td>
      <td>852.395945</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1992</td>
      <td>16317921</td>
      <td>41.674</td>
      <td>649.341395</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>1997</td>
      <td>22227415</td>
      <td>41.763</td>
      <td>635.341351</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>2002</td>
      <td>25268405</td>
      <td>42.129</td>
      <td>726.734055</td>
    </tr>
    <tr>
      <th>Afghanistan</th>
      <td>2007</td>
      <td>31889923</td>
      <td>43.828</td>
      <td>974.580338</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">Europe</th>
      <th>Albania</th>
      <td>1952</td>
      <td>1282697</td>
      <td>55.230</td>
      <td>1601.056136</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1957</td>
      <td>1476505</td>
      <td>59.280</td>
      <td>1942.284244</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1962</td>
      <td>1728137</td>
      <td>64.820</td>
      <td>2312.888958</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1967</td>
      <td>1984060</td>
      <td>66.220</td>
      <td>2760.196931</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1972</td>
      <td>2263554</td>
      <td>67.690</td>
      <td>3313.422188</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1977</td>
      <td>2509048</td>
      <td>68.930</td>
      <td>3533.003910</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1982</td>
      <td>2780097</td>
      <td>70.420</td>
      <td>3630.880722</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1987</td>
      <td>3075321</td>
      <td>72.000</td>
      <td>3738.932735</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1992</td>
      <td>3326498</td>
      <td>71.581</td>
      <td>2497.437901</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>1997</td>
      <td>3428038</td>
      <td>72.950</td>
      <td>3193.054604</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>2002</td>
      <td>3508512</td>
      <td>75.651</td>
      <td>4604.211737</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>2007</td>
      <td>3600523</td>
      <td>76.423</td>
      <td>5937.029526</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Africa</th>
      <th>Algeria</th>
      <td>1952</td>
      <td>9279525</td>
      <td>43.077</td>
      <td>2449.008185</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>1957</td>
      <td>10270856</td>
      <td>45.685</td>
      <td>3013.976023</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>1962</td>
      <td>11000948</td>
      <td>48.303</td>
      <td>2550.816880</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>1967</td>
      <td>12760499</td>
      <td>51.407</td>
      <td>3246.991771</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>1972</td>
      <td>14760787</td>
      <td>54.518</td>
      <td>4182.663766</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>1977</td>
      <td>17152804</td>
      <td>58.014</td>
      <td>4910.416756</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Asia</th>
      <th>Yemen, Rep.</th>
      <td>1982</td>
      <td>9657618</td>
      <td>49.113</td>
      <td>1977.557010</td>
    </tr>
    <tr>
      <th>Yemen, Rep.</th>
      <td>1987</td>
      <td>11219340</td>
      <td>52.922</td>
      <td>1971.741538</td>
    </tr>
    <tr>
      <th>Yemen, Rep.</th>
      <td>1992</td>
      <td>13367997</td>
      <td>55.599</td>
      <td>1879.496673</td>
    </tr>
    <tr>
      <th>Yemen, Rep.</th>
      <td>1997</td>
      <td>15826497</td>
      <td>58.020</td>
      <td>2117.484526</td>
    </tr>
    <tr>
      <th>Yemen, Rep.</th>
      <td>2002</td>
      <td>18701257</td>
      <td>60.308</td>
      <td>2234.820827</td>
    </tr>
    <tr>
      <th>Yemen, Rep.</th>
      <td>2007</td>
      <td>22211743</td>
      <td>62.698</td>
      <td>2280.769906</td>
    </tr>
    <tr>
      <th rowspan="24" valign="top">Africa</th>
      <th>Zambia</th>
      <td>1952</td>
      <td>2672000</td>
      <td>42.038</td>
      <td>1147.388831</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1957</td>
      <td>3016000</td>
      <td>44.077</td>
      <td>1311.956766</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1962</td>
      <td>3421000</td>
      <td>46.023</td>
      <td>1452.725766</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1967</td>
      <td>3900000</td>
      <td>47.768</td>
      <td>1777.077318</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1972</td>
      <td>4506497</td>
      <td>50.107</td>
      <td>1773.498265</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1977</td>
      <td>5216550</td>
      <td>51.386</td>
      <td>1588.688299</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1982</td>
      <td>6100407</td>
      <td>51.821</td>
      <td>1408.678565</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1987</td>
      <td>7272406</td>
      <td>50.821</td>
      <td>1213.315116</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1992</td>
      <td>8381163</td>
      <td>46.100</td>
      <td>1210.884633</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>1997</td>
      <td>9417789</td>
      <td>40.238</td>
      <td>1071.353818</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>2002</td>
      <td>10595811</td>
      <td>39.193</td>
      <td>1071.613938</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>2007</td>
      <td>11746035</td>
      <td>42.384</td>
      <td>1271.211593</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1952</td>
      <td>3080907</td>
      <td>48.451</td>
      <td>406.884115</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1957</td>
      <td>3646340</td>
      <td>50.469</td>
      <td>518.764268</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1962</td>
      <td>4277736</td>
      <td>52.358</td>
      <td>527.272182</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1967</td>
      <td>4995432</td>
      <td>53.995</td>
      <td>569.795071</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1972</td>
      <td>5861135</td>
      <td>55.635</td>
      <td>799.362176</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1977</td>
      <td>6642107</td>
      <td>57.674</td>
      <td>685.587682</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1982</td>
      <td>7636524</td>
      <td>60.363</td>
      <td>788.855041</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1987</td>
      <td>9216418</td>
      <td>62.351</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1992</td>
      <td>10704340</td>
      <td>60.377</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>1997</td>
      <td>11404948</td>
      <td>46.809</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>2002</td>
      <td>11926563</td>
      <td>39.989</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>2007</td>
      <td>12311143</td>
      <td>43.487</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 4 columns</p>
</div>




```python
country_index.loc['Americas','Canada']
```

    /home/derek/anaconda2/envs/py3/lib/python3.5/site-packages/ipykernel/__main__.py:1: PerformanceWarning: indexing past lexsort depth may impact performance.
      if __name__ == '__main__':





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>year</th>
      <th>pop</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>continent</th>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="12" valign="top">Americas</th>
      <th>Canada</th>
      <td>1952</td>
      <td>14785584</td>
      <td>68.750</td>
      <td>11367.16112</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1957</td>
      <td>17010154</td>
      <td>69.960</td>
      <td>12489.95006</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1962</td>
      <td>18985849</td>
      <td>71.300</td>
      <td>13462.48555</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1967</td>
      <td>20819767</td>
      <td>72.130</td>
      <td>16076.58803</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1972</td>
      <td>22284500</td>
      <td>72.880</td>
      <td>18970.57086</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1977</td>
      <td>23796400</td>
      <td>74.210</td>
      <td>22090.88306</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1982</td>
      <td>25201900</td>
      <td>75.760</td>
      <td>22898.79214</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1987</td>
      <td>26549700</td>
      <td>76.860</td>
      <td>26626.51503</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1992</td>
      <td>28523502</td>
      <td>77.950</td>
      <td>26342.88426</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1997</td>
      <td>30305843</td>
      <td>78.610</td>
      <td>28954.92589</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>2002</td>
      <td>31902268</td>
      <td>79.770</td>
      <td>33328.96507</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>2007</td>
      <td>33390141</td>
      <td>80.653</td>
      <td>36319.23501</td>
    </tr>
  </tbody>
</table>
</div>




```python
#boolean indexing
large_pop = df[df['pop'] > 300000000]
```


```python
large_pop
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>288</th>
      <td>China</td>
      <td>1952</td>
      <td>5.562635e+08</td>
      <td>Asia</td>
      <td>44.00000</td>
      <td>400.448611</td>
    </tr>
    <tr>
      <th>289</th>
      <td>China</td>
      <td>1957</td>
      <td>6.374080e+08</td>
      <td>Asia</td>
      <td>50.54896</td>
      <td>575.987001</td>
    </tr>
    <tr>
      <th>290</th>
      <td>China</td>
      <td>1962</td>
      <td>6.657700e+08</td>
      <td>Asia</td>
      <td>44.50136</td>
      <td>487.674018</td>
    </tr>
    <tr>
      <th>291</th>
      <td>China</td>
      <td>1967</td>
      <td>7.545500e+08</td>
      <td>Asia</td>
      <td>58.38112</td>
      <td>612.705693</td>
    </tr>
    <tr>
      <th>292</th>
      <td>China</td>
      <td>1972</td>
      <td>8.620300e+08</td>
      <td>Asia</td>
      <td>63.11888</td>
      <td>676.900092</td>
    </tr>
    <tr>
      <th>293</th>
      <td>China</td>
      <td>1977</td>
      <td>9.434550e+08</td>
      <td>Asia</td>
      <td>63.96736</td>
      <td>741.237470</td>
    </tr>
    <tr>
      <th>294</th>
      <td>China</td>
      <td>1982</td>
      <td>1.000281e+09</td>
      <td>Asia</td>
      <td>65.52500</td>
      <td>962.421380</td>
    </tr>
    <tr>
      <th>295</th>
      <td>China</td>
      <td>1987</td>
      <td>1.084035e+09</td>
      <td>Asia</td>
      <td>67.27400</td>
      <td>1378.904018</td>
    </tr>
    <tr>
      <th>296</th>
      <td>China</td>
      <td>1992</td>
      <td>1.164970e+09</td>
      <td>Asia</td>
      <td>68.69000</td>
      <td>1655.784158</td>
    </tr>
    <tr>
      <th>297</th>
      <td>China</td>
      <td>1997</td>
      <td>1.230075e+09</td>
      <td>Asia</td>
      <td>70.42600</td>
      <td>2289.234136</td>
    </tr>
    <tr>
      <th>298</th>
      <td>China</td>
      <td>2002</td>
      <td>1.280400e+09</td>
      <td>Asia</td>
      <td>72.02800</td>
      <td>3119.280896</td>
    </tr>
    <tr>
      <th>299</th>
      <td>China</td>
      <td>2007</td>
      <td>1.318683e+09</td>
      <td>Asia</td>
      <td>72.96100</td>
      <td>4959.114854</td>
    </tr>
    <tr>
      <th>696</th>
      <td>India</td>
      <td>1952</td>
      <td>3.720000e+08</td>
      <td>Asia</td>
      <td>37.37300</td>
      <td>546.565749</td>
    </tr>
    <tr>
      <th>697</th>
      <td>India</td>
      <td>1957</td>
      <td>4.090000e+08</td>
      <td>Asia</td>
      <td>40.24900</td>
      <td>590.061996</td>
    </tr>
    <tr>
      <th>698</th>
      <td>India</td>
      <td>1962</td>
      <td>4.540000e+08</td>
      <td>Asia</td>
      <td>43.60500</td>
      <td>658.347151</td>
    </tr>
    <tr>
      <th>699</th>
      <td>India</td>
      <td>1967</td>
      <td>5.060000e+08</td>
      <td>Asia</td>
      <td>47.19300</td>
      <td>700.770611</td>
    </tr>
    <tr>
      <th>700</th>
      <td>India</td>
      <td>1972</td>
      <td>5.670000e+08</td>
      <td>Asia</td>
      <td>50.65100</td>
      <td>724.032527</td>
    </tr>
    <tr>
      <th>701</th>
      <td>India</td>
      <td>1977</td>
      <td>6.340000e+08</td>
      <td>Asia</td>
      <td>54.20800</td>
      <td>813.337323</td>
    </tr>
    <tr>
      <th>702</th>
      <td>India</td>
      <td>1982</td>
      <td>7.080000e+08</td>
      <td>Asia</td>
      <td>56.59600</td>
      <td>855.723538</td>
    </tr>
    <tr>
      <th>703</th>
      <td>India</td>
      <td>1987</td>
      <td>7.880000e+08</td>
      <td>Asia</td>
      <td>58.55300</td>
      <td>976.512676</td>
    </tr>
    <tr>
      <th>704</th>
      <td>India</td>
      <td>1992</td>
      <td>8.720000e+08</td>
      <td>Asia</td>
      <td>60.22300</td>
      <td>1164.406809</td>
    </tr>
    <tr>
      <th>705</th>
      <td>India</td>
      <td>1997</td>
      <td>9.590000e+08</td>
      <td>Asia</td>
      <td>61.76500</td>
      <td>1458.817442</td>
    </tr>
    <tr>
      <th>706</th>
      <td>India</td>
      <td>2002</td>
      <td>1.034173e+09</td>
      <td>Asia</td>
      <td>62.87900</td>
      <td>1746.769454</td>
    </tr>
    <tr>
      <th>707</th>
      <td>India</td>
      <td>2007</td>
      <td>1.110396e+09</td>
      <td>Asia</td>
      <td>64.69800</td>
      <td>2452.210407</td>
    </tr>
    <tr>
      <th>1619</th>
      <td>United States</td>
      <td>2007</td>
      <td>3.011399e+08</td>
      <td>Americas</td>
      <td>78.24200</td>
      <td>42951.653090</td>
    </tr>
  </tbody>
</table>
</div>




```python
large_pop['country'].unique()
```




    array(['China', 'India', 'United States'], dtype=object)



You can also chain together multiple criteria for boolean indexing:


```python
multi_criteria = df[(df['country']=='Canada') | (df['year'] > 1990)]
```


```python
multi_criteria
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Afghanistan</td>
      <td>1992</td>
      <td>16317921</td>
      <td>Asia</td>
      <td>41.674</td>
      <td>649.341395</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Afghanistan</td>
      <td>1997</td>
      <td>22227415</td>
      <td>Asia</td>
      <td>41.763</td>
      <td>635.341351</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Afghanistan</td>
      <td>2002</td>
      <td>25268405</td>
      <td>Asia</td>
      <td>42.129</td>
      <td>726.734055</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>31889923</td>
      <td>Asia</td>
      <td>43.828</td>
      <td>974.580338</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Albania</td>
      <td>1992</td>
      <td>3326498</td>
      <td>Europe</td>
      <td>71.581</td>
      <td>2497.437901</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Albania</td>
      <td>1997</td>
      <td>3428038</td>
      <td>Europe</td>
      <td>72.950</td>
      <td>3193.054604</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Albania</td>
      <td>2002</td>
      <td>3508512</td>
      <td>Europe</td>
      <td>75.651</td>
      <td>4604.211737</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Albania</td>
      <td>2007</td>
      <td>3600523</td>
      <td>Europe</td>
      <td>76.423</td>
      <td>5937.029526</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Algeria</td>
      <td>1992</td>
      <td>26298373</td>
      <td>Africa</td>
      <td>67.744</td>
      <td>5023.216647</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Algeria</td>
      <td>1997</td>
      <td>29072015</td>
      <td>Africa</td>
      <td>69.152</td>
      <td>4797.295051</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Algeria</td>
      <td>2002</td>
      <td>31287142</td>
      <td>Africa</td>
      <td>70.994</td>
      <td>5288.040382</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Algeria</td>
      <td>2007</td>
      <td>33333216</td>
      <td>Africa</td>
      <td>72.301</td>
      <td>6223.367465</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Angola</td>
      <td>1992</td>
      <td>8735988</td>
      <td>Africa</td>
      <td>40.647</td>
      <td>2627.845685</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Angola</td>
      <td>1997</td>
      <td>9875024</td>
      <td>Africa</td>
      <td>40.963</td>
      <td>2277.140884</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Angola</td>
      <td>2002</td>
      <td>10866106</td>
      <td>Africa</td>
      <td>41.003</td>
      <td>2773.287312</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Angola</td>
      <td>2007</td>
      <td>12420476</td>
      <td>Africa</td>
      <td>42.731</td>
      <td>4797.231267</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Argentina</td>
      <td>1992</td>
      <td>33958947</td>
      <td>Americas</td>
      <td>71.868</td>
      <td>9308.418710</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Argentina</td>
      <td>1997</td>
      <td>36203463</td>
      <td>Americas</td>
      <td>73.275</td>
      <td>10967.281950</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Argentina</td>
      <td>2002</td>
      <td>38331121</td>
      <td>Americas</td>
      <td>74.340</td>
      <td>8797.640716</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Argentina</td>
      <td>2007</td>
      <td>40301927</td>
      <td>Americas</td>
      <td>75.320</td>
      <td>12779.379640</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Australia</td>
      <td>1992</td>
      <td>17481977</td>
      <td>Oceania</td>
      <td>77.560</td>
      <td>23424.766830</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Australia</td>
      <td>1997</td>
      <td>18565243</td>
      <td>Oceania</td>
      <td>78.830</td>
      <td>26997.936570</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Australia</td>
      <td>2002</td>
      <td>19546792</td>
      <td>Oceania</td>
      <td>80.370</td>
      <td>30687.754730</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Australia</td>
      <td>2007</td>
      <td>20434176</td>
      <td>Oceania</td>
      <td>81.235</td>
      <td>34435.367440</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Austria</td>
      <td>1992</td>
      <td>7914969</td>
      <td>Europe</td>
      <td>76.040</td>
      <td>27042.018680</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Austria</td>
      <td>1997</td>
      <td>8069876</td>
      <td>Europe</td>
      <td>77.510</td>
      <td>29095.920660</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Austria</td>
      <td>2002</td>
      <td>8148312</td>
      <td>Europe</td>
      <td>78.980</td>
      <td>32417.607690</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Austria</td>
      <td>2007</td>
      <td>8199783</td>
      <td>Europe</td>
      <td>79.829</td>
      <td>36126.492700</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Bahrain</td>
      <td>1992</td>
      <td>529491</td>
      <td>Asia</td>
      <td>72.601</td>
      <td>19035.579170</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Bahrain</td>
      <td>1997</td>
      <td>598561</td>
      <td>Asia</td>
      <td>73.925</td>
      <td>20292.016790</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1618</th>
      <td>United States</td>
      <td>2002</td>
      <td>287675526</td>
      <td>Americas</td>
      <td>77.310</td>
      <td>39097.099550</td>
    </tr>
    <tr>
      <th>1619</th>
      <td>United States</td>
      <td>2007</td>
      <td>301139947</td>
      <td>Americas</td>
      <td>78.242</td>
      <td>42951.653090</td>
    </tr>
    <tr>
      <th>1628</th>
      <td>Uruguay</td>
      <td>1992</td>
      <td>3149262</td>
      <td>Americas</td>
      <td>72.752</td>
      <td>8137.004775</td>
    </tr>
    <tr>
      <th>1629</th>
      <td>Uruguay</td>
      <td>1997</td>
      <td>3262838</td>
      <td>Americas</td>
      <td>74.223</td>
      <td>9230.240708</td>
    </tr>
    <tr>
      <th>1630</th>
      <td>Uruguay</td>
      <td>2002</td>
      <td>3363085</td>
      <td>Americas</td>
      <td>75.307</td>
      <td>7727.002004</td>
    </tr>
    <tr>
      <th>1631</th>
      <td>Uruguay</td>
      <td>2007</td>
      <td>3447496</td>
      <td>Americas</td>
      <td>76.384</td>
      <td>10611.462990</td>
    </tr>
    <tr>
      <th>1640</th>
      <td>Venezuela</td>
      <td>1992</td>
      <td>20265563</td>
      <td>Americas</td>
      <td>71.150</td>
      <td>10733.926310</td>
    </tr>
    <tr>
      <th>1641</th>
      <td>Venezuela</td>
      <td>1997</td>
      <td>22374398</td>
      <td>Americas</td>
      <td>72.146</td>
      <td>10165.495180</td>
    </tr>
    <tr>
      <th>1642</th>
      <td>Venezuela</td>
      <td>2002</td>
      <td>24287670</td>
      <td>Americas</td>
      <td>72.766</td>
      <td>8605.047831</td>
    </tr>
    <tr>
      <th>1643</th>
      <td>Venezuela</td>
      <td>2007</td>
      <td>26084662</td>
      <td>Americas</td>
      <td>73.747</td>
      <td>11415.805690</td>
    </tr>
    <tr>
      <th>1652</th>
      <td>Vietnam</td>
      <td>1992</td>
      <td>69940728</td>
      <td>Asia</td>
      <td>67.662</td>
      <td>989.023149</td>
    </tr>
    <tr>
      <th>1653</th>
      <td>Vietnam</td>
      <td>1997</td>
      <td>76048996</td>
      <td>Asia</td>
      <td>70.672</td>
      <td>1385.896769</td>
    </tr>
    <tr>
      <th>1654</th>
      <td>Vietnam</td>
      <td>2002</td>
      <td>80908147</td>
      <td>Asia</td>
      <td>73.017</td>
      <td>1764.456677</td>
    </tr>
    <tr>
      <th>1655</th>
      <td>Vietnam</td>
      <td>2007</td>
      <td>85262356</td>
      <td>Asia</td>
      <td>74.249</td>
      <td>2441.576404</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>West Bank and Gaza</td>
      <td>1992</td>
      <td>2104779</td>
      <td>Asia</td>
      <td>69.718</td>
      <td>6017.654756</td>
    </tr>
    <tr>
      <th>1665</th>
      <td>West Bank and Gaza</td>
      <td>1997</td>
      <td>2826046</td>
      <td>Asia</td>
      <td>71.096</td>
      <td>7110.667619</td>
    </tr>
    <tr>
      <th>1666</th>
      <td>West Bank and Gaza</td>
      <td>2002</td>
      <td>3389578</td>
      <td>Asia</td>
      <td>72.370</td>
      <td>4515.487575</td>
    </tr>
    <tr>
      <th>1667</th>
      <td>West Bank and Gaza</td>
      <td>2007</td>
      <td>4018332</td>
      <td>Asia</td>
      <td>73.422</td>
      <td>3025.349798</td>
    </tr>
    <tr>
      <th>1676</th>
      <td>Yemen, Rep.</td>
      <td>1992</td>
      <td>13367997</td>
      <td>Asia</td>
      <td>55.599</td>
      <td>1879.496673</td>
    </tr>
    <tr>
      <th>1677</th>
      <td>Yemen, Rep.</td>
      <td>1997</td>
      <td>15826497</td>
      <td>Asia</td>
      <td>58.020</td>
      <td>2117.484526</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>Yemen, Rep.</td>
      <td>2002</td>
      <td>18701257</td>
      <td>Asia</td>
      <td>60.308</td>
      <td>2234.820827</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>Yemen, Rep.</td>
      <td>2007</td>
      <td>22211743</td>
      <td>Asia</td>
      <td>62.698</td>
      <td>2280.769906</td>
    </tr>
    <tr>
      <th>1688</th>
      <td>Zambia</td>
      <td>1992</td>
      <td>8381163</td>
      <td>Africa</td>
      <td>46.100</td>
      <td>1210.884633</td>
    </tr>
    <tr>
      <th>1689</th>
      <td>Zambia</td>
      <td>1997</td>
      <td>9417789</td>
      <td>Africa</td>
      <td>40.238</td>
      <td>1071.353818</td>
    </tr>
    <tr>
      <th>1690</th>
      <td>Zambia</td>
      <td>2002</td>
      <td>10595811</td>
      <td>Africa</td>
      <td>39.193</td>
      <td>1071.613938</td>
    </tr>
    <tr>
      <th>1691</th>
      <td>Zambia</td>
      <td>2007</td>
      <td>11746035</td>
      <td>Africa</td>
      <td>42.384</td>
      <td>1271.211593</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>1992</td>
      <td>10704340</td>
      <td>Africa</td>
      <td>60.377</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>1997</td>
      <td>11404948</td>
      <td>Africa</td>
      <td>46.809</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>2002</td>
      <td>11926563</td>
      <td>Africa</td>
      <td>39.989</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>2007</td>
      <td>12311143</td>
      <td>Africa</td>
      <td>43.487</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
<p>576 rows × 6 columns</p>
</div>



### Q:
How many unique countries are there in our dataframe? Years?


```python
df.country.unique()
```




    array(['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
           'Australia', 'Austria', 'Bahrain', 'Bangladesh', 'Belgium', 'Benin',
           'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil',
           'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon',
           'Canada', 'Central African Republic', 'Chad', 'Chile', 'China',
           'Colombia', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.',
           'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba', 'Czech Republic',
           'Denmark', 'Djibouti', 'Dominican Republic', 'Ecuador', 'Egypt',
           'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Ethiopia',
           'Finland', 'France', 'Gabon', 'Gambia', 'Germany', 'Ghana',
           'Greece', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Haiti',
           'Honduras', 'Hong Kong, China', 'Hungary', 'Iceland', 'India',
           'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
           'Jamaica', 'Japan', 'Jordan', 'Kenya', 'Korea, Dem. Rep.',
           'Korea, Rep.', 'Kuwait', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
           'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania',
           'Mauritius', 'Mexico', 'Mongolia', 'Montenegro', 'Morocco',
           'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands',
           'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman',
           'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland',
           'Portugal', 'Puerto Rico', 'Reunion', 'Romania', 'Rwanda',
           'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
           'Sierra Leone', 'Singapore', 'Slovak Republic', 'Slovenia',
           'Somalia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan',
           'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tanzania',
           'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
           'Uganda', 'United Kingdom', 'United States', 'Uruguay', 'Venezuela',
           'Vietnam', 'West Bank and Gaza', 'Yemen, Rep.', 'Zambia', 'Zimbabwe'], dtype=object)



### Exercise
Write a function 'print_stats()' that will print a given country's life expectancy, population and gdp per capita in a given year. (note data is available only for every 5 years between 1952 and 2007).


```python
def print_stats(df,country,year):
    """ Prints the life expectancy, gdp per capita and population
    of country in year. """
    spec = ['lifeExp','pop','gdpPercap']

    print('Statistics for', country, 'in', year)
    print(df[(df['country']==country) & (df['year']==year)][spec])
```


```python
print_stats(df, 'Canada', 2007)
```

    Statistics for Canada in 2007
         lifeExp       pop    gdpPercap
    251   80.653  33390141  36319.23501


## Groupby
We can use the groupby method to split up the data according to repeated values in each column. For example, group the data by continent. This is helpful if we want to repeat an analysis on each group of data from a continent.


```python
continents = df.groupby('continent')
continents
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x7fbac870b128>




```python
len(continents)
```




    5




```python
#helpful way to visualize the groupby object: gives first row of each group
continents.first()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Africa</th>
      <td>Algeria</td>
      <td>1952</td>
      <td>9279525</td>
      <td>43.077</td>
      <td>2449.008185</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>Argentina</td>
      <td>1952</td>
      <td>17876956</td>
      <td>62.485</td>
      <td>5911.315053</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>Afghanistan</td>
      <td>1952</td>
      <td>8425333</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>Albania</td>
      <td>1952</td>
      <td>1282697</td>
      <td>55.230</td>
      <td>1601.056136</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>Australia</td>
      <td>1952</td>
      <td>8691212</td>
      <td>69.120</td>
      <td>10039.595640</td>
    </tr>
  </tbody>
</table>
</div>



### Q:
List the names of the continents and the number of data points in each.


```python
continents.size()
```




    continent
    Africa      624
    Americas    300
    Asia        396
    Europe      360
    Oceania      24
    dtype: int64



### Q:
How many unique countries are there grouped together in the Americas continent?


```python
len(continents.get_group('Americas')['country'].unique())
```




    25



You can use an aggregate function to get the mean life expectancy in the different continents


```python
continents['lifeExp'].mean()
```




    continent
    Africa      48.865330
    Americas    64.658737
    Asia        60.064903
    Europe      71.903686
    Oceania     74.326208
    Name: lifeExp, dtype: float64



The previous cell showed mean life expectancy values aggregated over all the years.

Alternatively, we can groupby multiple columns and use an aggregate function to get the mean life expectancy/population/gdpPercap in a specific continent in a specific year of interest:


```python
df.groupby(['continent', 'year']).agg(np.mean)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>pop</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
    <tr>
      <th>continent</th>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="12" valign="top">Africa</th>
      <th>1952</th>
      <td>4.570010e+06</td>
      <td>39.135500</td>
      <td>1252.572466</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>5.093033e+06</td>
      <td>41.266346</td>
      <td>1385.236062</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>5.702247e+06</td>
      <td>43.319442</td>
      <td>1598.078825</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>6.447875e+06</td>
      <td>45.334538</td>
      <td>2050.363801</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>7.305376e+06</td>
      <td>47.450942</td>
      <td>2339.615674</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>8.328097e+06</td>
      <td>49.580423</td>
      <td>2585.938508</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>9.602857e+06</td>
      <td>51.592865</td>
      <td>2481.592960</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>1.105450e+07</td>
      <td>53.344788</td>
      <td>2282.668991</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>1.267464e+07</td>
      <td>53.629577</td>
      <td>2281.810333</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1.430448e+07</td>
      <td>53.598269</td>
      <td>2378.759555</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1.603315e+07</td>
      <td>53.325231</td>
      <td>2599.385159</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>1.787576e+07</td>
      <td>54.806038</td>
      <td>3089.032605</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">Americas</th>
      <th>1952</th>
      <td>1.380610e+07</td>
      <td>53.279840</td>
      <td>4079.062552</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>1.547816e+07</td>
      <td>55.960280</td>
      <td>4616.043733</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>1.733081e+07</td>
      <td>58.398760</td>
      <td>4901.541870</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>1.922986e+07</td>
      <td>60.410920</td>
      <td>5668.253496</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>2.117537e+07</td>
      <td>62.394920</td>
      <td>6491.334139</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>2.312271e+07</td>
      <td>64.391560</td>
      <td>7352.007126</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>2.521164e+07</td>
      <td>66.228840</td>
      <td>7506.737088</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>2.731016e+07</td>
      <td>68.090720</td>
      <td>7793.400261</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>2.957096e+07</td>
      <td>69.568360</td>
      <td>8044.934406</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>3.187602e+07</td>
      <td>71.150480</td>
      <td>8889.300863</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>3.399091e+07</td>
      <td>72.422040</td>
      <td>9287.677107</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>3.595485e+07</td>
      <td>73.608120</td>
      <td>11003.031625</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">Asia</th>
      <th>1952</th>
      <td>4.228356e+07</td>
      <td>46.314394</td>
      <td>5195.484004</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>4.735699e+07</td>
      <td>49.318544</td>
      <td>5787.732940</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>5.140476e+07</td>
      <td>51.563223</td>
      <td>5729.369625</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>5.774736e+07</td>
      <td>54.663640</td>
      <td>5971.173374</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>6.518098e+07</td>
      <td>57.319269</td>
      <td>8187.468699</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>7.225799e+07</td>
      <td>59.610556</td>
      <td>7791.314020</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>7.909502e+07</td>
      <td>62.617939</td>
      <td>7434.135157</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>8.700669e+07</td>
      <td>64.851182</td>
      <td>7608.226508</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>9.494825e+07</td>
      <td>66.537212</td>
      <td>8639.690248</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1.025238e+08</td>
      <td>68.020515</td>
      <td>9834.093295</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1.091455e+08</td>
      <td>69.233879</td>
      <td>10174.090397</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>1.155138e+08</td>
      <td>70.728485</td>
      <td>12473.026870</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">Europe</th>
      <th>1952</th>
      <td>1.393736e+07</td>
      <td>64.408500</td>
      <td>5661.057435</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>1.459635e+07</td>
      <td>66.703067</td>
      <td>6963.012816</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>1.534517e+07</td>
      <td>68.539233</td>
      <td>8365.486814</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>1.603930e+07</td>
      <td>69.737600</td>
      <td>10143.823757</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>1.668784e+07</td>
      <td>70.775033</td>
      <td>12479.575246</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>1.723882e+07</td>
      <td>71.937767</td>
      <td>14283.979110</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>1.770890e+07</td>
      <td>72.806400</td>
      <td>15617.896551</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>1.810314e+07</td>
      <td>73.642167</td>
      <td>17214.310727</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>1.860476e+07</td>
      <td>74.440100</td>
      <td>17061.568084</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1.896480e+07</td>
      <td>75.505167</td>
      <td>19076.781802</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1.927413e+07</td>
      <td>76.700600</td>
      <td>21711.732422</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>1.953662e+07</td>
      <td>77.648600</td>
      <td>25054.481636</td>
    </tr>
    <tr>
      <th rowspan="12" valign="top">Oceania</th>
      <th>1952</th>
      <td>5.343003e+06</td>
      <td>69.255000</td>
      <td>10298.085650</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>5.970988e+06</td>
      <td>70.295000</td>
      <td>11598.522455</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>6.641759e+06</td>
      <td>71.085000</td>
      <td>12696.452430</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>7.300207e+06</td>
      <td>71.310000</td>
      <td>14495.021790</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>8.053050e+06</td>
      <td>71.910000</td>
      <td>16417.333380</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>8.619500e+06</td>
      <td>72.855000</td>
      <td>17283.957605</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>9.197425e+06</td>
      <td>74.290000</td>
      <td>18554.709840</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>9.787208e+06</td>
      <td>75.320000</td>
      <td>20448.040160</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>1.045983e+07</td>
      <td>76.945000</td>
      <td>20894.045885</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1.112072e+07</td>
      <td>78.190000</td>
      <td>24024.175170</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1.172741e+07</td>
      <td>79.740000</td>
      <td>26938.778040</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>1.227497e+07</td>
      <td>80.719500</td>
      <td>29810.188275</td>
    </tr>
  </tbody>
</table>
</div>



You can also retrieve a particular group with the get_group() command.


```python
continents.get_group('Africa').describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>pop</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>624.00000</td>
      <td>6.240000e+02</td>
      <td>624.00000</td>
      <td>624.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1979.50000</td>
      <td>9.916003e+06</td>
      <td>48.86533</td>
      <td>2193.754578</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17.27411</td>
      <td>1.549092e+07</td>
      <td>9.15021</td>
      <td>2827.929863</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1952.00000</td>
      <td>6.001100e+04</td>
      <td>23.59900</td>
      <td>241.165877</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1965.75000</td>
      <td>1.342075e+06</td>
      <td>42.37250</td>
      <td>761.247010</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1979.50000</td>
      <td>4.579311e+06</td>
      <td>47.79200</td>
      <td>1192.138217</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1993.25000</td>
      <td>1.080149e+07</td>
      <td>54.41150</td>
      <td>2377.417422</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2007.00000</td>
      <td>1.350312e+08</td>
      <td>76.44200</td>
      <td>21951.211760</td>
    </tr>
  </tbody>
</table>
</div>



### Q:
What is the maximum life expectancy for a country in Asia?


```python
continents.get_group('Asia').lifeExp.max()
```




    82.602999999999994



What country is this? When was the measurement taken? We can figure this out in a few different ways:


```python
continents.get_group('Asia').lifeExp.idxmax()
```




    803




```python
#idxmax convenience function will return the index with max value
df[df['continent']=='Asia']['lifeExp'].idxmax()
```




    803




```python
df.loc[803]
```




    country            Japan
    year                2007
    pop          1.27468e+08
    continent           Asia
    lifeExp           82.603
    gdpPercap        31656.1
    Name: 803, dtype: object



How can we rank each country based on their lifeExp?

Let's create a new column 'lifeExp_rank' that creates an ordered ranking based on the longest life expectancy.


```python
sorted_by_lifeExp = df.sort_values('lifeExp', ascending=False)
```


```python
sorted_by_lifeExp['lifeExp_rank'] = np.arange(len(sorted_by_lifeExp)) + 1
```


```python
#lists all rows in order of lifeExp
sorted_by_lifeExp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
      <th>lifeExp_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>803</th>
      <td>Japan</td>
      <td>2007</td>
      <td>127467972</td>
      <td>Asia</td>
      <td>82.603</td>
      <td>31656.06806</td>
      <td>1</td>
    </tr>
    <tr>
      <th>671</th>
      <td>Hong Kong, China</td>
      <td>2007</td>
      <td>6980412</td>
      <td>Asia</td>
      <td>82.208</td>
      <td>39724.97867</td>
      <td>2</td>
    </tr>
    <tr>
      <th>802</th>
      <td>Japan</td>
      <td>2002</td>
      <td>127065841</td>
      <td>Asia</td>
      <td>82.000</td>
      <td>28604.59190</td>
      <td>3</td>
    </tr>
    <tr>
      <th>695</th>
      <td>Iceland</td>
      <td>2007</td>
      <td>301931</td>
      <td>Europe</td>
      <td>81.757</td>
      <td>36180.78919</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1487</th>
      <td>Switzerland</td>
      <td>2007</td>
      <td>7554661</td>
      <td>Europe</td>
      <td>81.701</td>
      <td>37506.41907</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### split,apply and combine: the power of groupby
What if we want to rank each country by max life expectancy for each year that data was collected?

Applying a function on grouped selections can simplify this process:


```python
def ranker(df):
    """Assigns a rank to each country based on lifeExp, with 1 having the highest lifeExp.
    Assumes the data is DESC sorted by lifeExp."""
    df['lifeExp_rank'] = np.arange(len(df)) + 1
    return df
```


```python
#apply the ranking function on a per year basis:
sorted_by_lifeExp = sorted_by_lifeExp.groupby('year').apply(ranker)
```

We can now subset my new dataframe by year to view the lifeExp ranks for each year


```python
sorted_by_lifeExp[sorted_by_lifeExp.year == 2007].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
      <th>lifeExp_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>803</th>
      <td>Japan</td>
      <td>2007</td>
      <td>127467972</td>
      <td>Asia</td>
      <td>82.603</td>
      <td>31656.06806</td>
      <td>1</td>
    </tr>
    <tr>
      <th>671</th>
      <td>Hong Kong, China</td>
      <td>2007</td>
      <td>6980412</td>
      <td>Asia</td>
      <td>82.208</td>
      <td>39724.97867</td>
      <td>2</td>
    </tr>
    <tr>
      <th>695</th>
      <td>Iceland</td>
      <td>2007</td>
      <td>301931</td>
      <td>Europe</td>
      <td>81.757</td>
      <td>36180.78919</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1487</th>
      <td>Switzerland</td>
      <td>2007</td>
      <td>7554661</td>
      <td>Europe</td>
      <td>81.701</td>
      <td>37506.41907</td>
      <td>4</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Australia</td>
      <td>2007</td>
      <td>20434176</td>
      <td>Oceania</td>
      <td>81.235</td>
      <td>34435.36744</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



We can also subset by country=='Canada' to see how Canada's ranking has changed over the years:


```python
sorted_by_lifeExp[(sorted_by_lifeExp['country']=='Canada')]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
      <th>lifeExp_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>251</th>
      <td>Canada</td>
      <td>2007</td>
      <td>33390141</td>
      <td>Americas</td>
      <td>80.653</td>
      <td>36319.23501</td>
      <td>10</td>
    </tr>
    <tr>
      <th>250</th>
      <td>Canada</td>
      <td>2002</td>
      <td>31902268</td>
      <td>Americas</td>
      <td>79.770</td>
      <td>33328.96507</td>
      <td>9</td>
    </tr>
    <tr>
      <th>249</th>
      <td>Canada</td>
      <td>1997</td>
      <td>30305843</td>
      <td>Americas</td>
      <td>78.610</td>
      <td>28954.92589</td>
      <td>10</td>
    </tr>
    <tr>
      <th>248</th>
      <td>Canada</td>
      <td>1992</td>
      <td>28523502</td>
      <td>Americas</td>
      <td>77.950</td>
      <td>26342.88426</td>
      <td>5</td>
    </tr>
    <tr>
      <th>247</th>
      <td>Canada</td>
      <td>1987</td>
      <td>26549700</td>
      <td>Americas</td>
      <td>76.860</td>
      <td>26626.51503</td>
      <td>6</td>
    </tr>
    <tr>
      <th>246</th>
      <td>Canada</td>
      <td>1982</td>
      <td>25201900</td>
      <td>Americas</td>
      <td>75.760</td>
      <td>22898.79214</td>
      <td>8</td>
    </tr>
    <tr>
      <th>245</th>
      <td>Canada</td>
      <td>1977</td>
      <td>23796400</td>
      <td>Americas</td>
      <td>74.210</td>
      <td>22090.88306</td>
      <td>9</td>
    </tr>
    <tr>
      <th>244</th>
      <td>Canada</td>
      <td>1972</td>
      <td>22284500</td>
      <td>Americas</td>
      <td>72.880</td>
      <td>18970.57086</td>
      <td>9</td>
    </tr>
    <tr>
      <th>243</th>
      <td>Canada</td>
      <td>1967</td>
      <td>20819767</td>
      <td>Americas</td>
      <td>72.130</td>
      <td>16076.58803</td>
      <td>7</td>
    </tr>
    <tr>
      <th>242</th>
      <td>Canada</td>
      <td>1962</td>
      <td>18985849</td>
      <td>Americas</td>
      <td>71.300</td>
      <td>13462.48555</td>
      <td>7</td>
    </tr>
    <tr>
      <th>241</th>
      <td>Canada</td>
      <td>1957</td>
      <td>17010154</td>
      <td>Americas</td>
      <td>69.960</td>
      <td>12489.95006</td>
      <td>10</td>
    </tr>
    <tr>
      <th>240</th>
      <td>Canada</td>
      <td>1952</td>
      <td>14785584</td>
      <td>Americas</td>
      <td>68.750</td>
      <td>11367.16112</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



## Visualization

Make sure you use the following %magic command to allow for inline plotting


```python
%matplotlib inline
```

We can specify the type of plot with the kind argument. Also, choose the independent and dependent variables with x and y arguments.


* Plot year vs life expectancy in a scatter plot.


```python
df.plot(x='year',y='lifeExp',kind='scatter')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbac8710b00>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_76_1.png)


- Plot gdp per capita vs life expectancy in a scatter plot


```python
df.plot(x='gdpPercap',y='lifeExp',kind='scatter', alpha = 0.2, s=50, marker='o')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbac86e87f0>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_78_1.png)


What's going on with those points on the right?

High gdp per capita, yet not particularly high lifeExp. We can use boolean selection to rapidly subset and check them out.


```python
df[df['gdpPercap'] > 55000]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>852</th>
      <td>Kuwait</td>
      <td>1952</td>
      <td>160000</td>
      <td>Asia</td>
      <td>55.565</td>
      <td>108382.35290</td>
    </tr>
    <tr>
      <th>853</th>
      <td>Kuwait</td>
      <td>1957</td>
      <td>212846</td>
      <td>Asia</td>
      <td>58.033</td>
      <td>113523.13290</td>
    </tr>
    <tr>
      <th>854</th>
      <td>Kuwait</td>
      <td>1962</td>
      <td>358266</td>
      <td>Asia</td>
      <td>60.470</td>
      <td>95458.11176</td>
    </tr>
    <tr>
      <th>855</th>
      <td>Kuwait</td>
      <td>1967</td>
      <td>575003</td>
      <td>Asia</td>
      <td>64.624</td>
      <td>80894.88326</td>
    </tr>
    <tr>
      <th>856</th>
      <td>Kuwait</td>
      <td>1972</td>
      <td>841934</td>
      <td>Asia</td>
      <td>67.712</td>
      <td>109347.86700</td>
    </tr>
    <tr>
      <th>857</th>
      <td>Kuwait</td>
      <td>1977</td>
      <td>1140357</td>
      <td>Asia</td>
      <td>69.343</td>
      <td>59265.47714</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.hist(column='lifeExp')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7fbac5fb3b00>]], dtype=object)




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_81_1.png)



```python
df.lifeExp.plot.hist(bins=200)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbac5fa1a90>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_82_1.png)



```python
df['lifeExp'].plot(kind='kde')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbac5e5a588>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_83_1.png)


## Exercise
Write a function that will take two countries as an argument and plot the life expectancy vs year for each country on the same axis.


```python
def compare_lifeExp(country1, country2):
    """Plot life expectancy vs year for country1 and country2"""
    ax = plt.subplot()
    for c in [country1,country2]:
        df[df['country']==c].plot(x='year',y='lifeExp', ax=ax)
    plt.legend((country1,country2))
```


```python
compare_lifeExp('Canada', 'Mexico')
```


![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_86_0.png)


## Exercises

Suzy wrote some code to determine which country had the lowest life expectancy in 1982.

What is wrong with her solution?


```python
spec=['country','lifeExp']
df[df['year']==1982][spec].min()
```




    country    Afghanistan
    lifeExp         38.445
    dtype: object



We can do a quick check to look up Afghanistan's life expectancy in 1982.


```python
df[(df['year']==1982) & (df['country']=='Afghanistan')]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Afghanistan</td>
      <td>1982</td>
      <td>12881816</td>
      <td>Asia</td>
      <td>39.854</td>
      <td>978.011439</td>
    </tr>
  </tbody>
</table>
</div>



This doesnt match with the answer above because the min() function was applied to each column (country and lifeExp).

She should have done this:


```python
df.loc[df[df['year']==1982]['lifeExp'].idxmin()]['country']
```




    'Sierra Leone'



### Putting it together:

We can use all of these ideas to generate a plot that looks at a subset of the data.

* Plot GDP per capita vs life expectancy in 2007 for each continent.


```python
continents = df.groupby(['continent'])
for continent in continents.groups:
    group = continents.get_group(continent)
    group[group['year']==2007].plot(kind='scatter', x='gdpPercap',
                                    y='lifeExp', title=continent)
    plt.axis([-10000,60000,30,90])
```


![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_94_0.png)



![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_94_1.png)



![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_94_2.png)



![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_94_3.png)



![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_94_4.png)



```python
#Example
fig,ax = plt.subplots(1,1)
colours = ['m','b','r','g','y']
for continent, colour in zip(continents.groups, colours):
    group = continents.get_group(continent)
    group[group['year']==2007].plot(kind='scatter',x='gdpPercap',y='lifeExp',label=continent,ax=ax,color=colour,alpha=0.5)
ax.set_title(2007)
plt.legend(loc='lower right')
```




    <matplotlib.legend.Legend at 0x7f5791e0db00>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_95_1.png)


### Exercise
Write a function the takes a country as an argument and plots the life expectancy against GDP per capita for all years in a scatter plot. Also print the year of the minimum/maximum lifeExp and the year of the miniimim/maximum GDP per capita.


```python
def compare_gdp_lifeExp(df,country):
    """ plot GDP per capita against life expectancy for a given country.
    print year of min/max gdp per capita and life expectancy
    """

    sub = df[df['country']==country]
    sub.plot(x='gdpPercap',y='lifeExp',kind='scatter',title=country)

    print('Year of Min/Max GDP per capita')
    print(df.iloc[[sub['gdpPercap'].idxmin(),sub['gdpPercap'].idxmax()]]['year'])
    print('Year of Min/Max life expectancy')
    print(df.iloc[[sub['lifeExp'].idxmin(),sub['lifeExp'].idxmax()]]['year'])
```


```python
compare_gdp_lifeExp(df,'Zimbabwe')
```

    Year of Min/Max GDP per capita
    1692    1952
    1696    1972
    Name: year, dtype: int64
    Year of Min/Max life expectancy
    1702    2002
    1699    1987
    Name: year, dtype: int64



![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_98_1.png)



```python
compare_gdp_lifeExp(df,'Canada')
```

    Year of Min/Max GDP per capita
    240    1952
    251    2007
    Name: year, dtype: int64
    Year of Min/Max life expectancy
    240    1952
    251    2007
    Name: year, dtype: int64



![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_99_1.png)


## Rapid plotting with seaborn


```python
import seaborn as sns
```

    /home/derek/anaconda2/envs/py3/lib/python3.5/site-packages/IPython/html.py:14: ShimWarning: The `IPython.html` package has been deprecated. You should import from `notebook` instead. `IPython.html.widgets` has moved to `ipywidgets`.
      "`IPython.html.widgets` has moved to `ipywidgets`.", ShimWarning)



```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>pop</th>
      <th>continent</th>
      <th>lifeExp</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1952</td>
      <td>8425333</td>
      <td>Asia</td>
      <td>28.801</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1957</td>
      <td>9240934</td>
      <td>Asia</td>
      <td>30.332</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>1962</td>
      <td>10267083</td>
      <td>Asia</td>
      <td>31.997</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>1967</td>
      <td>11537966</td>
      <td>Asia</td>
      <td>34.020</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>1972</td>
      <td>13079460</td>
      <td>Asia</td>
      <td>36.088</td>
      <td>739.981106</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_context("talk")
sns.factorplot(data=df, x='year', y='lifeExp', hue='continent', size=8)
```




    <seaborn.axisgrid.FacetGrid at 0x7f57901cfa20>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_103_1.png)



```python
sns.regplot(data=df, x='year', y='gdpPercap', fit_reg=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fbabf5472b0>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_104_1.png)



```python
sns.lmplot(data=df, x='year', y='gdpPercap', hue='continent')
```




    <seaborn.axisgrid.FacetGrid at 0x7f579197f710>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_105_1.png)



```python
sns.lmplot(data=df, x='year', y='gdpPercap', row='continent')
```




    <seaborn.axisgrid.FacetGrid at 0x7f578da9eb38>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_106_1.png)



```python
sns.factorplot(data=df, x='continent', y='gdpPercap', kind='bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f5791ac21d0>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_107_1.png)



```python
g = sns.FacetGrid(df, col='continent', row='year')
g.map(plt.hist, 'lifeExp')
```




    <seaborn.axisgrid.FacetGrid at 0x7f578d928eb8>




![png]({filename}images/UofT-pandas-final_files/UofT-pandas-final_108_1.png)

