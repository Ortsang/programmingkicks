# groupby
this file explain how to use groupby in pandas

通过对DataFrame对象调用groupby()函数返回的结果是一个DataFrameGroupBy对象，而不是一个DataFrame或者Series对象，所以，它们中的一些方法或者函数是无法直接调用的，需要按照GroupBy对象中具有的函数和方法进行调用。

```python
grouped = df.groupby('Gender') #指定单个列名进行groupby
grouped_muti = df.groupby(['Gender', 'Age']) #指定多个列名进行groupby

df = grouped.get_group('Female').reset_index() #通过调用get_group()函数可以返回一个按照分组得到的DataFrame对象，所以接下来的使用就可以按照·DataFrame·对象来使用。
print(df)
```

- functions
可以对groupby对象使用不同的函数。

```python
print(grouped.count())
print(grouped.max()[['Age', 'Score']])
print(grouped.mean()[['Age', 'Score']])
```