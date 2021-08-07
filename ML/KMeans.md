# KMeans

## 1 - 'clustering with more than two features'

### data
[e-commerce dataset](https://github.com/MauricioLetelier/E-commerce-Clustering)

### analysing

- visualizing

first showing the correaltion between variables
```python
import plotly.express as px
fig = px.scatter_matrix(df.drop("ID",axis=1),
width=1200, height=1600)
fig.show()
```

relationship among variables can be shown in different shape
    - n_clicks, how many clicks of the costumers
    - n_visits, how many visit
    - days_since_registration
    - amount_spent

```python
fig1 = px.scatter(df, x="n_clicks", y="n_visits", color="days_since_registration",
                 size="amount_spent")
fig1.update_layout(title="4 Features Representation")
fig1.show()
```

and can be shown in 3D shape
```python
fig2 = px.scatter_3d(df, x="n_clicks", y="n_visits",z="amount_discount",
                     color="days_since_registration",size="amount_spent")
fig2.update_layout(title="5 Features Representation")
fig2.show()
```


- Kmean analysing
the Kmean is by SKlearn

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np
X=df.drop("ID",axis=1)
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
inertia = []
for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),
                  yaxis={'title':'Inertia'},
                 annotations=[
        dict(
            x=3,
            y=inertia[2],
            xref="x",
            yref="y",
            text="Elbow!",
            showarrow=True,
            arrowhead=7,
            ax=20,
            ay=-40
        )
    ])
```
we could see there is 3 or 4 elbows/clusters can be choosen

```python
kmeans = KMeans(
        n_clusters=3, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
kmeans.fit(X)
clusters=pd.DataFrame(X,columns=df.drop("ID",axis=1).columns)
clusters['label']=kmeans.labels_
polar=clusters.groupby("label").mean().reset_index()
polar=pd.melt(polar,id_vars=["label"])
fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=800,width=1400)
fig.show()
```

## 引用
1. ['Clustering With More Than Two Features? Try This To Explain Your Findings'](https://towardsdatascience.com/clustering-with-more-than-two-features-try-this-to-explain-your-findings-b053007d680a)