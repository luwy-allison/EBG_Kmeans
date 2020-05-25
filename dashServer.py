## dash latest(0526 04:19)
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash_table
from dash.dependencies import Input, Output
import math
import pandas as pd
import os
import csv
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from DataObject import*

#"more.buy","more.no trade","more.sell","equal.buy","equal.no trade","equal.sell","less.buy","less.no trade","less.sell"
app = dash.Dash()
app.layout= html.Div([
    html.H2("EBG behavior data"),
    html.Div(style={'background-color':'#F2E9E7', 'text-align':'center'},
             children=[
                 dcc.Tabs(id='condition',
                          value='delta price',
                          children=[
                     dcc.Tab(value='delta price',
                             label="delta Price"),
                     dcc.Tab(value='delta asset',
                             label="delta Asset"),   
                     dcc.Tab(value='delta cash',
                             label="delta Cash")
                 ]),
                 html.Div(children=[
                     dcc.Tabs([
                         dcc.Tab(label="trial choose",
                                 children=[
                                     html.Div(style={'display':'flex', 'text-align':'center'},
                                              children=[
                                                  html.Div(id='settings',
                                                           style={'width':'400px',
                                                                  'background-color':'#F5F5F5',
                                                                  'margin-right':'20px',
                                                                  'padding':'20px'},
                                                           children=[
                                                               html.Font(children=['trial range']),
                                                               html.Br(),
                                                               html.Br(),
                                                               'start__ ',dcc.Input(id='startTrial', type='number', min=1, max=100, value=1),
                                                               html.Br(),
                                                               html.Br(),
                                                               'end__ ',dcc.Input(id='endTrial', type='number', min=1, max=100, value=100)
                                                           ]),
                                                  html.Div(id='plot',
                                                           style={'width':'1200px',
                                                                  'background-color':'#F5FFFA',
                                                                  'padding':'20px'},
                                                           children=[
                                                               'test',
                                                                dcc.Graph(id='ElbowMethodPlot',style={'display': 'inline-block'})
                                                           ])
                                              ])
                                 ]),
                         dcc.Tab(label="kmeans results",
                                 children=[
                                     html.Div(style={'padding':'20px', 'text-align':'center'},
                                              children=[
                                                  dcc.Input(id='kChoose', type='number', min=1, max=12, value=1),
                                                  html.Br(),
                                                  html.Br(),
                                                  html.Div(style={'display':'flex',
                                                                  'padding':'20px',
                                                                  'text-align':'center'},
                                                           children=[html.Div(id='clusterBarPlot', style={'text-align':'center',
                                                                                                          'padding':'20px'}),
                                                                     html.Div(id='dataSummary', style={'text-align':'center',
                                                                                                       'padding':'20px'})])
                                                  
                                              ])

                                 ]), 
                         dcc.Tab(label="hkmeans results",
                                 children=[
                                     html.Div(style={'text-align':'center'},
                                              children=[
                                                  html.Br(),
                                                  'kchoose ',
                                                  dcc.Input(id='hkChoose', type='number', min=1, max=12, value=1),
                                                  html.Br(),html.Br(),
                                                  'treshold ',
                                                  dcc.Input(id='threshold', type='number', min=0, max=10, value=1, step=0.01),
                                                  html.Br(),
                                                  html.Br(),
                                                  html.Div(style={'text-align':'center'},
                                                           children=[dcc.Graph(id='dendrogram',style={'display': 'inline-block'})]),
                                                  html.Div(style={'display':'flex',
                                                                  'padding':'20px',
                                                                  'text-align':'center'},
                                                           children=[html.Div(id='hkClusterBarPlot', style={'text-align':'center',
                                                                                                           'padding':'20px'}),
                                                                     html.Div(id='hkDataSummary', style={'text-align':'center',
                                                                                                         'padding':'20px'})])
                                              ])
                                 ])
                     ])
                 ])
             ])
])

@app.callback(
    Output(component_id="dendrogram",component_property='figure'),
    [Input(component_id='condition',component_property='value'),
     Input(component_id='startTrial',component_property='value'),
     Input(component_id='endTrial',component_property='value'),
     Input(component_id='threshold',component_property='value')])
def update_dendrogram(condition,startTrial,endTrial,threshold):
    data = Data(condition,startTrial,endTrial)
    ### dendrogram
    X=np.asarray(data.ForK_dropna)
    fig = ff.create_dendrogram(X,color_threshold=threshold)
    fig.update_layout(width=1000, height=600, font={'size':8})
    return fig

@app.callback(
    [Output(component_id='hkClusterBarPlot',component_property='children'),
     Output(component_id='hkDataSummary',component_property='children')],
    [Input(component_id='condition',component_property='value'),
     Input(component_id='startTrial',component_property='value'),
     Input(component_id='endTrial',component_property='value'),
     Input(component_id='hkChoose',component_property='value')]
)
def update_hkCluster(condition,startTrial,endTrial,hkChoose):
    data = Data(condition,startTrial,endTrial)
    ### hierachical clustering(hkmeans)
#     hkChoose=4
    X=np.array(data.ForK_dropna)
    hkmeanModel = AgglomerativeClustering(n_clusters=hkChoose).fit(X)

    # Add label to the list
    data.label_pd=data.wide_dropna_pd
    data.label_pd['label_h']=hkmeanModel.labels_
    data.label=[data.label_pd.columns.values.tolist()]+data.label_pd.values.tolist()

    # counting cluster behavior
    # index: [0:"subj",1:"more.buy",2:"more.no trade",3:"more.sell",4:"equal.buy",5:"equal.no trade",6:"equal.sell",7:"less.buy",8:"less.no trade",9:"less.sell",10:"label"]
    clusterBehavior=[["more.buy","more.no trade","more.sell","equal.buy","equal.no trade","equal.sell","less.buy","less.no trade","less.sell"]]
    for i in range(hkChoose):
        clusterBehavior.append([0,0,0,0,0,0,0,0,0])
    memberCount=[0]*hkChoose
    for i in range(len(data.label)):
        for k in range(hkChoose):
            if data.label[i][19] == k:
                memberCount[k]+=1
                for col in range(1,10):
                    clusterBehavior[k+1][col-1]+=data.label[i][col+9]
    # calculate behavior ratio
    behaviorMean=[]
    behaviorMean.append(clusterBehavior[0])
    for i in range(1,hkChoose+1):
        temp=[]
        print(i)
        for j in range(0,9):
            temp.append(clusterBehavior[i][j]/memberCount[i-1])
        behaviorMean.append(temp)
    # plot cluster behavior ratio bar plot
    fig_list=[]
    table_list=[]
    for n_cluster in range(1,hkChoose+1):
        table_list.append(html.Br())
        if n_cluster > 1:
            fig_list.append(html.Br())
            table_list.append(html.Br())

        clusterSummary=pd.DataFrame()
        clusterSummary["Scenario"]=["more","more","more","same","same","same","less","less","less"]
        clusterSummary["Action"]=["buy","no trade","sell","buy","no trade","sell","buy","no trade","sell"]
        clusterSummary["n"]=memberCount[n_cluster-1]
        clusterSummary["Mean"]=behaviorMean[n_cluster]
        table_list.append(dash_table.DataTable(style_table={'height':'400px',
                                                            'width':'400px'},
                                               style_cell={'textAlign': 'center',
                                                           'padding': '5px'},
                                               style_header={'fontWeight':'bold'},
                                               columns=[{"name": i, "id": i} for i in clusterSummary.columns],data=clusterSummary.to_dict('records')))
        
        fig = go.Figure(data=[
            go.Bar(name='buy', x=['more','same','less'], y=[behaviorMean[n_cluster][0],behaviorMean[n_cluster][3],behaviorMean[n_cluster][6]]),
            go.Bar(name='no trade', x=['more','same','less'], y=[behaviorMean[n_cluster][1],behaviorMean[n_cluster][4],behaviorMean[n_cluster][7]]),
            go.Bar(name='sell', x=['more','same','less'], y=[behaviorMean[n_cluster][2],behaviorMean[n_cluster][5],behaviorMean[n_cluster][8]])    
        ])
        fig.update_layout(barmode='group', width=700, height=400)
        fig_list.append(dcc.Graph(figure=fig))
    
    return fig_list,table_list

@app.callback(
    Output(component_id="ElbowMethodPlot",component_property='figure'),
    [Input(component_id='condition',component_property='value'),
     Input(component_id='startTrial',component_property='value'),
     Input(component_id='endTrial',component_property='value')])
def update_elbowgraph(condition,startTrial,endTrial):
    data = Data(condition,startTrial,endTrial)
    # Elbow's method
    K = []
    for i in range(1,13):
        K.append(i)
    X = np.array(data.ForK_dropna)
    distortions=[]
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)
        
    fig = go.Figure(go.Scatter(x=K, y=distortions, mode='lines+markers'))
    fig.update_layout(title='The Elbow Method showing the optimal k',
                      xaxis_title='k',
                      yaxis_title='Distortion')
    print(distortions)
    return fig

@app.callback(
    [Output(component_id='clusterBarPlot',component_property='children'),
     Output(component_id='dataSummary',component_property='children')],
    [Input(component_id='condition',component_property='value'),
     Input(component_id='startTrial',component_property='value'),
     Input(component_id='endTrial',component_property='value'),
     Input(component_id='kChoose',component_property='value')]
)
def update_clusterGraph(condition,startTrial,endTrial,kChoose):
    data = Data(condition,startTrial,endTrial)
    # Run kmeans after choose k deltaPrice
#     kChoose=4
    X=np.array(data.ForK_dropna)
    kmeanModel = KMeans(n_clusters=kChoose).fit(X)

    # Add label to the list
    data.label_pd=data.wide_dropna_pd
    data.label_pd['label']=kmeanModel.labels_
    data.label=[data.label_pd.columns.values.tolist()]+data.label_pd.values.tolist()
    # counting cluster behavior
    # index: [0:"subj",1:"more.buy",2:"more.no trade",3:"more.sell",4:"equal.buy",5:"equal.no trade",6:"equal.sell",7:"less.buy",8:"less.no trade",9:"less.sell",10:"label"]
    clusterBehavior=[["more.buy","more.no trade","more.sell","equal.buy","equal.no trade","equal.sell","less.buy","less.no trade","less.sell"]]
    for i in range(kChoose):
        clusterBehavior.append([0,0,0,0,0,0,0,0,0])
    memberCount=[0]*kChoose
    for i in range(len(data.label)):
        for k in range(kChoose):
            if data.label[i][19] == k:
                memberCount[k]+=1
                for col in range(1,10):
                    clusterBehavior[k+1][col-1]+=data.label[i][col+9]
    # calculate behavior ratio
    behaviorMean=[]
    behaviorMean.append(clusterBehavior[0])
    for i in range(1,kChoose+1):
        temp=[]
        print(i)
        for j in range(0,9):
            temp.append(clusterBehavior[i][j]/memberCount[i-1])
        behaviorMean.append(temp)
    # plot cluster behavior ratio bar plot
    fig_list=[]
    table_list=[]
    for n_cluster in range(1,kChoose+1):
        table_list.append(html.Br())
        if n_cluster > 1:
            fig_list.append(html.Br())
            table_list.append(html.Br())

        clusterSummary=pd.DataFrame()
        clusterSummary["Scenario"]=["more","more","more","same","same","same","less","less","less"]
        clusterSummary["Action"]=["buy","no trade","sell","buy","no trade","sell","buy","no trade","sell"]
        clusterSummary["n"]=memberCount[n_cluster-1]
        clusterSummary["Mean"]=behaviorMean[n_cluster]
        table_list.append(dash_table.DataTable(style_table={'height':'400px',
                                                            'width':'400px'},
                                               style_cell={'textAlign': 'center',
                                                           'padding': '5px'},
                                               style_header={'fontWeight':'bold'},
                                               columns=[{"name": i, "id": i} for i in clusterSummary.columns],data=clusterSummary.to_dict('records')))
        
        fig = go.Figure(data=[
            go.Bar(name='buy', x=['more','same','less'], y=[behaviorMean[n_cluster][0],behaviorMean[n_cluster][3],behaviorMean[n_cluster][6]]),
            go.Bar(name='no trade', x=['more','same','less'], y=[behaviorMean[n_cluster][1],behaviorMean[n_cluster][4],behaviorMean[n_cluster][7]]),
            go.Bar(name='sell', x=['more','same','less'], y=[behaviorMean[n_cluster][2],behaviorMean[n_cluster][5],behaviorMean[n_cluster][8]])    
        ])
        fig.update_layout(barmode='group', width=700, height=400)
        fig_list.append(dcc.Graph(figure=fig))
    
    return fig_list,table_list

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)