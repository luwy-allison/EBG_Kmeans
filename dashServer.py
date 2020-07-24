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
from lrtest import*

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
                                                                dcc.Graph(id='ElbowMethodPlot',style={'display': 'inline-block'})
                                                           ])
                                              ])
                                 ]),
                         dcc.Tab(label="kmeans results",
                                 children=[
                                     html.Div(style={'padding':'10px', 'text-align':'center'},
                                              children=[
                                                  'cluster number ',
                                                  dcc.Input(id='kChoose', type='number', min=1, max=12, value=1),
                                                  html.Br(),
                                                  html.Br(),
                                                  dcc.Store(id='clusterBehaviorCount'),
                                                  dcc.Store(id='behaviorMean'),
                                                  html.Div(style={'display':'flex',
                                                                  'padding':'10px',
                                                                  'text-align':'center'},
                                                           children=[html.Div(id='clusterBarPlot', style={'text-align':'center',
                                                                                                          'padding':'10px'}),
                                                                     html.Div(id='dataSummary', style={'text-align':'center',
                                                                                                       'padding':'10px'}),
                                                                     html.Div(id='clusterSubj',style={'text-align':'center',
                                                                                                      'padding':'10px'})])
                                                  
                                              ])

                                 ]), 
                         dcc.Tab(label="hkmeans results",
                                 children=[
                                     html.Div(style={'text-align':'center'},
                                              children=[
                                                  html.Br(),
                                                  'cluster number ',
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
                                                                                                         'padding':'20px'}),
                                                                     html.Div(id='hkClusterSubj', style={'text-align':'center',
                                                                                                         'padding':'20px'})])
                                              ])
                                 ]),
                         dcc.Tab(label="likelihood ratio test", 
                                 children=[
                                     html.Div(style={'display':'flex', 'text-align':'center'},
                                              children=[
                                                  html.Div(id='modelOption',
                                                           style={'width':'400px',
                                                                  'background-color':'#F5F5F5',
                                                                  'margin-right':'20px',
                                                                  'padding':'20px'},
                                                           children=[
                                                               html.Font(children=['condition']),
                                                               html.Br(),
                                                               dcc.Dropdown(id='compareCondition', 
                                                                            options=[{'label':'intra-condition','value':'intra'}, 
                                                                                     {'label':'inter-condition','value':'inter'}],
                                                                            value='intra'),
                                                               html.Br(),
                                                               'method',
                                                               html.Br(),
                                                               dcc.Dropdown(id='method',
                                                                            options=[{'label':'kmeans','value':'kmeans'},
                                                                                     {'label':'hkmeans','value':'hkmeans'}],
                                                                            value='kmeans'),
                                                               html.Br(),
                                                               'cluster',
                                                               html.Br(),
                                                               dcc.Input(id='lrtCluster', type='number', min=1, max=10, value=1),
                                                               html.Br(),
                                                               html.Br(),
                                                               'scenario',
                                                               html.Br(),
                                                               dcc.Dropdown(id='scenarioChoose',
                                                                            options=[{'label':'more','value':0},
                                                                                     {'label':'same','value':3},
                                                                                     {'label':'less','value':6}
                                                                                    ],
                                                                            value=0),
                                                               html.Br(),
                                                               'general model',
                                                               html.Br(),
                                                               dcc.Dropdown(id='general_model',
                                                                            options=[{'label':'df=2(not given)','value':'typical(not given)'},
                                                                                     {'label':'df=1(buy=no trade)','value':'equal(0,1)'},
                                                                                     {'label':'df=1(no trade=sell)','value':'equal(1,2)'},
                                                                                     {'label':'df=1(buy=sell)','value':'equal(0,2)'}]),
                                                               html.Br(),
                                                               'restrict model',
                                                               html.Br(),
                                                               dcc.Dropdown(id='restrict_model',
                                                                            options=[{'label':'df=1(buy=no_trade)','value':'equal(0,1)'},
                                                                                     {'label':'df=1(no_trade=sell)','value':'equal(1,2)'},
                                                                                     {'label':'df=1(buy=sell)','value':'equal(0,2)'},
                                                                                     {'label':'df=1(buy assigned)','value':'assigned(0)'},
                                                                                     {'label':'df=1(no_trade assigned)','value':'assigned(1)'},
                                                                                     {'label':'df=1(sell assigned)','value':'assigned(2)'},
                                                                                     {'label':'df=0(all assigned)','value':'typical(given all)'}]),
                                                               html.Br(),
                                                               html.Div(id='assigning'),
                                                               dcc.Store(id='assignedValues')
                                                               
                                                           ]),
                                                  html.Div(id='lrtestResult',
                                                           style={'width':'1200px',
                                                                  'background-color':'#F5FFFA',
                                                                  'padding':'20px'},
                                                           children=[
                                                                dcc.Graph(id='clusterBar',style={'display': 'inline-block'}),
                                                                dcc.Graph(id='lrtestBar',style={'display': 'inline-block'}),
                                                                html.Div(id='lrtestPvalue')
                                                           ])
                                              ])
                                 ])
                     ])
                 ])
             ])
])

@app.callback(Output(component_id='assigning',component_property='children'),
              [Input(component_id='restrict_model',component_property='value')])
def update_assigning_option(mode): 
    
    if mode == 'assigned(0)':
        options = [dcc.Input(id='assign_p', type='number', min=0, max=1, value=0.5)]
    elif mode == 'assigned(1)':
        options = [dcc.Input(id='assign_q', type='number', min=0, max=1, value=0.5)]
    elif mode == 'assigned(2)':
        options = [dcc.Input(id='assign_r', type='number', min=0, max=1, value=0.5)]
    elif mode == 'typical(given all)':
        options = []
        options.append(dcc.Input(id='assign_p', type='number', min=0, max=1, value=0.2))
        options.append(dcc.Input(id='assign_q', type='number', min=0, max=1, value=0.3))
        options.append(dcc.Input(id='assign_r', type='number', min=0, max=1, value=0.5))
    else:
        options = None
    return options
    
@app.callback(Output(component_id='assignedValues',component_property='data'),
              [Input(component_id='assigning',component_property='children')])
def store_assigning_values(assigned_val):

    if assigned_val != None:
        print(assigned_val)
        print(len(assigned_val))
        if len(assigned_val)>1:
            storeVal = []
            for i in range(len(assigned_val)):
                # print(assigned_val[i])
                storeVal.append(assigned_val[i]['props']['value'])
        else:
            storeVal = assigned_val[0]['props']['value']
    else:
        storeVal = None
    return storeVal
# @app.callback(Output(component_id=''),
#               Input(component_id='compareCondition',component_property='value'))
# def update_model_selection(compare_mode):
#     if compare_mode =='intra':
        
@app.callback([Output(component_id='clusterBar',component_property='figure'),
               Output(component_id='lrtestBar',component_property='figure'),
               Output(component_id='lrtestPvalue',component_property='children')],
              [Input(component_id='compareCondition',component_property='value'),
               Input(component_id='method',component_property='value'),
               Input(component_id='lrtCluster',component_property='value'),
               Input(component_id='behaviorMean',component_property='data'),
               Input(component_id='clusterBehaviorCount',component_property='data'),
               Input(component_id='scenarioChoose',component_property='value'),
               Input(component_id='general_model',component_property='value'),
               Input(component_id='restrict_model',component_property='value'),
               Input(component_id='assignedValues',component_property='data')])
def update_lrtest(compare_mode,method,cluster,behaviorMean,clusterBehaviorCount,scenario,gen_model,res_model,assignedValues):
    clusterBarFig = go.Figure(data=[
        go.Bar(name='buy', x=['more','same','less'], y=[behaviorMean[cluster][0],behaviorMean[cluster][3],behaviorMean[cluster][6]]),
        go.Bar(name='no trade', x=['more','same','less'], y=[behaviorMean[cluster][1],behaviorMean[cluster][4],behaviorMean[cluster][7]]),
        go.Bar(name='sell', x=['more','same','less'], y=[behaviorMean[cluster][2],behaviorMean[cluster][5],behaviorMean[cluster][8]])    
    ])
    clusterBarFig.update_layout(barmode='group', width=700, height=400)
    print(assignedValues)
    if gen_model =='typical(not given)':
        gen_par=()
#         gen_assignedValues=None
    elif gen_model=='equal(0,1)':
        gen_par=(0,1)
#         gen_assignedValues=None
    elif gen_model=='equal(0,2)':
        gen_par=(0,2)
#         gen_assignedValues=None
    elif gen_model=='equal(1,2)':
        gen_par=(1,2)
#         gen_assignedValues=None
        
    if res_model=='equal(0,1)':
        res_par=(0,1)
        res_assignedValues=None
    elif res_model=='equal(0,2)':
        res_par=(0,2)
        res_assignedValues=None
    elif res_model=='equal(1,2)':
        res_par=(1,2)
        res_assignedValues=None
    elif res_model=='assigned(0)':
        res_par=(0,)
        res_assignedValues = assignedValues
    elif res_model=='assigned(1)':
        res_par=(1,)
        res_assignedValues = assignedValues
    elif res_model=='assigned(2)':
        res_par=(2,)
        res_assignedValues = assignedValues
    elif res_model=='typical(given all)':
        res_par=(1,2,3)
        res_assignedValues = assignedValues
    scenario = int(scenario)
    general = toLikelihood(behaviorMean[cluster][scenario:scenario+3], count = clusterBehaviorCount[cluster][scenario:scenario+3], parameterChoose=gen_par, assignedValue = None)
    restrict = toLikelihood(behaviorMean[cluster][scenario:scenario+3], count = clusterBehaviorCount[cluster][scenario:scenario+3], parameterChoose=res_par, assignedValue = res_assignedValues)
    test_G,test_p = likelihood_ratio_test(general,restrict)
    return clusterBarFig,clusterBarFig,'p value={:.5f}'.format(test_p)

@app.callback(
    [Output(component_id='hkClusterBarPlot',component_property='children'),
     Output(component_id='hkDataSummary',component_property='children'),
     Output(component_id='hkClusterSubj',component_property='children')],
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
    summary_list=[]
    table_list=[]
    for n_cluster in range(1,hkChoose+1):
        summary_list.append(html.Br())
        table_list.append(html.Br())
        if n_cluster > 1:
            fig_list.append(html.Br())
            summary_list.append(html.Br())
            table_list.append(html.Br())

        clusterSummary=pd.DataFrame()
        clusterSummary["Scenario"]=["more","more","more","same","same","same","less","less","less"]
        clusterSummary["Action"]=["buy","no trade","sell","buy","no trade","sell","buy","no trade","sell"]
        clusterSummary["n"]=memberCount[n_cluster-1]
        clusterSummary["Mean"]=behaviorMean[n_cluster]
        summary_list.append(dash_table.DataTable(style_table={'height':'400px',
                                                            'width':'400px'},
                                                 style_cell={'textAlign': 'center',
                                                           'padding': '5px'},
                                                 style_header={'fontWeight':'bold'},
                                                 columns=[{"name": i, "id": i} for i in clusterSummary.columns],data=clusterSummary.to_dict('records')))
        clusterSubj=data.wide
        fig = go.Figure(data=[
            go.Bar(name='buy', x=['more','same','less'], y=[behaviorMean[n_cluster][0],behaviorMean[n_cluster][3],behaviorMean[n_cluster][6]]),
            go.Bar(name='no trade', x=['more','same','less'], y=[behaviorMean[n_cluster][1],behaviorMean[n_cluster][4],behaviorMean[n_cluster][7]]),
            go.Bar(name='sell', x=['more','same','less'], y=[behaviorMean[n_cluster][2],behaviorMean[n_cluster][5],behaviorMean[n_cluster][8]])    
        ])
        fig.update_layout(barmode='group', width=700, height=400)
        fig_list.append(dcc.Graph(figure=fig))
    
    return fig_list, summary_list, table_list
    
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
     Output(component_id='dataSummary',component_property='children'),
     Output(component_id='behaviorMean',component_property='data'),
     Output(component_id='clusterBehaviorCount',component_property='data')],
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
    clusterBehaviorCount=[["more.buy","more.no trade","more.sell","equal.buy","equal.no trade","equal.sell","less.buy","less.no trade","less.sell"]]
    for i in range(kChoose):
        clusterBehavior.append([0,0,0,0,0,0,0,0,0])
        clusterBehaviorCount.append([0,0,0,0,0,0,0,0,0])
    memberCount=[0]*kChoose
    for i in range(len(data.label)):
        for k in range(kChoose):
            if data.label[i][19] == k:
                memberCount[k]+=1
                for col in range(1,10):
                    clusterBehaviorCount[k+1][col-1]+=data.label[i][col]
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
    return fig_list,table_list,behaviorMean,clusterBehaviorCount

@app.callback(Output(component_id='lrtCluster',component_property='max'),
              [Input(component_id='kChoose',component_property='value'),
               Input(component_id='hkChoose',component_property='value'),
               Input(component_id='method',component_property='value')])
def clusterNum(knum,hknum,method):
    if method == 'kmeans':
        clusterMax = knum
    elif method == 'hkmeans':
        clusterMax = hknum
    else:
        print('method error')
    return clusterMax

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)