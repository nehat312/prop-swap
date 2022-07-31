#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import dash as dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import plotly as ply
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import scipy.stats as st
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot

import sys
import os
import datetime
import time

import requests
from bs4 import BeautifulSoup
import re

#import json
#import nltk

print("\nIMPORT SUCCESS")

#%%
# CLEAN DATA IMPORT
rollup_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/ncaab_data_rollup_5-2-22'

tr_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'
kp_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/kenpom_pull_3-14-22'

# NOTE: PANDAS OPENPYXL PACKAGE / EXTENSION REQUIRED TO IMPORT .xlsx FILES

# ROLLED-UP DATA
rollup = pd.read_excel(rollup_filepath + '.xlsx', sheet_name='ROLLUP') #index_col='Team'

# TEAMRANKINGS DATA
tr = pd.read_excel(tr_filepath + '.xlsx') #index_col='Team'

# KENPOM DATA
kp = pd.read_excel(kp_filepath + '.xlsx') #index_col='Team'

# HISTORICAL GAME DATA
#historical_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/MNCAAB-historical'
#regular = pd.read_excel(historical_filepath + '.xlsx', sheet_name='REGULAR') #index_col='Team'
#tourney = pd.read_excel(historical_filepath + '.xlsx', sheet_name='TOURNEY') #index_col='Team'

print("\nIMPORT SUCCESS")

#%%
# FINAL PRE-PROCESSING
tr['opponent-stocks-per-game'] = tr['opponent-blocks-per-game'] + tr['opponent-steals-per-game']
rollup['opponent-stocks-per-game'] = rollup['opponent-blocks-per-game'] + rollup['opponent-steals-per-game']

tr = tr.round(2)
rollup = rollup.round(2)

print(tr.head())
#print(tr.info())
#print(tr.index)
#print(tr)

#%%
# REFINED DATAFRAMES - keeping only unique, essential, or most valuable columns from each data set.

tr_df = tr[['Team', 'win-pct-all-games',
               'average-scoring-margin', #'opponent-average-scoring-margin',
               'points-per-game', 'opponent-points-per-game',
               'offensive-efficiency', 'defensive-efficiency', 'net-adj-efficiency',
               'effective-field-goal-pct', 'opponent-effective-field-goal-pct',
               #'true-shooting-percentage',  #'opponent-true-shooting-percentage',
               'three-point-pct', 'two-point-pct', 'free-throw-pct',
               'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
               'assists-per-game', 'opponent-assists-per-game',
               #'turnovers-per-game', 'opponent-turnovers-per-game',
               'assist--per--turnover-ratio', 'opponent-assist--per--turnover-ratio',
               'stocks-per-game', 'opponent-stocks-per-game',
               #'blocks-per-game', 'steals-per-game',
               #'opponent-blocks-per-game','opponent-steals-per-game',
               ]]


rollup_df = rollup[['TR_Team', 'win-pct-all-games',
                    'average-scoring-margin', #'opponent-average-scoring-margin',
                    'points-per-game', 'opponent-points-per-game',
                    'offensive-efficiency', 'defensive-efficiency', 'net-adj-efficiency',
                    'effective-field-goal-pct', 'opponent-effective-field-goal-pct',
                    #'true-shooting-percentage',  #'opponent-true-shooting-percentage',
                    'three-point-pct', 'two-point-pct', 'free-throw-pct',
                    'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
                    'assists-per-game', 'opponent-assists-per-game',
                    #'turnovers-per-game', 'opponent-turnovers-per-game',
                    'assist--per--turnover-ratio', 'opponent-assist--per--turnover-ratio',
                    'stocks-per-game', 'opponent-stocks-per-game',
                    'alias', 'turner_name', 'conf_alias', # 'name', 'school_ncaa',
                    'venue_city', 'venue_state', 'venue_name', 'venue_capacity', #'venue_id', 'GBQ_id',
                    #'logo_large', 'logo_medium', 'logo_small',
                    'mascot', 'mascot_name', 'mascot_common_name', 'tax_species', 'tax_genus', 'tax_family',
                    'tax_order', 'tax_class', 'tax_phylum', 'tax_kingdom', 'tax_domain',
                    'Conference', 'Rank', 'Seed', 'Win', 'Loss',
                    'Adj EM', 'AdjO', 'AdjD', 'AdjT', 'Luck',
                    'SOS Adj EM', 'SOS OppO', 'SOS OppD', 'NCSOS Adj EM'
                    ]]


#%%
# RENAME COLUMNS TO IMPROVE APP OPTICS
app_cols = {'Team': 'TEAM', 'win-pct-all-games':'WIN%',
            'average-scoring-margin':'AVG_MARGIN', #'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'points-per-game': 'PTS/GM',  'opponent-points-per-game':'OPP_PTS/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF', 'net-adj-efficiency':'NET_EFF',
            'effective-field-goal-pct':'EFG%', #'true-shooting-percentage':'TS%',
            'opponent-effective-field-goal-pct':'OPP_EFG%', #'opponent-true-shooting-percentage':'OPP_TS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%',
            'assists-per-game':'AST/GM', 'opponent-assists-per-game':'OPP_AST/GM',
            'assist--per--turnover-ratio':'AST/TO', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'stocks-per-game':'S+B/GM', 'opponent-stocks-per-game':'OPP_S+B/GM',
            #'turnovers-per-game':'TO/GM', 'opponent-turnovers-per-game':'OPP_TO/GM',
            #'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM', 'blocks-per-game':'B/GM', 'steals-per-game':'S/GM',
            }


tr_cols = {'Team': 'TEAM', 'points-per-game':'PTS/GM', 'average-scoring-margin':'AVG_MARGIN', 'win-pct-all-games':'WIN%', 'win-pct-close-games':'WIN%_CLOSE',
            'effective-field-goal-pct':'EFG%', 'true-shooting-percentage':'TS%', 'effective-possession-ratio': 'POSS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'field-goals-made-per-game':'FGM/GM', 'field-goals-attempted-per-game':'FGA/GM', 'three-pointers-made-per-game':'3PM/GM', 'three-pointers-attempted-per-game':'3PA/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF',
            'total-rebounds-per-game':'TRB/GM', 'offensive-rebounds-per-game':'ORB/GM', 'defensive-rebounds-per-game':'DRB/GM',
            'offensive-rebounding-pct':'ORB%', 'defensive-rebounding-pct':'DRB%', 'total-rebounding-percentage':'TRB%',
            'blocks-per-game':'B/GM', 'steals-per-game':'S/GM', 'assists-per-game':'AST/GM', 'turnovers-per-game':'TO/GM',
            'assist--per--turnover-ratio':'AST/TO', 'possessions-per-game':'POSS/GM', 'personal-fouls-per-game':'PF/GM',
            'opponent-points-per-game':'OPP_PTS/GM', 'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'opponent-effective-field-goal-pct':'OPP_EFG%', 'opponent-true-shooting-percentage':'OPP_TS%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%', 'opponent-shooting-pct':'OPP_FG%',
            'opponent-assists-per-game':'OPP_AST/GM', 'opponent-turnovers-per-game':'OPP_TO/GM', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'opponent-offensive-rebounds-per-game':'OPP_OREB/GM', 'opponent-defensive-rebounds-per-game':'OPP_DREB/GM', 'opponent-total-rebounds-per-game':'OPP_TREB/GM',
            'opponent-offensive-rebounding-pct':'OPP_OREB%', 'opponent-defensive-rebounding-pct':'OPP_DREB%',
            'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM',
            'opponent-effective-possession-ratio':'OPP_POSS%',
            'net-avg-scoring-margin':'NET_AVG_MARGIN', 'net-points-per-game':'NET_PTS/GM',
            'net-adj-efficiency':'NET_EFF',
            'net-effective-field-goal-pct':'NET_EFG%', 'net-true-shooting-percentage':'NET_TS%',
            'stocks-per-game':'S+B/GM', 'opponent-stocks-per-game':'OPP_S+B/GM', 'total-turnovers-per-game':'TTL_TO/GM',
            'net-assist--per--turnover-ratio':'NET_AST/TO',
            'net-total-rebounds-per-game':'NET_TREB/GM', 'net-off-rebound-pct':'NET_OREB%', 'net-def-rebound-pct':'NET_DREB%'
            }

#%%
tr_df.columns = tr_df.columns.map(app_cols)

print(tr_df.columns)
print(tr_df.info())

#%%
# DATA ROLLUP - PRE-PROCESSING
print(rollup.info())

#%%
# ROLLUP COLUMNS (FILTERED)
print(rollup_df.columns)
print('*'*100)

# GBQ / KP COLS
#print(f'BIG QUERY / KENPOM DATA:')
#print(rollup_df.columns[63:]) #print(rollup.columns[-40:])

#%%
# DROP NULL VALUES
rollup_df.dropna(inplace=True)
print(rollup_df.info())

#%%
# PRE-PROCESSING ROLLUP FILE
roll_cols = {'TR_Team': 'TEAM', 'win-pct-all-games':'WIN%',
            'average-scoring-margin':'AVG_MARGIN', #'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'points-per-game': 'PTS/GM',  'opponent-points-per-game':'OPP_PTS/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF', 'net-adj-efficiency':'NET_EFF',
            'effective-field-goal-pct':'EFG%', #'true-shooting-percentage':'TS%',
            'opponent-effective-field-goal-pct':'OPP_EFG%', #'opponent-true-shooting-percentage':'OPP_TS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%',
            'assists-per-game':'AST/GM', 'opponent-assists-per-game':'OPP_AST/GM',
            'assist--per--turnover-ratio':'AST/TO', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'stocks-per-game':'S+B/GM', 'opponent-stocks-per-game':'OPP_S+B/GM',
            #'turnovers-per-game':'TO/GM', 'opponent-turnovers-per-game':'OPP_TO/GM',
            #'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM', 'blocks-per-game':'B/GM', 'steals-per-game':'S/GM',
            'alias':'ABBR', 'name':'NICKNAME', 'turner_name':'INSTITUTION', 'conf_alias':'CONF',
            'venue_city':'CITY', 'venue_state':'STATE', 'venue_name':'ARENA', 'venue_capacity':'ARENA_CAP',
            #'logo_large', 'logo_medium', 'logo_small',
            'mascot':'MASCOT', 'mascot_name':'MASCOT_NAME', 'mascot_common_name':'MASCOT_LABEL',
            'tax_species':'SPECIES', 'tax_genus':'GENUS', 'tax_family':'FAMILY',
            'tax_order':'ORDER', 'tax_class':'CLASS', 'tax_phylum':'PHYLUM', 'tax_kingdom':'KINGDOM', 'tax_domain':'DOMAIN',
            'Conference':'KP_CONF', 'Rank':'KP_RANK', 'Seed':'SEED', 'Win':'WIN', 'Loss':'LOSS', 'Adj EM':'ADJ_EM',
            'AdjO':'ADJ_O', 'AdjD':'ADJ_D', 'AdjT':'ADJ_T', 'Luck':'LUCK',
            'SOS Adj EM':'SOS_ADJ_EM', 'SOS OppO':'SOS_OPP_O', 'SOS OppD':'SOS_OPP_D', 'NCSOS Adj EM':'NCSOS_ADJ_EM'
            }

#%%
# MAP COLUMN LABELING TO DATAFRAME
rollup_df.columns = rollup_df.columns.map(roll_cols)

print(rollup_df.columns)
print(rollup_df.info())


#%% [markdown]
# * Dash App Architecture

#%%

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
ncaab_app = dash.Dash('NCAAM BASKETBALL DASHBOARD', external_stylesheets=external_stylesheets) #
application = ncaab_app.server

# {current_time:%Y-%m-%d %H:%M}

ncaab_app.layout = html.Div([html.H1('NCAAM BASKETBALL DASHBOARD',
                                     style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)', # #rgb(223,187,133) #3a7c89 #42B7B9
                                            'color': 'black', 'fontWeight': 'bold', 'fontSize': '36px', #'#F1F1F1'
                                            'border': '5px solid black', 'font-family': 'Arial'}), #Garamond
                             dcc.Tabs(id='ncaa-tabs',
                                      children=[
                                          dcc.Tab(label='TEAM VIZ', value='TEAM VIZ',
                                                  style={'textAlign': 'Center', 'backgroundColor': '#42B7B9',
                                                         'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                                         'border': '3px solid black', 'font-family': 'Arial'},
                                                  selected_style={'textAlign': 'Center', 'backgroundColor': '#42B7B9',
                                                         'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                                         'border': '3px solid black', 'font-family': 'Arial'}),
                                          dcc.Tab(label='STAT VIZ', value='STAT VIZ',
                                                  style={'textAlign': 'Center', 'backgroundColor': '#F1F1F1', ##D691C1
                                                         'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                                         'border': '3px solid black', 'font-family': 'Arial'},
                                                  selected_style={'textAlign': 'Center', 'backgroundColor': '#F1F1F1',  ##D691C1
                                                         'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                                         'border': '3px solid black', 'font-family': 'Arial'}),
                                          dcc.Tab(label='CAT VIZ', value='CAT VIZ',
                                                  style={'textAlign': 'Center', 'backgroundColor': '#C75DAB', #A7D3D4,,#E4C1D9,
                                                         'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                                         'border': '3px solid black', 'font-family': 'Arial'},
                                                  selected_style={'textAlign': 'Center', 'backgroundColor': '#C75DAB', #D691C1,#C75DAB
                                                         'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                                         'border': '3px solid black', 'font-family': 'Arial'})]),
                             html.Div(id='dash-layout')])


team_viz_layout = html.Div([html.H1('TEAM DATABASE',
                                    style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
                                           'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                           'border': '2px solid black', 'font-family': 'Arial'}), #padding: 20px;
                            dash_table.DataTable(tr_df.to_dict('records'),
                                                    columns=[{"name": i, "id": i} for i in tr_df.columns],
                                                    id='tr-df',
                                                    style_data={'textAlign': 'Center', 'fontWeight': 'bold', 'border': '2px solid black'},
                                                    style_cell={'textAlign': 'Center', 'fontWeight': 'bold', 'padding': '5px'},
                                                                   # 'maxHeight': '400px', 'maxWidth': '60px'  324f6e - TBD  #B10DC9 - fuschia #7FDBFF - Aqua
                                                    style_header={'backgroundColor': '#7FDBFF', 'color': 'black',
                                                                  'fontWeight': 'bold', 'border': '2px solid black'},
                                                    sort_action='native',
                                                    #style_data_conditional = [styles],
                                                 ),
                            ])


stat_viz_layout = html.Div([html.H1('STAT VIZ [TBU]',
                                    style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
                                           'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                           'border': '4px solid black', 'font-family': 'Arial'}),
                            dcc.Graph(id='charta'),
                            html.Br(),
                            html.P('STAT A'),
                            dcc.Dropdown(id='stata',
                                         options=[#{'label': 'WIN%', 'value': 'WIN%'},
                                                  #{'label': 'AVG_MARGIN', 'value': 'AVG_MARGIN'},
                                                  {'label': 'OFFENSIVE EFFICIENCY', 'value': 'OFF_EFF'},
                                                  {'label': 'DEFENSIVE EFFICIENCY', 'value': 'DEF_EFF'},
                                                  {'label': 'EFG%', 'value': 'EFG%'},
                                                  {'label': 'OPP EFG%', 'value': 'OPP_EFG%'},
                                                  #{'label': 'TS%', 'value': 'TS%'}, {'label': 'OPP TS%', 'value': 'OPP_TS%'},
                                                  {'label': 'AST/TO', 'value': 'AST/TO'},
                                                  {'label': 'OPP AST/TO', 'value': 'OPP_AST/TO'},
                                                  #{'label': 'STL+BLK/GM', 'value': 'STL+BLK/GM'},
                                                  # #{'label': 'OPP_STL+BLK/GM', 'value': 'OPP_STL+BLK/GM'},
                                                    ], value='OFF_EFF'), #, clearable=False
                            ])

efg_win = px.scatter(rollup_df, x=rollup_df['EFG%'], y=rollup_df['WIN%'], hover_data=['TEAM'], color=rollup_df['CONF'], color_continuous_scale='Tropic') # barmode='group',  #, barmode='group' color=rollup_df['CONF']
margin_win = px.scatter(rollup_df, x=rollup_df['AVG_MARGIN'], y=rollup_df['WIN%'], hover_data=['TEAM'], color=rollup_df['CONF'], color_continuous_scale='Tropic')
o_eff_win = px.scatter(rollup_df, x=rollup_df['O_EFF'], y=rollup_df['WIN%'], hover_data=['TEAM'], color=rollup_df['CONF'], color_continuous_scale='Tropic')
d_eff_win = px.scatter(rollup_df, x=rollup_df['D_EFF'], y=rollup_df['WIN%'], hover_data=['TEAM'], color=rollup_df['CONF'], color_continuous_scale='Tropic') #plasma, thermal, spectral

subplot_1x1 = html.Div([html.H1('EFG% vs. WIN%',
                                style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
                                       'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                       'border': '2px solid black', 'font-family': 'Arial'},
                                ),
                        # html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='subplot-1-1x1', figure=efg_win),
                        ])

subplot_1x2 = html.Div([html.H1('AVG. MARGIN vs. WIN%',
                                style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
                                       'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                       'border': '2px solid black', 'font-family': 'Arial'},
                                ),
                        # html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='subplot-1-1x2', figure=margin_win),
                        ])

subplot_2x1 = html.Div([html.H1('OFFENSIVE EFFICIENCY vs. WIN%',
                                style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
                                       'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                       'border': '2px solid black', 'font-family': 'Arial'},
                                ),
                        # html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='subplot-1-2x1', figure=o_eff_win),
                        ])

subplot_2x2 = html.Div([html.H1('DEFENSIVE EFFICIENCY vs. WIN%',
                                style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
                                       'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                       'border': '2px solid black', 'font-family': 'Arial'},
                                ),
                        # html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='subplot-1-2x2', figure=d_eff_win),
                        ])


stat_viz_layout_sp = html.Div([html.Div(subplot_1x1,style={'width':'49%','display':'inline-block'}),
                      html.Div(subplot_1x2,style={'width':'49%','display':'inline-block'}),
                      html.Div(subplot_2x1,style={'width':'49%','display':'inline-block'}),
                      html.Div(subplot_2x2,style={'width':'49%','display':'inline-block'})])

cat_viz_layout = html.Div([html.H1('CAT VIZ [TBU]',
                                   style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
                                          'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
                                          'border': '4px solid black', 'font-family': 'Arial'}),
                           dcc.Graph(id='chartb'),
                           html.Br(),
                           html.P('STAT B'),
                           dcc.Dropdown(id='statb',
                               options=[{'label': 'MASCOT', 'value': 'MASCOT'},
                                        {'label': 'MASCOT CATEGORY', 'value': 'MASCOT_LABEL'},
                                        {'label': 'MASCOT SPECIES', 'value': 'SPECIES'},
                                        {'label': 'MASCOT ORDER', 'value': 'ORDER'},
                                        #{'label': 'MASCOT CLASS', 'value': 'CLASS'},
                                        ], value='MASCOT_LABEL'),
                           ])

# TAB CONFIGURATION CALLBACK
@ncaab_app.callback(Output(component_id='dash-layout', component_property='children'),
                    [Input(component_id='ncaa-tabs', component_property='value')])

def update_layout(tab):
    if tab == 'TEAM VIZ':
        return team_viz_layout
    elif tab == 'STAT VIZ':
        return stat_viz_layout_sp
    elif tab == 'CAT VIZ':
        return cat_viz_layout


# TEAM VIZ CALLBACK
@ncaab_app.callback(Output(component_id='tr-df', component_property='figure'),
                    [Input(component_id='statb', component_property='value'),])

def display_dataframe(df):
    return df


# CAT VIZ CALLBACK
@ncaab_app.callback(Output(component_id='chartb', component_property='figure'),
                     [Input(component_id='statb', component_property='value')])

def display_chart(statb):
    fig = px.histogram(data_frame=rollup_df, x=rollup_df[statb], title=f'{statb} HISTOGRAM',
                       marginal="box", nbins=30, opacity=0.75,
                       color_discrete_sequence=['#009B9E', '#C75DAB'],  #'#FFBD59', '#3BA27A'

                       ) # hover data - school venue? team performance?
    return fig

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8035)

#%%