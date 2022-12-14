## LIBRARY IMPORTS ##
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np

import plotly as ply
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from PIL import Image


import datetime

# import yfinance as yf

# import dash as dash
# from dash import dash_table
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# from dash.exceptions import PreventUpdate
# import dash_bootstrap_components as dbc

# import scipy.stats as stats
# import statistics
# import time
# import pickle


## VISUAL SETTINGS ##

# pd.options.display.float_format = '${:,.2f}'.format
# pd.set_option('display.max_colwidth', 200)


## DIRECTORY CONFIGURATION ##
## DATA IMPORTS ##

engine_csv = r'https://raw.githubusercontent.com/nehat312/prop-swap/main/data/engine.csv'
all_investor_idx = pd.read_csv(engine_csv) #, header=0, index_col=0 #, encoding='utf-8'

## TIME INTERVALS ##

today = datetime.date.today()
before = today - datetime.timedelta(days=1095) #700
start_date = '2000-01-01'
end_date = today #'2022-06-30'  #'2022-03-31'
mrq = '2022-06-30'
mry = '2021-12-31'

## VARIABLE ASSIGNMENT

all_investor_idx = all_investor_idx.sort_values(by='TTL_VOL_RANK')

investor_cols = ['INVESTOR', 'INVESTOR_TYPE', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', 'C-SUITE']
mf_cols = ['INVESTOR', 'INVESTOR_TYPE', 'MF_AVG_PRICE_MM', 'MF_UNITS_PROP', 'MF_AVG_PPU',  'MF_QUALITY', 'MF_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE'] # AVG_QUALITY, 'TTL_VOL_RANK', 'TTL_SF_RANK',
sc_cols = ['INVESTOR', 'INVESTOR_TYPE', 'SC_AVG_PRICE_MM', 'SC_SF_PROP', 'SC_AVG_PSF',  'SC_QUALITY', 'SC_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE']
nnn_cols = ['INVESTOR', 'INVESTOR_TYPE', 'NNN_AVG_PRICE_MM', 'NNN_SF_PROP', 'NNN_AVG_PSF',  'NNN_QUALITY', 'NNN_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE']
mall_cols = ['INVESTOR', 'INVESTOR_TYPE', 'MALL_AVG_PRICE_MM', 'MALL_SF_PROP', 'MALL_AVG_PSF',  'MALL_QUALITY', 'MALL_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE']
ss_cols = ['INVESTOR', 'INVESTOR_TYPE', 'SS_AVG_PRICE_MM', 'SS_SF_PROP',  'SS_AVG_PSF',  'SS_QUALITY', 'SS_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE']
ind_cols = ['INVESTOR', 'INVESTOR_TYPE', 'IND_AVG_PRICE_MM', 'IND_SF_PROP', 'IND_AVG_PSF',  'IND_QUALITY', 'IND_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', ]
fs_cols = ['INVESTOR', 'INVESTOR_TYPE', 'FS_AVG_PRICE_MM', 'FS_KEYS_PROP', 'FS_AVG_PPK',  'FS_QUALITY', 'FS_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', ]
ls_cols = ['INVESTOR', 'INVESTOR_TYPE', 'LS_AVG_PRICE_MM', 'LS_KEYS_PROP', 'LS_AVG_PPK',  'LS_QUALITY', 'LS_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', ]
cbd_cols = ['INVESTOR', 'INVESTOR_TYPE', 'CBD_AVG_PRICE_MM', 'CBD_SF_PROP', 'CBD_AVG_PSF',  'CBD_QUALITY', 'CBD_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', ]
sub_cols = ['INVESTOR', 'INVESTOR_TYPE', 'SUB_AVG_PRICE_MM', 'SUB_SF_PROP', 'SUB_AVG_PSF',  'SUB_QUALITY', 'SUB_VOL_RANK', 'CITY', 'STATE', 'COUNTRY', 'MSA', 'WEBSITE', ]


#%%
# all_reits_trading = yf.download(tickers = reit_tickers,
#         period = "max", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#         interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#         start = start_date, #'2000-01-01'
#         end = today,
#         group_by = 'column',
#         auto_adjust = True,
#         prepost = False,
#         threads = True,
#         proxy = None, #"PROXY_SERVER"
#         timeout=12)

#%%
##################
# FORMAT / STYLE #
##################

## COLOR SCALES ##
Sunset = px.colors.sequential.Sunset
Sunsetdark = px.colors.sequential.Sunsetdark
Tropic = px.colors.diverging.Tropic
Temps = px.colors.diverging.Temps
Tealrose = px.colors.diverging.Tealrose
Blackbody = px.colors.sequential.Blackbody
Ice = px.colors.sequential.ice
Ice_r = px.colors.sequential.ice_r
Dense = px.colors.sequential.dense
# YlOrRd = px.colors.sequential.YlOrRd
# Mint = px.colors.sequential.Mint
# Electric = px.colors.sequential.Electric

## SECTOR COLORS ##

sector_colors = {'apartment':'#FFDF00',
                 'office':'#29609C',
                 'hotel':'#E9EDED',
                 'mall':'#D5FF0A',
                 'strip_center':'#46D8BF',
                 'net_lease':'#EEFCF7',
                 'industrial':'#535865',
                 'self_storage':'#5F8C95',
                 'data_center':'#3AA5C3',
                 'healthcare':'#FF3363',

                 'apartment_avg':'#FFDF00',
                 'office_avg':'#29609C',
                 'hotel_avg':'#E9EDED',
                 'mall_avg':'#D5FF0A',
                 'strip_center_avg':'#46D8BF',
                 'net_lease_avg':'#EEFCF7',
                 'industrial_avg':'#535865',
                 'self_storage_avg':'#5F8C95',
                 'data_center_avg':'#3AA5C3',
                 'healthcare_avg':'#FF3363',
                 }

chart_labels = {'INVESTOR_TYPE': 'INVESTOR TYPE',
                'MF_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'MF_AVG_PPU': 'AVG. PPU ($)',
                'SC_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'SC_AVG_PSF': 'AVG. PSF ($)',
                'NNN_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'NNN_AVG_PSF': 'AVG. PSF ($)',
                'MALL_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'MALL_AVG_PSF': 'AVG. PSF ($)',
                'IND_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'IND_AVG_PSF': 'AVG. PSF ($)',
                'SS_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'SS_AVG_PSF': 'AVG. PSF ($)',
                'CBD_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'CBD_AVG_PSF': 'AVG. PSF ($)',
                'SUB_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'SUB_AVG_PSF': 'AVG. PSF ($)',
                'FS_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'FS_AVG_PSF': 'AVG. PSF ($)',
                'LS_AVG_PRICE_MM': 'AVG. PRICE ($MM)',
                'LS_AVG_PSF': 'AVG. PSF ($)',

                # 'apartment':'APARTMENT',
                # 'office':'OFFICE',
                # 'hotel':'LODGING',
                # 'strip_center':'STRIP_CENTER',
                # 'net_lease':'NET LEASE',
                # 'mall':'MALL',
                # 'industrial':'INDUSTRIAL',
                # 'self_storage':'SELF-STORAGE',
                # 'data_center':'DATA CENTER',
                # 'healthcare':'HEALTHCARE',

#                 'reportPeriod':'REPORT PERIOD',
#                 'ticker':'TICKER',
#                 'company':'COMPANY',
#                 'city':'CITY',
#                 'state':'STATE',

                }

# #%%

#####################
### STREAMLIT APP ###
#####################

## CONFIGURATION ##
st.set_page_config(page_title='PROP/SWAP',
                   layout='wide',
                   initial_sidebar_state='auto') #, page_icon=":emoji:"

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)


## CSS CUSTOMIZATION ##
th_props = [('font-size', '12px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#EBEDE9'), #6d6d6d #29609C
            ('background-color', '#29609C') #f7f7f9
            ]

td_props = [('font-size', '12px'),
            # ('text-align', 'center'),
            # ('font-weight', 'bold'),
            ]

df_styles = [dict(selector="th", props=th_props),
             dict(selector="td", props=td_props)]


# col_format_dict = {'profitMargin': "{:.1%}", 'payoutRatio': "{:.1%}", 'dividendYield': "{:.1%}",
#                    'dividendsPerBasicCommonShare': "${:.2}", #'Price_Actual': "${:.2}",
#                    'priceToEarningsRatio': "{:.1}x", 'priceToBookValue': "{:.1}x",
#                    'enterpriseValueOverEBIT': "{:.1}x", 'enterpriseValueOverEBITDA': "{:.1}x",
#                    'shares': "{:,}",
#                    'marketCapitalization': "${:,}",
#                    'earningBeforeInterestTaxes': "${:,}",
#                    'earningsBeforeInterestTaxesDepreciationAmortization': "${:,}",
#                    'assets': "${:,}", 'debt': "${:,}", 'totalLiabilities': "${:,}",
#                    'cashAndEquivalents': "${:,}",
#                    'netIncome': "${:,}", 'netIncomeToNonControllingInterests': "${:,}",
#                    'enterpriseValue': "${:,}", 'netCashFlow': "${:,}",
#                    'capitalExpenditure': "${:,}", 'netCashFlowBusinessAcquisitionsDisposals': "${:,}",
#                    'investedCapital': "${:,}", 'investments': "${:,}",
#                    'propertyPlantEquipmentNet': "${:,}", 'tangibleAssetValue': "${:,}",
#                    }


## SIDEBAR ##

# sidebar_header = st.sidebar.subheader('VISUALIZATION TIMEFRAME:')
# sidebar_start = st.sidebar.date_input('START DATE', before)
# sidebar_end = st.sidebar.date_input('END DATE', today)
# if sidebar_start < sidebar_end:
#     st.sidebar.success('START DATE: `%s`\n\nEND DATE: `%s`' % (sidebar_start, sidebar_end))
# else:
#     st.sidebar.error('ERROR: END DATE BEFORE START DATE')

# sector_sidebar_select = st.sidebar.selectbox('SECTOR', (sector_list_of_names), help='SELECT CRE SECTOR')
# ticker_sidebar_select = st.sidebar.selectbox('TICKER', (sector_dict['apartment'])) #sector_sidebar_select

## EXTERNAL LINKS ##
github_link = '[GITHUB REPOSITORY](https://github.com/nehat312/prop-swap/)'
reit_comps_link = '[REIT TRADING COMPS](<TBU>)'
tbu_link = '[TBU](<TBU>)'

link_col_1, link_col_2, link_col_3 = st.columns(3)
ext_link_1 = link_col_1.markdown(github_link, unsafe_allow_html=True)
ext_link_2 = link_col_2.markdown(reit_comps_link, unsafe_allow_html=True)
ext_link_3 = link_col_3.markdown(tbu_link, unsafe_allow_html=True)

## INTRODUCTION ##
st.container()
st.title('PROP/SWAP')
st.subheader('*VIRTUAL CRE BROKER*')

st.title('PROP/SWAP')

## SECTOR TABS ##
tab_0, tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7, tab_8, tab_9, tab_10 = st.tabs(['ALL SECTORS', 'MULTIFAMILY', 'STRIP CENTER', 'NNN', 'MALL', 'INDUSTRIAL', 'SELF-STORAGE', 'CBD OFFICE', 'SUB OFFICE', 'FS HOTEL', 'LS HOTEL']) #'DATA CENTER', 'HEALTHCARE'

# prop_params_header = st.header('INPUT PROPERTY PARAMETERS:')
#
# sector = st.selectbox('PROPERTY TYPE:',
#                       ("MULTIFAMILY",
#                        "STRIP CENTER", "NNN RETAIL", "MALL",
#                        "SELF-STORAGE", "INDUSTRIAL",
#                        "FULL-SERVICE HOTEL", "LIMITED-SERVICE HOTEL",
#                        "CBD OFFICE", "SUBURBAN OFFICE")
#                       )

@st.cache(persist=True, allow_output_mutation=True)
def filter_buyers(sector, prop_size, min_prop_price, prop_qual):
    if sector == 'MULTIFAMILY':
        for investors in all_investor_idx:
            mf_size_filter = all_investor_idx[all_investor_idx.MF_UNITS_PROP >= prop_size]
            mf_min_price_filter = mf_size_filter[mf_size_filter.MF_AVG_PRICE_MM >= min_prop_price]
            mf_qual_filter = mf_min_price_filter[(mf_min_price_filter.MF_QUALITY >= (prop_qual - 1)) & (
                        mf_min_price_filter.MF_QUALITY <= (prop_qual + 1))]
            mf_buyer_recs = mf_qual_filter.sort_values(by='MF_VOL_RANK', ascending=True)[:50]
            mf_buyer_recs = pd.DataFrame(data=mf_buyer_recs, columns=mf_cols)
        return mf_buyer_recs
    elif sector == 'STRIP CENTER':
        for investors in all_investor_idx:
            sc_size_filter = all_investor_idx[all_investor_idx['SC_SF_PROP'] >= prop_size]
            sc_min_price_filter = sc_size_filter[sc_size_filter['SC_AVG_PRICE_MM'] >= min_prop_price]
            sc_qual_filter = sc_min_price_filter[(sc_min_price_filter['SC_QUALITY'] >= (prop_qual - 1)) & (
                        sc_min_price_filter['SC_QUALITY'] <= (prop_qual + 1))]
            sc_buyer_recs = sc_qual_filter.sort_values(by='SC_VOL_RANK', ascending=True)[:50]
            sc_buyer_recs = pd.DataFrame(data=sc_buyer_recs, columns=sc_cols)
        return sc_buyer_recs
    elif sector == 'NNN RETAIL':
        for investors in all_investor_idx:
            nnn_size_filter = all_investor_idx[all_investor_idx['NNN_SF_PROP'] >= prop_size]
            nnn_min_price_filter = nnn_size_filter[nnn_size_filter['NNN_AVG_PRICE_MM'] >= min_prop_price]
            nnn_qual_filter = nnn_min_price_filter[(nnn_min_price_filter['NNN_QUALITY'] >= (prop_qual - 1)) & (
                        nnn_min_price_filter['NNN_QUALITY'] <= (prop_qual + 1))]
            nnn_buyer_recs = nnn_qual_filter.sort_values(by='NNN_VOL_RANK', ascending=True)[:50]
            nnn_buyer_recs = pd.DataFrame(data=nnn_buyer_recs, columns=nnn_cols)
        return nnn_buyer_recs
    elif sector == 'MALL':
        for investors in all_investor_idx:
            mall_size_filter = all_investor_idx[all_investor_idx['MALL_SF_PROP'] >= prop_size]
            mall_min_price_filter = mall_size_filter[mall_size_filter['MALL_AVG_PRICE_MM'] >= min_prop_price]
            mall_qual_filter = mall_min_price_filter[(mall_min_price_filter['MALL_QUALITY'] >= (prop_qual - 2)) & (
                        mall_min_price_filter['MALL_QUALITY'] <= (prop_qual + 2))]
            mall_buyer_recs = mall_qual_filter.sort_values(by='MALL_VOL_RANK', ascending=False)[:10]
            mall_buyer_recs = pd.DataFrame(data=mall_buyer_recs, columns=mall_cols)
        return mall_buyer_recs
    elif sector == 'SELF-STORAGE':
        for investors in all_investor_idx:
            ss_size_filter = all_investor_idx[all_investor_idx['SS_SF_PROP'] >= prop_size]
            ss_min_price_filter = ss_size_filter[ss_size_filter['SS_AVG_PRICE_MM'] >= min_prop_price]
            ss_qual_filter = ss_min_price_filter[(ss_min_price_filter['SS_QUALITY'] >= (prop_qual - 1)) & (
                        ss_min_price_filter['SS_QUALITY'] <= (prop_qual + 1))]
            ss_buyer_recs = ss_qual_filter.sort_values(by='SS_VOL_RANK', ascending=True)[:50]
            ss_buyer_recs = pd.DataFrame(data=ss_buyer_recs, columns=ss_cols)
        return ss_buyer_recs
    elif sector == 'INDUSTRIAL':
        for investors in all_investor_idx:
            ind_size_filter = all_investor_idx[all_investor_idx['IND_SF_PROP'] >= prop_size]
            ind_min_price_filter = ind_size_filter[ind_size_filter['IND_AVG_PRICE_MM'] >= min_prop_price]
            ind_qual_filter = ind_min_price_filter[(ind_min_price_filter['IND_QUALITY'] >= (prop_qual - 1)) & (
                        ind_min_price_filter['IND_QUALITY'] <= (prop_qual + 1))]
            ind_buyer_recs = ind_qual_filter.sort_values(by='IND_VOL_RANK', ascending=True)[:50]
            ind_buyer_recs = pd.DataFrame(data=ind_buyer_recs, columns=ind_cols)
        return ind_buyer_recs
    elif sector == 'FULL-SERVICE HOTEL':
        for investors in all_investor_idx:
            fs_size_filter = all_investor_idx[all_investor_idx['FS_KEYS_PROP'] >= prop_size]
            fs_min_price_filter = fs_size_filter[fs_size_filter['FS_AVG_PRICE_MM'] >= min_prop_price]
            fs_qual_filter = fs_min_price_filter[(fs_min_price_filter['FS_QUALITY'] >= (prop_qual - 1)) & (
                        fs_min_price_filter['FS_QUALITY'] <= (prop_qual + 1))]
            fs_buyer_recs = fs_qual_filter.sort_values(by='FS_VOL_RANK', ascending=True)[:50]
            fs_buyer_recs = pd.DataFrame(data=fs_buyer_recs, columns=fs_cols)
        return fs_buyer_recs
    elif sector == 'LIMITED-SERVICE HOTEL':
        for investors in all_investor_idx:
            ls_size_filter = all_investor_idx[all_investor_idx['LS_KEYS_PROP'] >= prop_size]
            ls_min_price_filter = ls_size_filter[ls_size_filter['LS_AVG_PRICE_MM'] >= min_prop_price]
            ls_qual_filter = ls_min_price_filter[(ls_min_price_filter['LS_QUALITY'] >= (prop_qual - 1)) & (
                        ls_min_price_filter['LS_QUALITY'] <= (prop_qual + 1))]
            ls_buyer_recs = ls_qual_filter.sort_values(by='LS_VOL_RANK', ascending=True)[:50]
            ls_buyer_recs = pd.DataFrame(data=ls_buyer_recs, columns=ls_cols)
        return ls_buyer_recs
    elif sector == 'CBD OFFICE':
        for investors in all_investor_idx:
            cbd_size_filter = all_investor_idx[all_investor_idx['CBD_SF_PROP'] >= prop_size]
            cbd_min_price_filter = cbd_size_filter[cbd_size_filter['CBD_AVG_PRICE_MM'] >= min_prop_price]
            cbd_qual_filter = cbd_min_price_filter[(cbd_min_price_filter['CBD_QUALITY'] >= (prop_qual - 1)) & (
                        cbd_min_price_filter['CBD_QUALITY'] <= (prop_qual + 1))]
            cbd_buyer_recs = cbd_qual_filter.sort_values(by='CBD_VOL_RANK', ascending=True)[:50]
            cbd_buyer_recs = pd.DataFrame(data=cbd_buyer_recs, columns=cbd_cols)
        return cbd_buyer_recs
    elif sector == 'SUB OFFICE':
        for investors in all_investor_idx:
            sub_size_filter = all_investor_idx[all_investor_idx['SUB_SF_PROP'] >= prop_size]
            sub_min_price_filter = sub_size_filter[sub_size_filter['SUB_AVG_PRICE_MM'] >= min_prop_price]
            sub_qual_filter = sub_min_price_filter[(sub_min_price_filter['SUB_QUALITY'] >= (prop_qual - 1)) & (
                        sub_min_price_filter['SUB_QUALITY'] <= (prop_qual + 1))]
            sub_buyer_recs = sub_qual_filter.sort_values(by='SUB_VOL_RANK', ascending=True)[:50]
            sub_buyer_recs = pd.DataFrame(data=sub_buyer_recs, columns=sub_cols)
        return sub_buyer_recs

with tab_0:
    all_sectors_header = st.subheader('COMMERCIAL REAL ESTATE INVESTOR UNIVERSE')
    # ECONOMIC INDICATORS

with tab_1:
    with st.form('MULTIFAMILY PROPERTY PARAMETERS'):
        multifamily_header = st.subheader('MULTIFAMILY')
        mf_prop_size = st.number_input('*TOTAL MF UNITS [25-1,000 UNITS]', min_value=25, max_value=500, step=25, value=100)  # list(range(25,750,25)))
        mf_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='MF', min_value=0, max_value=100, value=10, step=5)
        # mf_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))
        mf_params_submit = st.form_submit_button("PROP/SWAP")

with tab_2:
    strip_center_header = st.subheader('STRIP CENTER')
    sc_prop_size = st.number_input('*TOTAL SC SF [5K-500K SF]', min_value=5000, max_value=500000, step=5000,
                                value=50000)  # list(range(5000,505000,5000)))
    sc_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='SC', min_value=0, max_value=100, value=10, step=5)
    # sc_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_3:
    nnn_header = st.subheader('NNN')
    nnn_prop_size = st.number_input('*TOTAL NNN SF [5K-500K SF]', min_value=5000, max_value=500000, step=5000, value=50000)  # list(range(5000,505000,5000)))
    nnn_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='NNN', min_value=0, max_value=100, value=10, step=5)
    # nnn_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_4:
    mall_header = st.subheader('MALL')
    mall_prop_size = st.number_input('*TOTAL MALL SF [50K-1MM SF]', min_value=50000, max_value=500000, step=5000, value=250000)  # list(range(50000,1010000,10000)))
    mall_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='MALL', min_value=0, max_value=100, value=10, step=5)
    # mall_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_5:
    industrial_header = st.subheader('INDUSTRIAL')
    ind_prop_size = st.number_input('*TOTAL INDUSTRIAL SF [5K-1MM SF]', min_value=10000, max_value=500000, step=5000, value=100000)  # list(range(5000,1005000,5000)))
    ind_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='IND', min_value=0, max_value=100, value=10, step=5)
    # ind_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_6:
    self_storage_header = st.subheader('SELF-STORAGE')
    ss_prop_size = st.number_input('*TOTAL SELF-STORAGE SF [5K-500K SF]', min_value=0, max_value=500000, step=2500, value=100)  # list(range(0,525000,25000)))
    ss_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='SS', min_value=0, max_value=100, value=10, step=5)
    # ss_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_7:
    cbd_office_header = st.subheader('CBD OFFICE')
    cbd_prop_size = st.number_input('*TOTAL CBD OFFICE SF [10K-500K SF]', min_value=10000, max_value=500000, step=5000,
                                value=100000)  # list(range(10000,505000,5000)))
    cbd_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='CBD', min_value=0, max_value=100, value=10, step=5)
    # cbd_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_8:
    sub_office_header = st.subheader('SUBURBAN OFFICE')
    sub_prop_size = st.number_input('*TOTAL SUB OFFICE SF: [10K-500K SF]', min_value=10000, max_value=500000, step=5000, value=100000)  # list(range(10000,505000,5000)))
    sub_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='SUB', min_value=0, max_value=100, value=10, step=5)
    # sub_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_9:
    fs_hotel_header = st.subheader('FULL-SERVICE HOTEL')
    fs_prop_size = st.number_input('*TOTAL FS KEYS [25-1,000 KEYS]', min_value=25, max_value=750, step=25, value=100)  # list(range(25,750,25)))
    fs_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='FS', min_value=0, max_value=100, value=10,                                     step=5)
    # fs_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

with tab_10:
    Ls_hotel_header = st.subheader('LIMITED-SERVICE HOTEL')
    Ls_prop_size = st.number_input('*TOTAL LS KEYS [25-1,000 KEYS]', min_value=25, max_value=750, step=25, value=100)  # list(range(25,750,25)))
    Ls_min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', key='LS', min_value=0, max_value=100, value=10, step=5)
    # Ls_prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))


# with st.form("PROPERTY PARAMETERS"):
#     if sector == "MULTIFAMILY":
#         prop_size = st.number_input('*TOTAL MF UNITS [25-1,000 UNITS]', min_value=25, max_value=500, step=25, value=100) #list(range(25,750,25)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))
#     if sector == "STRIP CENTER":
#         prop_size = st.number_input('*TOTAL SC SF [5K-500K SF]', min_value=5000, max_value=500000, step=5000, value=50000) #list(range(5000,505000,5000)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))
#     if sector == "NNN RETAIL":
#         prop_size = st.number_input('*TOTAL NNN SF [5K-500K SF]', min_value=5000, max_value=500000, step=5000, value=50000) #list(range(5000,505000,5000)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))
#     if sector == "MALL":
#         prop_size = st.number_input('*TOTAL MALL SF [50K-1MM SF]', min_value=50000, max_value=500000, step=5000, value=250000) #list(range(50000,1010000,10000)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))
#     if sector == "SELF-STORAGE":
#         prop_size = st.number_input('*TOTAL SELF-STORAGE SF [5K-500K SF]', min_value=0, max_value=500000, step=2500, value=100) #list(range(0,525000,25000)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1,6,1)))
#     if sector == "INDUSTRIAL":
#         prop_size = st.number_input('*TOTAL INDUSTRIAL SF [5K-1MM SF]', min_value=10000, max_value=500000, step=5000, value=100000) #list(range(5000,1005000,5000)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1,6,1)))
#     if sector == "CBD OFFICE":
#         prop_size = st.number_input('*TOTAL CBD OFFICE SF [10K-500K SF]', min_value=10000, max_value=500000, step=5000, value=100000) #list(range(10000,505000,5000)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1,6,1)))
#     if sector == "SUBURBAN OFFICE":
#         prop_size = st.number_input('*TOTAL SUB OFFICE SF: [10K-500K SF]', min_value=10000, max_value=500000, step=5000, value=100000) # list(range(10000,505000,5000)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10, step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1,6,1)))
#
#     if sector == "FULL-SERVICE HOTEL":
#         prop_size = st.number_input('*TOTAL FS KEYS [25-1,000 KEYS]', min_value=25, max_value=750, step=25,
#                                         value=100)  # list(range(25,750,25)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10,
#                                              step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))
#     if sector == "LIMITED-SERVICE HOTEL":
#         prop_size = st.number_input('*TOTAL LS KEYS [25-1,000 KEYS]', min_value=25, max_value=750, step=25,
#                                         value=100)  # list(range(25,750,25)))
#         min_prop_price = st.number_input('*MINIMUM VALUATION [$0MM-$100MM]', min_value=0, max_value=100, value=10,
#                                              step=5)
#         prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:', list(range(1, 6, 1)))

        # prop_cap_rate = st.selectbox('*EST. CAP RATE:', list(range(1, 6, 1)))
        # property_region = st.selectbox('*PROPERTY REGION:', ("NORTHEAST", "MID-ATLANTIC", "SOUTHEAST", "WEST", "NORTHWEST", "MIDWEST", "SOUTHWEST"))

        # elif min_prop_price > 0:
        #     implied_ppu_title = st.write('*IMPLIED VALUE / UNIT:')
        #     implied_ppu = st.markdown(round(min_prop_price * 1_000_000 / prop_size))


    # params_submit = st.form_submit_button("PROP/SWAP")




### *****
### *****

## TARGET INVESTOR DATAFRAME ##

### *****
### *****
    # if mf_params_submit:
    #     buyer_rec_df = filter_buyers(sector, prop_size, min_prop_price, prop_qual)




        # st.write("TARGETED INVESTOR POOL:")
        # st.dataframe(buyer_rec_df)
            # buyer_rec_df = buyer_rec_df.set_index('INVESTOR')

        ## DATAFRAME STYLING ##

        # def df_style_map(val):
        #     if val == 'United States':
        #         color = 'black'
        #     else:
        #         color = 'pink'
        #         return f'background-color: {color}'
        #
        # st.dataframe(buyer_rec_df.style.applymap(df_style_map, subset=['COUNTRY']))




## VALUATION METRICS ##
        # if sector == 'MULTIFAMILY':
        #     per_unit_valuation = round(buyer_rec_df['MF_AVG_PPU'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION: ${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}/UNIT')
        #     # st.metric('ESTIMATED PROPERTY VALUATION: $', (prop_valuation / 1_000_000))
        #     # st.metric('ESTIMATED PROPERTY VALUATION: $/UNIT', per_unit_valuation)
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        #     mf_chart_1 = px.scatter(buyer_rec_df, #all_investor_idx
        #                             x=buyer_rec_df['MF_AVG_PRICE_MM'],
        #                             y=buyer_rec_df['MF_AVG_PPU'],
        #                             # color=buyer_rec_df['INVESTOR_TYPE'],
        #                             color=buyer_rec_df['MF_AVG_PPU'],
        #                             color_continuous_scale=Sunsetdark, #'YlOrRd', #'Tropic',
        #                             color_discrete_sequence=Sunsetdark,
        #                             hover_name=buyer_rec_df['INVESTOR'],
        #                             hover_data=buyer_rec_df[['MSA']],
        #                             title='TARGETED INVESTOR POOL',
        #                             labels=chart_labels,
        #                             )
        #
        #     st.plotly_chart(mf_chart_1, use_container_width=False, sharing="streamlit")
        #
        #
        #     mf_chart_2 = px.bar(buyer_rec_df,
        #                         x=buyer_rec_df['INVESTOR'],
        #                         y=buyer_rec_df['MF_AVG_PPU'],
        #                         color=buyer_rec_df['INVESTOR_TYPE'],
        #                         color_continuous_scale=Tropic,
        #                         color_discrete_sequence=Tropic,
        #                         # category_orders={'total descending'},
        #                         hover_name=buyer_rec_df['INVESTOR'],
        #                         hover_data=buyer_rec_df[['MSA']],
        #                         labels=chart_labels,
        #                         barmode='relative',
        #                         # size=buyer_rec_df['SC_VOL_RANK'],
        #                         height=500,
        #                         # width=400,
        #                         )
        #
        #     st.plotly_chart(mf_chart_2, use_container_width=False, sharing="streamlit")
        #
        # elif sector == 'STRIP CENTER':
        #     per_unit_valuation = round(buyer_rec_df['SC_AVG_PSF'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/SF')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        #     sc_chart_1 = px.scatter(buyer_rec_df,  # all_investor_idx
        #                             x=buyer_rec_df['SC_AVG_PRICE_MM'],
        #                             y=buyer_rec_df['SC_AVG_PSF'],
        #                             color=buyer_rec_df['INVESTOR_TYPE'],
        #                             color_continuous_scale='Tropic', #YlOrRd
        #                             hover_name=buyer_rec_df['INVESTOR'],
        #                             hover_data=buyer_rec_df[['MSA']],
        #                             title='TARGETED INVESTOR POOL',
        #                             labels={'SC_AVG_PRICE_MM': 'AVG. PRICE ($MM)', 'SC_AVG_PSF': 'AVG. PSF ($)', 'INVESTOR_TYPE': 'INVESTOR TYPE'},
        #                             )
        #
        #     st.plotly_chart(sc_chart_1, use_container_width=False, sharing="streamlit")
        #
        #     sc_chart_2 = px.bar(buyer_rec_df,
        #                         x=buyer_rec_df['INVESTOR'],
        #                         y=buyer_rec_df['SC_AVG_PSF'],
        #                         color=buyer_rec_df['INVESTOR_TYPE'],
        #                         color_continuous_scale='Tropic',
        #                         hover_name=buyer_rec_df['INVESTOR'],
        #                         hover_data=buyer_rec_df[['MSA']],
        #                         title='EST. VALUATION RANGE ($ PSF)',
        #                         labels={'SC_AVG_PRICE_MM': 'AVG. PRICE ($MM)', 'SC_AVG_PSF': 'AVG. PSF ($)', 'INVESTOR_TYPE': 'INVESTOR TYPE'},
        #                         barmode='relative',
        #                         )
        #
        #     st.plotly_chart(sc_chart_2, use_container_width=False, sharing="streamlit")
        #
        #
        #     #pattern_shape = "nation", pattern_shape_sequence = [".", "x", "+"]
        #
        #     # fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group", facet_row="time", facet_col="day",
        #     #        category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})
        #
        #     # fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")
        #
        #     # fig = px.parallel_categories(df, color="size", color_continuous_scale=px.colors.sequential.Inferno)
        #
        #     # fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
        #     #                   "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
        #     #                   "petal_width": "Petal Width", "petal_length": "Petal Length", },
        #     #                     color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
        #
        #
        # elif sector == 'NNN RETAIL':
        #     per_unit_valuation = round(buyer_rec_df['NNN_AVG_PSF'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/SF')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        # elif sector == 'MALL':
        #     per_unit_valuation = round(buyer_rec_df['MALL_AVG_PSF'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/SF')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        # elif sector == 'SELF-STORAGE':
        #     per_unit_valuation = round(buyer_rec_df['SS_AVG_PSF'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/SF')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        # elif sector == 'INDUSTRIAL':
        #     per_unit_valuation = round(buyer_rec_df['IND_AVG_PSF'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/SF')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        # elif sector == 'FULL-SERVICE HOTEL':
        #     per_unit_valuation = round(buyer_rec_df['FS_AVG_PPK'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/KEY')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        # elif sector == 'LIMITED-SERVICE HOTEL':
        #     per_unit_valuation = round(buyer_rec_df['LS_AVG_PPK'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/KEY')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        # elif sector == 'CBD OFFICE':
        #     per_unit_valuation = round(buyer_rec_df['CBD_AVG_PSF'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/SF')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)
        #
        # elif sector == 'SUB OFFICE':
        #     per_unit_valuation = round(buyer_rec_df['SUB_AVG_PSF'].mean())
        #     prop_valuation = per_unit_valuation * prop_size
        #     st.write(f'ESTIMATED PROPERTY VALUATION:')
        #     st.write(f'${(prop_valuation / 1_000_000):.2f}MM or {per_unit_valuation:.0f}$/SF')
        #     st.write("TARGETED INVESTOR POOL:")
        #     st.dataframe(buyer_rec_df)


#######################
## TABLEAU EMBEDDING ##
#######################

def main():
    html_temp = """<div class='tableauPlaceholder' id='viz1659419844202' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;MS&#47;MS874Y84Y&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;MS874Y84Y' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;MS&#47;MS874Y84Y&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1659419844202');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=600) #width=400,

if __name__ == "__main__":
    main()

# ## IMAGE @@
# test_img = Image.open('ROTATE.jpg')
# st.image(test_img)

## EXTERNAL LINKS ##

left_column, right_column = st.columns(2)
left_button = left_column.button('GITHUB REPOSITORY')
right_button = right_column.button('CONTACT INFORMATION')
if left_button:
    left_column.write('https://github.com/nehat312/prop-swap')
if right_button:
    right_column.write('')
    # left_column.write('https://public.tableau.com/shared/S4GKR7QYB?:display_count=n&:origin=viz_share_link')




# st.success('THANKS FOR PROP/SWAPPING')
# st.warning('NO BUYERS FOUND')
# st.write('*~PROP/SWAP BETA MODE~*')

st.stop()



### SCRATCH NOTES

### TRANSLATE QUALITY SCALE TO CAP RATE ###


## NUMERICAL CONVERSION ##

# mf_num_cols = ['MF_AVG_PRICE_MM', 'MF_UNITS_PROP', 'MF_AVG_PPU',  'AVG_QUALITY', 'MF_QUALITY', 'TTL_VOL_RANK', 'TTL_SF_RANK', 'MF_VOL_RANK',]

# for i in mf_num_cols:
#     pd.to_numeric(all_investor_idx[i])


# CONFIG TEMPLATE
    # st.set_page_config(page_title="CSS hacks", page_icon=":smirk:")
    #
    # c1 = st.container()
    # st.markdown("---")
    # c2 = st.container()
    # with c1:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="1")
    # with c2:
    #     st.markdown("Hello")
    #     st.slider("World", 0, 10, key="2")

# STYLE WITH CSS THROUGH MARKDOWN
    # st.markdown("""
    # <style>
    # div[data-testid="stBlock"] {
    #     padding: 1em 0;
    #     border: thick double #32a1ce;
    # }
    # </style>
    # """, unsafe_allow_html=True)


# STYLE WITH JS THROUGH HTML IFRAME
    # components.html("""
    # <script>
    # const elements = window.parent.document.querySelectorAll('div[data-testid="stBlock"]')
    # console.log(elements)
    # elements[0].style.backgroundColor = 'paleturquoise'
    # elements[1].style.backgroundColor = 'lightgreen'
    # </script>
    # """, height=0, width=0)


# st.markdown("""
#             <style>
#             div[data-testid="stBlock"] {padding: 1em 0; border: thick double #32a1ce; color: blue}
#             </style>
#             """,
#             unsafe_allow_html=True)

# style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)',
#                                            'color': 'black', 'fontWeight': 'bold', 'fontSize': '24px',
#                                            'border': '4px solid black', 'font-family': 'Arial'}),



# <div class='tableauPlaceholder' id='viz1659298369110' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;S4&#47;S4GKR7QYB&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;S4GKR7QYB' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;S4&#47;S4GKR7QYB&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1659298369110');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
# vizElement.parentNode.insertBefore(scriptElement, vizElement);
# </script>

#st.spinner()
#with st.spinner(text='CONNECTING'):
#    time.sleep(5)
#    st.success('LIVE')

#streamlit. slider ( label , min_value=None , max_value=None , value=None , step=None , format=None , key=None )