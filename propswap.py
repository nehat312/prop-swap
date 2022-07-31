## LIBRARY IMPORTS ##
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly as ply
import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

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

# pd.set_option('display.max_colwidth', 200)


## DATA IMPORTS ##

engine_csv = r'https://raw.githubusercontent.com/nehat312/command-center/main/prop-swap/data/engine.csv'
# engine_xlsx = r'https://raw.githubusercontent.com/nehat312/command-center/main/prop-swap/data/engine.csv'

all_investor_idx = pd.read_csv(engine_csv) #, header=0, index_col=0 #, encoding='utf-8'
#all_investor_idx = pd.read_excel(INVESTORS_PATH, sheet_name='PROPSWAP', header=0) #, engine='openpyxl'


## VARIABLE ASSIGNMENT

all_investor_idx = all_investor_idx.sort_values(by='TTL_VOL_RANK')

mf_num_cols = ['MF_AVG_PRICE_MM', 'MF_UNITS_PROP', 'MF_AVG_PPU',  'AVG_QUALITY', 'MF_QUALITY', 'TTL_VOL_RANK', 'TTL_SF_RANK', 'MF_VOL_RANK',]

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

# for i in mf_num_cols:
#     pd.to_numeric(all_investor_idx[i])

# print(all_investor_idx.info())


# #%%

#####################
### STREAMLIT APP ###
#####################

st.container()
left_column, right_column = st.columns(2)
left_button = left_column.button('PROP/SWAP')
right_button = right_column.button('UNIVERSE')
if left_button:
    left_column.write('*ERROR: CURRENTLY UNDER MAINTENANCE*')
if right_button:

### TABLEAU LINK ###
    #st.write('REAL ESTATE INVESTOR UNIVERSE:')
    right_column.write('https://public.tableau.com/shared/S4GKR7QYB?:display_count=n&:origin=viz_share_link')

# <div class='tableauPlaceholder' id='viz1659298369110' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;S4&#47;S4GKR7QYB&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;S4GKR7QYB' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;S4&#47;S4GKR7QYB&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1659298369110');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
# vizElement.parentNode.insertBefore(scriptElement, vizElement);
# </script>

st.title('PROP/SWAP')
st.header('*VIRTUAL CRE BROKER*')

prop_params_header = st.subheader('PROPERTY PARAMETERS:')

sector = st.selectbox(
    '*PROPERTY TYPE:',
    ("MULTIFAMILY",
     "STRIP CENTER", "NNN RETAIL", "MALL",
     "SELF-STORAGE", "INDUSTRIAL",
     "FULL-SERVICE HOTEL", "LIMITED-SERVICE HOTEL",
     "CBD OFFICE", "SUBURBAN OFFICE"))

with st.form("PROPERTY PARAMETERS"):
    if sector == "MULTIFAMILY":
        prop_size = st.slider('*TOTAL MF UNITS: [25-1,000 UNITS]', min_value = 0, max_value = 1000, step = 25)
        #prop_size = st.selectbox('*TOTAL MF UNITS: [25-1,000 UNITS]', list(range(25,750,25)))
    if sector == "FULL-SERVICE HOTEL":
        prop_size = st.selectbox('*TOTAL FS KEYS: [25-1,000 KEYS]', list(range(25,750,25)))
    if sector == "LIMITED-SERVICE HOTEL":
        prop_size = st.selectbox('*TOTAL LS KEYS: [25-1,000 KEYS]', list(range(25,750,25)))
    if sector == "STRIP CENTER":
        prop_size = st.selectbox('*TOTAL SC SF: [5K-1MM SF]', list(range(5000,1005000,5000)))
    if sector == "NNN RETAIL":
        prop_size = st.selectbox('*TOTAL NNN SF: [5K-500k SF]', list(range(5000,505000,5000)))
    if sector == "MALL":
        prop_size = st.selectbox('*TOTAL MALL SF: [10K-1MM SF]', list(range(10000,1010000,10000)))
    if sector == "SELF-STORAGE":
        prop_size = st.selectbox('*TOTAL SELF-STORAGE SF: [5K-500K SF]', list(range(0,525000,25000)))
    if sector == "INDUSTRIAL":
        prop_size = st.selectbox('*TOTAL INDUSTRIAL SF: [5K-1MM SF]', list(range(5000,1005000,5000)))
    if sector == "CBD OFFICE":
        prop_size = st.selectbox('*TOTAL CBD OFFICE SF: [10K-500K SF]', list(range(10000,505000,5000)))
    if sector == "SUBURBAN OFFICE":
        prop_size = st.selectbox('*TOTAL SUB OFFICE SF: [10K-500K SF]', list(range(10000,505000,5000)))

#streamlit. slider ( label , min_value=None , max_value=None , value=None , step=None , format=None , key=None )

    min_prop_price = st.slider('*MINIMUM SALE PRICE [$0MM-$100MM]:', min_value = 0, max_value = 100, step = 5)
        #min_prop_price = st.selectbox('*MINIMUM PRICE [$0MM-$100MM]:', (list(range(0,105,5))))

    # property_region = st.selectbox('*PROPERTY REGION:', ("NORTHEAST", "MID-ATLANTIC", "SOUTHEAST", "WEST", "NORTHWEST", "MIDWEST", "SOUTHWEST"))

    prop_qual = st.selectbox('*PROPERTY QUALITY [1-5]:',
                             list(range(1,6,1)))

    # prop_cap_rate = st.selectbox('*EST. CAP RATE:',
    #                          list(range(1, 6, 1)))

    if min_prop_price == 0:
        st.write('PLEASE INPUT VALUE ABOVE $0')

    # elif min_prop_price > 0:
    #     implied_ppu_title = st.write('*IMPLIED VALUE / UNIT:')
    #     implied_ppu = st.markdown(round(min_prop_price * 1_000_000 / prop_size))

    params_submit = st.form_submit_button("PROP/SWAP")

### PICKLE PICKLE PICKLE ###

    @st.cache(persist=True, allow_output_mutation=True)
    def filter_buyers(sector, prop_size, min_prop_price, prop_qual):
      if sector == 'MULTIFAMILY':
        for investors in all_investor_idx:
          mf_size_filter = all_investor_idx[all_investor_idx.MF_UNITS_PROP >= prop_size]
          mf_min_price_filter = mf_size_filter[mf_size_filter.MF_AVG_PRICE_MM >= min_prop_price]
          mf_qual_filter = mf_min_price_filter[(mf_min_price_filter.MF_QUALITY >= (prop_qual-1)) & (mf_min_price_filter.MF_QUALITY <= (prop_qual+1))]
          mf_buyer_recs = mf_qual_filter.sort_values(by = 'MF_VOL_RANK', ascending = True)[:50]
          mf_buyer_recs = pd.DataFrame(data = mf_buyer_recs, columns = mf_cols)
        return mf_buyer_recs
      elif sector == 'STRIP CENTER':
        for investors in all_investor_idx:
          sc_size_filter = all_investor_idx[all_investor_idx['SC_SF_PROP'] >= prop_size]
          sc_min_price_filter = sc_size_filter[sc_size_filter['SC_AVG_PRICE_MM'] >= min_prop_price]
          sc_qual_filter = sc_min_price_filter[(sc_min_price_filter['SC_QUALITY'] >= (prop_qual-1)) & (sc_min_price_filter['SC_QUALITY'] <= (prop_qual+1))]
          sc_buyer_recs = sc_qual_filter.sort_values(by = 'SC_VOL_RANK', ascending = True)[:25]
          sc_buyer_recs = pd.DataFrame(data = sc_buyer_recs, columns = sc_cols)
        return sc_buyer_recs
      elif sector == 'NNN RETAIL':
        for investors in all_investor_idx:
          nnn_size_filter = all_investor_idx[all_investor_idx['NNN_SF_PROP'] >= prop_size]
          nnn_min_price_filter = nnn_size_filter[nnn_size_filter['NNN_AVG_PRICE_MM'] >= min_prop_price]
          nnn_qual_filter = nnn_min_price_filter[(nnn_min_price_filter['NNN_QUALITY'] >= (prop_qual-1)) & (nnn_min_price_filter['NNN_QUALITY'] <= (prop_qual+1))]
          nnn_buyer_recs = nnn_qual_filter.sort_values(by = 'NNN_VOL_RANK', ascending = True)[:25]
          nnn_buyer_recs = pd.DataFrame(data = nnn_buyer_recs, columns = nnn_cols)
        return nnn_buyer_recs
      elif sector == 'MALL':
        for investors in all_investor_idx:
          mall_size_filter = all_investor_idx[all_investor_idx['MALL_SF_PROP'] >= prop_size]
          mall_min_price_filter = mall_size_filter[mall_size_filter['MALL_AVG_PRICE_MM'] >= min_prop_price]
          mall_qual_filter = mall_min_price_filter[(mall_min_price_filter['MALL_QUALITY'] >= (prop_qual-2)) & (mall_min_price_filter['MALL_QUALITY'] <= (prop_qual+2))]
          mall_buyer_recs = mall_qual_filter.sort_values(by = 'MALL_VOL_RANK', ascending = False)[:10]
          mall_buyer_recs = pd.DataFrame(data = mall_buyer_recs, columns = mall_cols)
        return mall_buyer_recs
      elif sector == 'SELF-STORAGE':
        for investors in all_investor_idx:
          ss_size_filter = all_investor_idx[all_investor_idx['SS_SF_PROP'] >= prop_size]
          ss_min_price_filter = ss_size_filter[ss_size_filter['SS_AVG_PRICE_MM'] >= min_prop_price]
          ss_qual_filter = ss_min_price_filter[(ss_min_price_filter['SS_QUALITY'] >= (prop_qual-1)) & (ss_min_price_filter['SS_QUALITY'] <= (prop_qual+1))]
          ss_buyer_recs = ss_qual_filter.sort_values(by = 'SS_VOL_RANK', ascending = True)[:25]
          ss_buyer_recs = pd.DataFrame(data = ss_buyer_recs, columns = ss_cols)
        return ss_buyer_recs
      elif sector == 'INDUSTRIAL':
        for investors in all_investor_idx:
          ind_size_filter = all_investor_idx[all_investor_idx['IND_SF_PROP'] >= prop_size]
          ind_min_price_filter = ind_size_filter[ind_size_filter['IND_AVG_PRICE_MM'] >= min_prop_price]
          ind_qual_filter = ind_min_price_filter[(ind_min_price_filter['IND_QUALITY'] >= (prop_qual-1)) & (ind_min_price_filter['IND_QUALITY'] <= (prop_qual+1))]
          ind_buyer_recs = ind_qual_filter.sort_values(by = 'IND_VOL_RANK', ascending = True)[:25]
          ind_buyer_recs = pd.DataFrame(data = ind_buyer_recs, columns = ind_cols)
        return ind_buyer_recs
      elif sector == 'FULL-SERVICE HOTEL':
        for investors in all_investor_idx:
          fs_size_filter = all_investor_idx[all_investor_idx['FS_KEYS_PROP'] >= prop_size]
          fs_min_price_filter = fs_size_filter[fs_size_filter['FS_AVG_PRICE_MM'] >= min_prop_price]
          fs_qual_filter = fs_min_price_filter[(fs_min_price_filter['FS_QUALITY'] >= (prop_qual-1)) & (fs_min_price_filter['FS_QUALITY'] <= (prop_qual+1))]
          fs_buyer_recs = fs_qual_filter.sort_values(by = 'FS_VOL_RANK', ascending = True)[:25]
          fs_buyer_recs = pd.DataFrame(data = fs_buyer_recs, columns = fs_cols)
        return fs_buyer_recs
      elif sector == 'LIMITED-SERVICE HOTEL':
        for investors in all_investor_idx:
          ls_size_filter = all_investor_idx[all_investor_idx['LS_KEYS_PROP'] >= prop_size]
          ls_min_price_filter = ls_size_filter[ls_size_filter['LS_AVG_PRICE_MM'] >= min_prop_price]
          ls_qual_filter = ls_min_price_filter[(ls_min_price_filter['LS_QUALITY'] >= (prop_qual-1)) & (ls_min_price_filter['LS_QUALITY'] <= (prop_qual+1))]
          ls_buyer_recs = ls_qual_filter.sort_values(by = 'LS_VOL_RANK', ascending = True)[:25]
          ls_buyer_recs = pd.DataFrame(data = ls_buyer_recs, columns = ls_cols)
        return ls_buyer_recs
      elif sector == 'CBD OFFICE':
        for investors in all_investor_idx:
          cbd_size_filter = all_investor_idx[all_investor_idx['CBD_SF_PROP'] >= prop_size]
          cbd_min_price_filter = cbd_size_filter[cbd_size_filter['CBD_AVG_PRICE_MM'] >= min_prop_price]
          cbd_qual_filter = cbd_min_price_filter[(cbd_min_price_filter['CBD_QUALITY'] >= (prop_qual-1)) & (cbd_min_price_filter['CBD_QUALITY'] <= (prop_qual+1))]
          cbd_buyer_recs = cbd_qual_filter.sort_values(by = 'CBD_VOL_RANK', ascending = True)[:25]
          cbd_buyer_recs = pd.DataFrame(data = cbd_buyer_recs, columns = cbd_cols)
        return cbd_buyer_recs
      elif sector == 'SUB OFFICE':
        for investors in all_investor_idx:
          sub_size_filter = all_investor_idx[all_investor_idx['SUB_SF_PROP'] >= prop_size]
          sub_min_price_filter = sub_size_filter[sub_size_filter['SUB_AVG_PRICE_MM'] >= min_prop_price]
          sub_qual_filter = sub_min_price_filter[(sub_min_price_filter['SUB_QUALITY'] >= (prop_qual-1)) & (sub_min_price_filter['SUB_QUALITY'] <= (prop_qual+1))]
          sub_buyer_recs = sub_qual_filter.sort_values(by = 'SUB_VOL_RANK', ascending = True)[:25]
          sub_buyer_recs = pd.DataFrame(data = sub_buyer_recs, columns = sub_cols)
        return sub_buyer_recs

## INVESTOR RECOMMENDATIONS ##
    if params_submit:
        st.write("RECOMMENDED INVESTOR POOL:")
        buyer_rec_df = filter_buyers(sector, prop_size, min_prop_price, prop_qual)
        # buyer_rec_df = buyer_rec_df.set_index('INVESTOR')
        # buyer_rec_df.set_index(0, inplace = True)

        st.dataframe(buyer_rec_df) # st.dataframe(buyer_rec_df.style.highlight_max(axis=0))

        if sector == 'MULTIFAMILY':
            per_unit_valuation = round(buyer_rec_df['MF_AVG_PPU'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED PROPERTY VALUE / UNIT:")
            st.write(per_unit_valuation)

            mf_chart_1 = px.scatter(buyer_rec_df, #all_investor_idx
                                  x=buyer_rec_df['MF_AVG_PRICE_MM'],
                                  y=buyer_rec_df['MF_AVG_PPU'],
                                  # hover_data=buyer_rec_df['INVESTOR'],
                                  color=buyer_rec_df['INVESTOR_TYPE'],
                                  color_continuous_scale='Tropic')

            st.write('TARGETED INVESTOR POOL -- VALUATION RANGE')
            st.plotly_chart(mf_chart_1, use_container_width=False, sharing="streamlit")

            # mf_chart_2 = px.bar(y=buyer_rec_df['INVESTOR_TYPE'],
            #                     x=buyer_rec_df['MF_AVG_PPU'],
            #                     color=buyer_rec_df['INVESTOR_TYPE'],
            #                     color_continuous_scale='Tropic')
            #
            # st.write('TARGETED INVESTOR POOL -- VALUATION RANGE')
            # st.plotly_chart(mf_chart_2)


            # plt.figure(figsize = (30, 20))
            # fig, ax = plt.subplots()
            # sns.barplot(y = buyer_rec_df['INVESTOR_TYPE'], x = buyer_rec_df['MF_AVG_PPU'], palette = 'mako', ci = None, orient = 'h')
            # plt.xlabel('AVG MULTIFAMILY PPU', fontsize = 18)
            # plt.ylabel('INVESTOR TYPE', fontsize = 18)
            # plt.legend(loc = "best")
            # st.pyplot(fig)

        elif sector == 'STRIP CENTER':
            per_unit_valuation = round(buyer_rec_df['SC_AVG_PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)

            sc_chart_1 = px.scatter(buyer_rec_df,  # all_investor_idx
                                    x=buyer_rec_df['SC_AVG_PRICE_MM'],
                                    y=buyer_rec_df['SC_AVG_PSF'],
                                    # hover_data=buyer_rec_df['INVESTOR'],
                                    color=buyer_rec_df['INVESTOR_TYPE'],
                                    size=buyer_rec_df['INVESTOR_TYPE'],
                                    color_continuous_scale='Tropic')

            st.write('TARGETED INVESTOR POOL -- VALUATION RANGE')
            st.plotly_chart(sc_chart_1, use_container_width=False, sharing="streamlit")

            # sc_chart_2 = px.parallel_categories(buyer_rec_df,
            #                                     color=buyer_rec_df['INVESTOR_TYPE'],
            #                                     color_continuous_scale='Tropic',) #px.colors.sequential.Inferno
            #
            #
            # st.write('TARGETED INVESTOR POOL -- VALUATION RANGE')
            # st.plotly_chart(sc_chart_2, use_container_width=False, sharing="streamlit")


            # sc_chart_2 = px.bar(y=buyer_rec_df['INVESTOR_TYPE'],
            #                     x=buyer_rec_df['SC_AVG_PSF'],
            #                     color=buyer_rec_df['INVESTOR_TYPE'],
            #                     color_continuous_scale='Tropic')

                # barmode = "group"
                #pattern_shape = "nation", pattern_shape_sequence = [".", "x", "+"]

            # fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group", facet_row="time", facet_col="day",
            #        category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

            # fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
            #                         color="species")

            # fig = px.parallel_categories(df, color="size", color_continuous_scale=px.colors.sequential.Inferno)

            # fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
            #                   "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
            #                   "petal_width": "Petal Width", "petal_length": "Petal Length", },
            #                     color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

            # st.write('TARGETED INVESTOR POOL -- VALUATION RANGE')
            # st.plotly_chart(sc_chart_2)


        elif sector == 'NNN RETAIL':
            per_unit_valuation = round(buyer_rec_df['NNN_AVG_PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)


        elif sector == 'MALL':
            per_unit_valuation = round(buyer_rec_df['MALL_AVG_PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)


        elif sector == 'SELF-STORAGE':
            per_unit_valuation = round(buyer_rec_df['SS_AVG_PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)



        elif sector == 'INDUSTRIAL':
            per_unit_valuation = round(buyer_rec_df['IND_AVG_PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)


        elif sector == 'FULL-SERVICE HOTEL':
            per_unit_valuation = round(buyer_rec_df['FS_AVG_PPK'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE / KEY:")
            st.write(per_unit_valuation)


        elif sector == 'LIMITED-SERVICE HOTEL':
            per_unit_valuation = round(buyer_rec_df['LS_AVG_PPK'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE / KEY:")
            st.write(per_unit_valuation)



        elif sector == 'CBD OFFICE':
            per_unit_valuation = round(buyer_rec_df['CBD_AVG_PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)


        elif sector == 'SUB OFFICE':
            per_unit_valuation = round(buyer_rec_df['SUB_AVG_PSF'].mean())
            prop_valuation = per_unit_valuation * prop_size
            st.write("ESTIMATED PROPERTY VALUE ($MM):")
            st.write(prop_valuation / 1_000_000)
            st.write("ESTIMATED VALUE PSF:")
            st.write(per_unit_valuation)


# style={'textAlign': 'Center', 'border': '4px solid black', 'font-family': 'Arial'}


### EXPLAIN QUALITY SCALE ###

## CREDITS / FOOTNOTES
st.success('THANKS FOR PROP/SWAPPING')
    #st.warning('NO BUYERS FOUND')
# st.write('*~PROP/SWAP BETA MODE~*')
st.stop()



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

#st.spinner()
#with st.spinner(text='CONNECTING'):
#    time.sleep(5)
#    st.success('LIVE')
