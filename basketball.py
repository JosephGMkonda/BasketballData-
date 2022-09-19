## importing libraries

from urllib.parse import urlsplit
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as pyt
import seaborn as sns
import numpy as np


## setting up the title

st.title(" NBS player statistics")

## setting up the sidebar with input features

st.sidebar.header(" User Input features ")
selectedYear = st.sidebar.selectbox('year',list(reversed(range( 1950,2022))))

# loading data through web scrapping 

@st.cache
def loading_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url,header=0)
    df = html[0]
    # deleting the repeated headers in the content
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'],axis=1)
    return playerstats

playerstats = loading_data(selectedYear)

# side bar positioning

sorting_teams = sorted(playerstats.Tm.unique())
selected_teams = st.sidebar.multiselect('Team',sorting_teams,sorting_teams)

unique_pos =  ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('position',unique_pos,unique_pos)

df_selected_team = playerstats[(playerstats.Tm.isin(selected_teams)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display player of the selected teams')
st.write('Data Dimension' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + 'colums')
st.dataframe(df_selected_team)

if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = pyt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()