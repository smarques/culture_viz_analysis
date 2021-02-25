import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from dateutil.parser import parse
import json
import matplotlib.pyplot as plt
import seaborn as sns     
import altair as alt
import plotly.express as px

#st.set_page_config(layout="wide")

file_object =  open('data.json', 'r')
dbdata = json.loads(file_object.read())
df = pd.json_normalize(dbdata)
df.drop('_id', axis =1, inplace = True)
possibleAreas = ['organizzazione','produzione','supporto','promozione','educazione']

for area in possibleAreas:
  df[area] = 0;
for index in df.index:
  for area in possibleAreas:
    if area in df.loc[index, 'roles']:
        df.loc[index, area] = 1

def annolo(s):
  if(s == '14 maggio 1985'):
    return 1985
  if(s == 'giugno 2020'):
    return 2020
  if(s == '11101970'):
    return 1970
  if(s =='12/07/1095'):
    return 1995
  try:
    p = parse(s)
  except:
    print(s)
  else:
    return (int)(p.year)

df['anni_di_attivita'] = df['anagrafica.data'].map(lambda x: 2021 - annolo(str(x))) 
#df.anni = df.anni.astype(int) 
# df.drop(columns=['roles','anagrafica.data'], inplace=True)

df['anagrafica.titoloStudio'].replace({'':'non compilato'}, inplace=True)

anagrafiche = df.loc[:,['anagrafica.ind_assoc', 'anagrafica.provincia','anagrafica.data',
       'anagrafica.formaGiuridica', 'anagrafica.titoloStudio',
       'anagrafica.altro_lavoro', 'anagrafica.comune','anni_di_attivita',
       'organizzazione', 'produzione', 'supporto', 'promozione', 'educazione']]




st.sidebar.markdown("## Attività")

organizzazione_cb = st.sidebar.checkbox('Organizzazione', True)
produzione_cb = st.sidebar.checkbox('Produzione', True)
supporto_cb = st.sidebar.checkbox('Supporto', True)
promozione_cb = st.sidebar.checkbox('Promozione', True)
educazione_cb = st.sidebar.checkbox('Educazione', True)


st.sidebar.markdown("## Provincia")
bl_cb = st.sidebar.checkbox('Belluno', True)
pd_cb = st.sidebar.checkbox('Padova', True)
ro_cb = st.sidebar.checkbox('Rovigo', True)
tv_cb = st.sidebar.checkbox('Treviso', True)
ve_cb = st.sidebar.checkbox('Venezia', True)
vr_cb = st.sidebar.checkbox('Verona', True)
vi_cb = st.sidebar.checkbox('Vicenza', True)

st.sidebar.markdown("## Individuo vs Associazione")
ind_cb = st.sidebar.checkbox('Individuo', True)
assoc_cb = st.sidebar.checkbox('Associazione', True)


def buildAttivitaQuery():
  q = []
  if(organizzazione_cb):
    q.append('organizzazione == 1')
  if(produzione_cb):
    q.append('produzione == 1')
  if(supporto_cb):
    q.append('supporto == 1')
  if(promozione_cb):
    q.append('promozione == 1')
  if(educazione_cb):
    q.append('educazione == 1')
  return ' | '.join(q)

def buildProvinciaQuery():
  q = []
  if(bl_cb):
    q.append('`anagrafica.provincia` == "BL"')
  if(pd_cb):
    q.append('`anagrafica.provincia` == "PD"')
  if(ro_cb):
    q.append('`anagrafica.provincia` == "RO"')
  if(tv_cb):
    q.append('`anagrafica.provincia` == "TV"')
  if(vi_cb):
    q.append('`anagrafica.provincia` == "VI"')
  if(ve_cb):
    q.append('`anagrafica.provincia` == "VE"')
  if(vr_cb):
    q.append('`anagrafica.provincia` == "VR"')
  return ' | '.join(q)

def buildIndAssocQuery():
  q = []
  if(ind_cb):
    q.append('`anagrafica.ind_assoc` == "individuo"')
  if(assoc_cb):
    q.append('`anagrafica.ind_assoc` == "associazione"')
  return ' | '.join(q)

qa = buildAttivitaQuery();
if (len(qa) == 0):
  selectedAnagrafiche = anagrafiche
else:
  selectedAnagrafiche = anagrafiche.query(qa)

qp = buildProvinciaQuery();
if (len(qp) > 0):
  selectedAnagrafiche = selectedAnagrafiche.query(qp)

qia = buildIndAssocQuery();
if (len(qia) > 0):
  selectedAnagrafiche = selectedAnagrafiche.query(qia)

st.title('Visualising Culture')
"""
## Anagrafica
"""
cols = ['anagrafica.ind_assoc', 'anagrafica.provincia',
       'anagrafica.formaGiuridica', 'anagrafica.titoloStudio',
       'anagrafica.altro_lavoro', 'anagrafica.comune','anni_di_attivita','anagrafica.data']
st_ms = st.multiselect("Scegli le colonne della tabella", cols, default=cols)
selectedAnagrafiche.loc[:,st_ms]

f = px.histogram(selectedAnagrafiche, x='anagrafica.provincia', title='Compilazioni per provincia')
f.update_xaxes(title='Provincia')
f.update_yaxes(title='No. di compilazioni',range=[0, 100])
st.plotly_chart(f)

f2 = px.histogram(selectedAnagrafiche, x='anni_di_attivita', title='Anni Di Attività', color="anagrafica.provincia")
f2.update_xaxes(title='Anni Di Attività',range=[0, 110])
f2.update_yaxes(title='No. di casi',range=[0, 50])
st.plotly_chart(f2)

f = px.histogram(selectedAnagrafiche, x='anagrafica.ind_assoc', title='Individuo vs Associazione',color="anagrafica.provincia")
f.update_xaxes(title='Ind vs Assoc')
f.update_yaxes(title='No. di casi',range=[0, 250])
st.plotly_chart(f)

f = px.histogram(selectedAnagrafiche, x='anagrafica.formaGiuridica', title='Forma Giuridica',color="anagrafica.provincia")
f.update_xaxes(title='Forma Giuridica')
f.update_yaxes(title='No. di casi',range=[0, 110])
st.plotly_chart(f)

f = px.histogram(selectedAnagrafiche, x='anagrafica.titoloStudio', title='Titolo di Studio',color="anagrafica.provincia")
f.update_xaxes(title='Titolo di Studio')
f.update_yaxes(title='No. di casi',range=[0, 150])
st.plotly_chart(f)

f = px.histogram(selectedAnagrafiche, x='anagrafica.altro_lavoro', title='Altra Occupazione',color="anagrafica.provincia")
f.update_xaxes(title='Altra Occupazione')
f.update_yaxes(title='No. di casi')
st.plotly_chart(f)