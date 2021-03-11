import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from dateutil.parser import parse
import json, re
import matplotlib.pyplot as plt
import seaborn as sns     
import altair as alt
import plotly.express as px
import nltk
import wordcloud
from functools import reduce
# from flatten_dict import flatten
from pprint import pprint

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
       'anagrafica.altro_lavoro', 'anagrafica.comune','anni_di_attivita','anagrafica.perc_reddito','anagrafica.perc_tempo',
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

def buildAttivitaList():
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
  return q;

def buildAttivitaQuery():  
  return ' | '.join(buildAttivitaList())

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

AttivitaQuery = buildAttivitaQuery();

if (len(AttivitaQuery) == 0):
  selectedAnagrafiche = anagrafiche
else:
  selectedAnagrafiche = anagrafiche.query(AttivitaQuery)

ProvinciaQuery = buildProvinciaQuery();
if (len(ProvinciaQuery) > 0):
  selectedAnagrafiche = selectedAnagrafiche.query(ProvinciaQuery)

IndAssocQuery = buildIndAssocQuery();
if (len(IndAssocQuery) > 0):
  selectedAnagrafiche = selectedAnagrafiche.query(IndAssocQuery)

###############
## START 
##############

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

timeismoney = selectedAnagrafiche[['anagrafica.perc_tempo','anagrafica.perc_reddito','anagrafica.provincia']].fillna(0).astype({'anagrafica.perc_tempo':int,'anagrafica.perc_reddito':int,'anagrafica.provincia':str}).query('`anagrafica.perc_tempo` > 0')


fig = px.scatter(timeismoney, x="anagrafica.perc_tempo", y="anagrafica.perc_reddito", color="anagrafica.provincia")
fig.update_xaxes(title='Percentuale Tempo Dedicato')
fig.update_yaxes(title='Percentuale Reddito')
st.plotly_chart(fig,use_container_width=True)

# tempo = pd.to_numeric(selectedAnagrafiche['anagrafica.perc_tempo']).fillna(0)
# tempo = tempo[tempo > 0].mean()
# reddito = pd.to_numeric(selectedAnagrafiche['anagrafica.perc_reddito']).fillna(0)
# reddito = reddito[reddito > 0].mean()
# tempo
# reddito
############################  
## WORD CLOUD
"""
## WORD CLOUD
"""

activities = {
    "produzione": [
        "Arti Visive",
        "Artigianato Artistico",
        "Costumi",
        "Danza",
        "Direzione Orchestra / Coro",
        "Disegno / Comics",
        "Game Art",
        "Musica",
        "Produzione Cinematografica",
        "Regia",
        "Sceneggiatura",
        "Scenografia",
        "Scrittura",
        "Spettacolo Itinerante",
        "Teatro / Performance",
        "Video / Film Making"
    ],
    "organizzazione": [
        "Gestione Istituzione culturale",
        "Gestione struttura / associazione spettacolo dal vivo",
        "Gestione sale cinematografiche",
        "Gestione spazi polifunzionali",
        "Curatela e/o organizzazione di mostre"
    ],
    "educazione": [
        "Laboratori con le scuole",
        "Corsi destinati alle aziende",
        "Laboratori in Istituzioni culturali",
        "Corsi individuali / destinati a privati"
    ],
    "promozione": [
        "Cicli di conferenze",
        "Presentazioni / Premi letterari",
        "Incontri / Seminari",
        "Indagini / Studi / Pubblicazioni",
        "Agente / Gallerista"
    ],
    "supporto": [
        "Attrezzista",
        #"Doppiaggio",
        "Effetti speciali",
        "Fornitura servizi museali",
        "Light designer",
        "Macchinista",
        "Montaggio",
       # "Restauro",
        "Rigger",
        "Servizi biglietteria",
        "Servizi ristorazione",
        "Supporto tecnico alla regia",
        "Tecnico del suono",
        "Tecnico di scena",
        "Truccatore"
    ]
}


textCols = list(filter(lambda s: re.search('riflessioni',s), df.columns));

@st.cache
def getFullFiltered(df):
  AttivitaQuery = buildAttivitaQuery();
  if (len(AttivitaQuery) == 0):
    fullFiltered = df
  else:
    fullFiltered = df.query(AttivitaQuery)

  ProvinciaQuery = buildProvinciaQuery();
  if (len(ProvinciaQuery) > 0):
    fullFiltered = fullFiltered.query(ProvinciaQuery)

  IndAssocQuery = buildIndAssocQuery();
  if (len(IndAssocQuery) > 0):
    fullFiltered = fullFiltered.query(IndAssocQuery)
  
  return fullFiltered


fullFiltered = getFullFiltered(df);


def buildWordCloud(AttivitaQuery,fullFiltered,textCols):
  wordcloudColumns =  list(filter(lambda s: re.search('['+AttivitaQuery+']',s), textCols));

  wordCloud = fullFiltered[wordcloudColumns].fillna('').copy()
  wordCloud['final'] = '';
  for k in wordcloudColumns:
    wordCloud['final'] = wordCloud['final'] + ' ' + wordCloud[k].map(str)

  allwords = wordCloud['final'].str.cat(sep=' ')

  from nltk.corpus import stopwords
  from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


  stop_words = set(stopwords.words('italian'))
  xtra_stops = set(["c'è",'già','me'])

  wc = WordCloud(colormap="hot", max_words=100, 
      stopwords=(stop_words | xtra_stops),width=1400, height=1400)
  wc.generate(allwords)
#image_colors = ImageColorGenerator(image)

  # show the figure
  f = plt.figure(figsize=(1400,1400))
  fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [100, 1]})
  axes[0].imshow(wc, interpolation="bilinear")
  for ax in axes:
          ax.set_axis_off()
  return fig

#st.pyplot(buildWordCloud(AttivitaQuery,fullFiltered,textCols))  #todo


### activities data
fullFiltered

"""
## COMPOSIZIONE DELLE ATTIVITA'
"""

import plotly.graph_objects as go

@st.cache
def plotSezioni():
  labels = list(activities.keys())
  sezioni = fullFiltered[labels].sum()
  fig = go.Figure(data=[go.Pie(labels=labels, values=sezioni)])
  return fig

#st.plotly_chart(plotSezioni(),use_container_width=True)
def myIsNa(val):
  return (str(val) == 'nan')

def getCostiFromActivityRow(row, k):
  keys = [n for n in row.keys() if n.startswith(k+'.costi')]
  return {z.replace(k+'.costi.',''): row[z] for z in keys}

def getMisureFromActivityRow(row, k):
  keys = [n for n in row.keys() if n.startswith(k+'.misure')]
  return {z.replace(k+'.misure.',''): row[z] for z in keys}

def getActivities(fullFiltered):
  activitiesNorm = [];
  for index, row in fullFiltered.iterrows():
    for section, acts in activities.items():
      for act in acts:
        k = section + '.activities.' + act;
        qR = k + '.quotaReddito'
        myrow = row.to_dict()
        if((k+'.riflessioni') in list(myrow.keys())):
          riflessioni = myrow[k+'.riflessioni']
        else:
          riflessioni = ''
        costi = getCostiFromActivityRow(myrow, k)
        misure = getMisureFromActivityRow(myrow, k)
        if(not(myIsNa(row[qR]))):
          newRow = {
            'role': section,
            'activity': act,
            'anagrafica.provincia': row['anagrafica.provincia'],
            'anagrafica.ind_assoc': row['anagrafica.ind_assoc'],
            'quotaReddito': row[qR],
            'quotaTempo': row[k+'.quotaTempo'],
            'fonti.servizi_settore_privato':  row[k+'.fonti.servizi_settore_privato'],
            'fonti.servizi_settore_pubblico':  row[k+'.fonti.servizi_settore_pubblico'],
            'fonti.contributi_settore_privato':  row[k+'.fonti.contributi_settore_privato'],
            'fonti.contributi_settore_pubblico':  row[k+'.fonti.contributi_settore_pubblico'],
            'fonti.autofinanziamento':  row[k+'.fonti.autofinanziamento'],
            'fonti.diritti':  row[k+'.fonti.diritti'],
            'reddito': row[k+'.reddito'],
            'prospettive': row[k+'.prospettive'],
            'riflessioni': riflessioni,
            **costi,
            **misure    
          }
          activitiesNorm.append(newRow)
          
  return pd.DataFrame(activitiesNorm)

activitiesNorm = getActivities(fullFiltered)
# fullFiltered.iloc[0]
# raga = fullFiltered.iloc[0].to_dict().keys()
activitiesNorm

# sezioni = [];
# attivita = [];
# for dataEl in dbdata:
#   for section, acts in activities.items():
#     if section in dataEl:
#       # sezioni.append([
#       #   section,*dataEl[section]['ambiti'].values()
#       # ])
#       newEl = dataEl[section]['ambiti']
#       newEl['section'] = section
#       sezioni.append(newEl)
#       for act in dataEl[section]['activities']:
#         # actData = flatten(dataEl[section]['activities'][act],reducer='dot')
#         actData = dataEl[section]['activities'][act]
#         actData['anagrafica'] = dataEl['anagrafica']
#         actData['section'] = section
#         actData['activity'] = act
#         attivita.append(actData)
#         # attivita.append([section,act,*dataEl[section]['ambiti'].values(),*actData.values()]);

# attivitaDF = pd.json_normalize(attivita).fillna(0)
# attivitaDF

# attivitaDF = pd.json_normalize(attivita).fillna(0)
# sezioniDF = pd.json_normalize(sezioni).fillna(0).query("`ambito_provinciale`> 0 | `ambito_regionale`> 0 | `ambito_nazionale`> 0 | `ambito_internazionale`> 0")
# sezioniDF['Sezione'] = sezioniDF['section']
# sezioniDF.rename(columns={'section': 'compilazioni'}, inplace=True)
# SezioniDescribe = sezioniDF.groupby('compilazioni').agg(
#     {
#          'compilazioni':"count",
#          'ambito_provinciale': "mean",  
#          'ambito_regionale': 'mean',
#          'ambito_nazionale': "mean",  
#          'ambito_internazionale': 'mean',
#          'Sezione': 'first'
#     }).copy()
