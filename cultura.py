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
from flatten_dict import flatten
from pprint import pprint
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.graph_objects as go
from copy import deepcopy

#st.set_page_config(layout="wide")

######### GENERAL

nltk.download('stopwords')
attivitaFilters = {}
provinciaFilters = {}
indAssocFilters = {}

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

def main():
  st.title('Visualising Culture')
  st.sidebar.title("Sezione")
  dfImmut = unwrangleMainDataFrame()

  #df = dfImmut.copy();
  df = deepcopy(dfImmut)
  app_mode = st.sidebar.selectbox("",[ "Anagrafiche", "Attività"]) #"Introduzione",
  if app_mode == "Introduzione":
    """
    ## Introduzione
    """
    st.sidebar.success('Coming Soon')
  elif app_mode == "Anagrafiche":
    main_anagrafiche(df)
  elif app_mode == "Attività":
    main_attivita(df)


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

@st.cache 
def unwrangleMainDataFrame():
  possibleAreas = ['organizzazione','produzione','supporto','promozione','educazione']
  file_object =  open('data.json', 'r')
  dbdata = json.loads(file_object.read())

  df = pd.json_normalize(dbdata)
  df.drop('_id', axis =1, inplace = True)

  for area in possibleAreas:
    df[area] = 0;
  for index in df.index:
    for area in possibleAreas:
      if area in df.loc[index, 'roles']:
          df.loc[index, area] = 1
  df['anni_di_attivita'] = df['anagrafica.data'].map(lambda x: 2021 - annolo(str(x))) 
  df['anagrafica.titoloStudio'].replace({'':'non compilato'}, inplace=True)

  return df

@st.cache 
def getActivitiesFullFiltered(df,attivitaFilters, provinciaFilters, indAssocFilters):
  AttivitaQuery = buildNormalizedAttivitaQuery(attivitaFilters);
  
  if (len(AttivitaQuery) == 0):
    fullFiltered = df
  else:
    fullFiltered = df.query(AttivitaQuery)

  ProvinciaQuery = buildProvinciaQuery(provinciaFilters);
  if (len(ProvinciaQuery) > 0):
    fullFiltered = fullFiltered.query(ProvinciaQuery)

  IndAssocQuery = buildIndAssocQuery(indAssocFilters);
  if (len(IndAssocQuery) > 0):
    fullFiltered = fullFiltered.query(IndAssocQuery)
  
  return fullFiltered

@st.cache
def getFullFiltered(df,attivitaFilters, provinciaFilters, indAssocFilters):
  AttivitaQuery = buildAttivitaQuery(attivitaFilters);
  if (len(AttivitaQuery) == 0):
    fullFiltered = df
  else:
    fullFiltered = df.query(AttivitaQuery)

  ProvinciaQuery = buildProvinciaQuery(provinciaFilters);
  if (len(ProvinciaQuery) > 0):
    fullFiltered = fullFiltered.query(ProvinciaQuery)

  IndAssocQuery = buildIndAssocQuery(indAssocFilters);
  if (len(IndAssocQuery) > 0):
    fullFiltered = fullFiltered.query(IndAssocQuery)
  
  return fullFiltered

def buildAttivitaList(attivitaFilters):
    q = []
    if(attivitaFilters['organizzazione_cb']):
      q.append('organizzazione == 1')
    if(attivitaFilters['produzione_cb']):
      q.append('produzione == 1')
    if(attivitaFilters['supporto_cb']):
      q.append('supporto == 1')
    if(attivitaFilters['promozione_cb']):
      q.append('promozione == 1')
    if(attivitaFilters['educazione_cb']):
      q.append('educazione == 1')
    return q;

def buildAttivitaQuery(attivitaFilters):  
  return ' | '.join(buildAttivitaList(attivitaFilters))

def buildNormalizedAttivitaQuery(attivitaFilters):
  q = []  
  if(attivitaFilters['organizzazione_cb']):
    q.append('role == "organizzazione"')
  if(attivitaFilters['produzione_cb']):
    q.append('role == "produzione"')
  if(attivitaFilters['supporto_cb']):
    q.append('role == "supporto"')
  if(attivitaFilters['promozione_cb']):
    q.append('role == "promozione"')
  if(attivitaFilters['educazione_cb']):
    q.append('role == "educazione"')
  return ' | '.join(q)

def buildProvinciaQuery(provinciaFilters):
  q = []
  if(provinciaFilters['bl_cb']):
    q.append('`anagrafica.provincia` == "BL"')
  if(provinciaFilters['pd_cb']):
    q.append('`anagrafica.provincia` == "PD"')
  if(provinciaFilters['ro_cb']):
    q.append('`anagrafica.provincia` == "RO"')
  if(provinciaFilters['tv_cb']):
    q.append('`anagrafica.provincia` == "TV"')
  if(provinciaFilters['vi_cb']):
    q.append('`anagrafica.provincia` == "VI"')
  if(provinciaFilters['ve_cb']):
    q.append('`anagrafica.provincia` == "VE"')
  if(provinciaFilters['vr_cb']):
    q.append('`anagrafica.provincia` == "VR"')
  return ' | '.join(q)

def buildIndAssocQuery(indAssocFilters):
  q = []
  if(indAssocFilters['ind_cb']):
    q.append('`anagrafica.ind_assoc` == "individuo"')
  if(indAssocFilters['assoc_cb']):
    q.append('`anagrafica.ind_assoc` == "associazione"')
  return ' | '.join(q)

def addFiltersToSidebar():
  global attivitaFilters,provinciaFilters,indAssocFilters
  st.sidebar.markdown("## Attività")
  attivitaFilters = {
    "organizzazione_cb" : st.sidebar.checkbox('Organizzazione', True),
    "produzione_cb" : st.sidebar.checkbox('Produzione', True),
    "supporto_cb" : st.sidebar.checkbox('Supporto', True),
    "promozione_cb" : st.sidebar.checkbox('Promozione', True),
    "educazione_cb" : st.sidebar.checkbox('Educazione', True)
  }


  st.sidebar.markdown("## Provincia")
  provinciaFilters = {
    "bl_cb" : st.sidebar.checkbox('Belluno', True),
    "pd_cb" : st.sidebar.checkbox('Padova', True),
    "ro_cb" : st.sidebar.checkbox('Rovigo', True),
    "tv_cb" : st.sidebar.checkbox('Treviso', True),
    "ve_cb" : st.sidebar.checkbox('Venezia', True),
    "vr_cb" : st.sidebar.checkbox('Verona', True),
    "vi_cb" : st.sidebar.checkbox('Vicenza', True)
  }

  st.sidebar.markdown("## Individuo vs Associazione")
  indAssocFilters = {
    "ind_cb" : st.sidebar.checkbox('Individuo', True),
    "assoc_cb" : st.sidebar.checkbox('Associazione', True)
  }

###### ANAGRAFICHE

def main_anagrafiche(df):
  global attivitaFilters,provinciaFilters,indAssocFilters
  
  def getSelectedAnagrafiche(anagrafiche,attivitaFilters,provinciaFilters,indAssocFilters):
    AttivitaQuery = buildAttivitaQuery(attivitaFilters)
    if (len(AttivitaQuery) == 0):
      selectedAnagrafiche = anagrafiche
    else:
      selectedAnagrafiche = anagrafiche.query(AttivitaQuery)
    ProvinciaQuery = buildProvinciaQuery(provinciaFilters)
    if (len(ProvinciaQuery) > 0):
      selectedAnagrafiche = selectedAnagrafiche.query(ProvinciaQuery)
    IndAssocQuery = buildIndAssocQuery(indAssocFilters)
    if (len(IndAssocQuery) > 0):
      selectedAnagrafiche = selectedAnagrafiche.query(IndAssocQuery)
    
    return selectedAnagrafiche


  
    
  def showAnagrafica(selectedAnagrafiche):

    cols = ['anagrafica.ind_assoc', 'anagrafica.provincia',
          'anagrafica.formaGiuridica', 'anagrafica.titoloStudio', 'anagrafica.mobilitati','anagrafica.dipendenti',
          'anagrafica.altro_lavoro', 'anagrafica.comune','anni_di_attivita','anagrafica.data']
    st_ms = st.multiselect("Scegli le colonne della tabella", cols, default=cols)
    selectedAnagrafiche.loc[:,st_ms]

  def dipVsMod(selectedAnagrafiche):
    dipVsMob = selectedAnagrafiche.fillna(0).astype({'anagrafica.mobilitati':int,'anagrafica.dipendenti':int,'anagrafica.provincia':str})
    dipVsMobLargest_value = dipVsMob['anagrafica.mobilitati'].max()
    totrecs = dipVsMob['anagrafica.mobilitati'].count()
    dipVsMobMax=100
    dipVsMobBins =list(range(0,10, 1)) + list(range(10,dipVsMobMax, 10)) + [dipVsMobLargest_value]
    histMob = np.histogram(dipVsMob['anagrafica.mobilitati'], bins=dipVsMobBins)
    histDip = np.histogram(dipVsMob['anagrafica.dipendenti'], bins=dipVsMobBins)
    labels = []
    for i, j in zip(histMob[1][0::1], histMob[1][1::1]):
        if j <= 10:
            labels.append('{}'.format(i))
        elif j <= dipVsMobMax:
            labels.append('{}-{}'.format(i, j))
        else:
            labels.append('> {}'.format(i))
    data = [go.Bar(x=labels, y=histMob[0]*100/totrecs,name="Mobilitati"),go.Bar(x=labels, y=histDip[0]*100/totrecs,name="Dipendenti")]

    layout = go.Layout(
        title="Mobilitati vs Dipendenti"
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

  def anagraficheTempoReddito(selectedAnagrafiche):
    timeismoney = selectedAnagrafiche[['anagrafica.perc_tempo','anagrafica.perc_reddito','anagrafica.provincia']].fillna(0).astype({'anagrafica.perc_tempo':int,'anagrafica.perc_reddito':int,'anagrafica.provincia':str}).query('`anagrafica.perc_tempo` > 0')
    f = px.scatter(timeismoney, x="anagrafica.perc_tempo", y="anagrafica.perc_reddito",height=400)
    f.update_xaxes(title='Percentuale Tempo Dedicato')
    f.update_yaxes(title='Percentuale Reddito')
    st.plotly_chart(f)

  def anagraficaCharts(selectedAnagrafiche):
    f = px.histogram(selectedAnagrafiche, x='anagrafica.provincia', title='Compilazioni per provincia',histnorm='percent')
    f.update_xaxes(title='Provincia')
    f.update_yaxes(title='% di compilazioni',range=[0, 40])
    st.plotly_chart(f)

    f = px.histogram(selectedAnagrafiche.query('`anagrafica.comune` == "Vicenza" | `anagrafica.comune` == "Padova" | `anagrafica.comune` == "Venezia" | `anagrafica.comune` == "Rovigo" | `anagrafica.comune` == "Treviso" | `anagrafica.comune` == "Belluno"'), x='anagrafica.comune', histnorm='percent',title='Compilazioni per capoluogo')
    f.update_xaxes(title='Capoluogo')
    f.update_yaxes(title='% di compilazioni',range=[0, 100])
    st.plotly_chart(f)

    f2 = px.histogram(selectedAnagrafiche, x='anni_di_attivita', title='Anni Di Attività',histnorm='percent')
    f2.update_xaxes(title='Anni Di Attività',range=[0, 110])
    f2.update_yaxes(title='% di casi',range=[0, 20])
    st.plotly_chart(f2)

    f = px.histogram(selectedAnagrafiche, x='anagrafica.ind_assoc', title='Individuo vs Associazione',histnorm='percent')
    f.update_xaxes(title='Ind vs Assoc')
    f.update_yaxes(title='% di casi',range=[0, 80])
    st.plotly_chart(f)

    f = px.histogram(selectedAnagrafiche, x='anagrafica.formaGiuridica', title='Forma Giuridica',histnorm='percent')
    f.update_xaxes(title='Forma Giuridica')
    f.update_yaxes(title='% di casi',range=[0, 30])
    st.plotly_chart(f)

    f = px.histogram(selectedAnagrafiche, x='anagrafica.titoloStudio', title='Titolo di Studio',histnorm='percent')
    f.update_xaxes(title='Titolo di Studio')
    f.update_yaxes(title='% di casi',range=[0, 150])
    st.plotly_chart(f)

    f = px.histogram(selectedAnagrafiche, x='anagrafica.altro_lavoro', title='Altra Occupazione',histnorm='percent')
    f.update_xaxes(title='Altra Occupazione')
    f.update_yaxes(title='% di casi')
    st.plotly_chart(f)

    anagraficheTempoReddito(selectedAnagrafiche)

    dipVsMod(selectedAnagrafiche)

  def anagraficaWordCloud(df,fullFiltered,attivitaFilters):
    textCols = list(filter(lambda s: re.search('riflessioni',s), df.columns));
    st.pyplot(buildWordCloud(buildAttivitaQuery(attivitaFilters),fullFiltered,textCols))


  def buildWordCloud(AttivitaQuery,fullFiltered,textCols):
    wordcloudColumns =  list(filter(lambda s: re.search('['+AttivitaQuery+']',s), textCols));

    wordCloud = fullFiltered[wordcloudColumns].fillna('').copy()
    wordCloud['final'] = '';
    for k in wordcloudColumns:
      wordCloud['final'] = wordCloud['final'] + ' ' + wordCloud[k].map(str)

    allwords = wordCloud['final'].str.cat(sep=' ')

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
  
  """
  ## Anagrafica
  """
  anagrafiche = df.loc[:,['anagrafica.ind_assoc', 'anagrafica.provincia','anagrafica.data',
        'anagrafica.formaGiuridica', 'anagrafica.titoloStudio','anagrafica.mobilitati','anagrafica.dipendenti',
        'anagrafica.altro_lavoro', 'anagrafica.comune','anni_di_attivita','anagrafica.perc_reddito','anagrafica.perc_tempo',
        'organizzazione', 'produzione', 'supporto', 'promozione', 'educazione']]

  addFiltersToSidebar()
  selectedAnagrafiche = getSelectedAnagrafiche(anagrafiche, attivitaFilters, provinciaFilters, indAssocFilters)
  showAnagrafica(selectedAnagrafiche)
  anagraficaCharts(selectedAnagrafiche)
  """
  ## WORD CLOUD
  """
  fullFiltered = getFullFiltered(df, attivitaFilters, provinciaFilters, indAssocFilters);
  anagraficaWordCloud(df,fullFiltered,attivitaFilters)

def main_attivita(df):
  global attivitaFilters,provinciaFilters,indAssocFilters
  addFiltersToSidebar()
  """
  # Attività
  """
  normalizedActivities = deepcopy(getActivities(df));
  normalizedActivities
  fullFiltered = getActivitiesFullFiltered(normalizedActivities, attivitaFilters, provinciaFilters, indAssocFilters);
  """
  ## COMPOSIZIONE DELLE ATTIVITA'
  """
  st.plotly_chart(plotSezioniAttivita(fullFiltered),use_container_width=True)
  attivitaNonCompilate(fullFiltered)
  """
  ## REDDITO PER SETTORE
  """
  st.plotly_chart(plotRedditiPerSettore(fullFiltered))
  """
  ## REDDITO PER ATTIVITA'
  """
  st.plotly_chart(plotRedditiPerAttivita(fullFiltered),use_container_width=True)
  """
  ## FONTI PER SETTORE
  """
  st.plotly_chart(plotFontiPerSettore(fullFiltered))
  """
  ## FONTI PER ATTIVITA'
  """
  st.plotly_chart(plotFontiPerAttivita(fullFiltered),use_container_width=True)
  """
  ## ASPETTATIVE PER SETTORE
  """
  st.plotly_chart(plotAspettativePerSettore(fullFiltered))
  """
  ## ASPETTATIVE PER ATTIVITA'
  """
  st.plotly_chart(plotAspettativePerAttivita(fullFiltered))


 
  ## activities data

#@st.cache
def attivitaNonCompilate(fullFiltered):
  compilate = getCountsByAct(fullFiltered)
  compilate

      
@st.cache 
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

def getCountsByAct(fullFiltered):
  ret = fullFiltered.groupby(['role','activity'])['anagrafica.provincia'].count().reset_index()
  ret['compilazioni'] = ret['anagrafica.provincia']
  return ret.drop('anagrafica.provincia', axis=1)

def plotSezioniAttivita(fullFiltered):
  sezAtt = getCountsByAct(fullFiltered)
  tot = sezAtt['compilazioni'].sum()
  sezAtt['compilazioni_perc'] = round(sezAtt['compilazioni'] * 100 * 100/ tot) / 100
  fig = px.sunburst(sezAtt, path=['role', 'activity'], values='compilazioni_perc',
                  #color='lifeExp', 
                 
                  # textinfo='label+percent entry',
                  color_continuous_scale='RdBu')
  fig.update_traces(textinfo="label+percent entry")
  #st.text("user_defined hovertemplate:" + fig.data[0].hovertemplate)
  fig.update_traces( hovertemplate = 'area=%{label}<br>compilazioni=%{value}%')
  return fig
  # labels = list(activities.keys())
  # sezioni = fullFiltered[labels].sum()
  # fig = go.Figure(data=[go.Pie(labels=labels, values=sezioni)])
  # return fig

# #st.plotly_chart(plotSezioni(),use_container_width=True)
def myIsNa(val):
  return (str(val) == 'nan')

def getCostiFromActivityRow(row, k):
  keys = [n for n in row.keys() if n.startswith(k+'.costi')]
  return {z.replace(k+'.costi.',''): row[z] for z in keys}

def getMisureFromActivityRow(row, k):
  keys = [n for n in row.keys() if n.startswith(k+'.misure')]
  return {z.replace(k+'.misure.',''): row[z] for z in keys}

def plotRedditiPerSettore(df):
  reddit = df.groupby(['role','activity','reddito'])['anagrafica.provincia'].count().reset_index()
  tot = reddit['anagrafica.provincia'].sum()
  reddit['perc'] = reddit['anagrafica.provincia'] / tot
  fig = px.histogram(reddit, x="reddito", y="perc", color="role", 
  category_orders={"reddito":["reddito_0_10000","reddito_10001_20000","reddito_20001_30000","reddito_30001_50000","reddito_50000_"]},
  barmode="group")
  return fig

def plotFontiPerSettore(df):
  fontiCols = list(filter(lambda s: re.search('fonti\..*',s), df.columns));
  fonti = df.groupby(['role'])[fontiCols].mean().reset_index()
  #fonti
  fig = px.bar(fonti, x="role", y=fontiCols  )
  return fig
 

def plotRedditiPerAttivita(df):
 
  reddit2 = df.groupby(['role','activity','reddito']).agg({"anagrafica.provincia" : "count"}).groupby(level=1).apply(lambda x: 100*x/x.sum()).reset_index()
  reddit2['perc'] = reddit2['anagrafica.provincia'] 
  #reddit2


  fig2 = px.histogram(reddit2, x="reddito", y="perc", color="role", facet_col="activity",facet_col_wrap=4, facet_row_spacing=0.05,facet_col_spacing=0.02,
  category_orders={"reddito":["reddito_0_10000","reddito_10001_20000","reddito_20001_30000","reddito_30001_50000","reddito_50000_"]},
  #barmode="group"
  height=2000
  )
  fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].replace(' / ',",")[0:15]+'...'))
  # dfz = px.data.tips()
  # dfz
  # fig2 = px.histogram(dfz, x="total_bill", y="tip", color="sex", facet_row="time", facet_col="day",
  #      category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

  return fig2
  # st.pyplot(g)

def plotFontiPerAttivita(df):
  fontiCols = list(filter(lambda s: re.search('fonti\..*',s), df.columns));
  # labs = {}
  # for c in fontiCols:
  #   labs[c] = c.replace('fonti.','')

  fonti = df.groupby(['role','activity'])[fontiCols].mean().reset_index().melt(id_vars=['role','activity'], value_vars=fontiCols)
  fonti['variable'] = fonti['variable'].str.replace('fonti.','')
  fonti['variable'] = fonti['variable'].str.replace('_',' ')
  fig2 = px.histogram(fonti, x="variable", y="value", color="variable", facet_col="activity",facet_col_wrap=4, 
  labels={'variable':'Fonte'}, 
  facet_row_spacing=0.05,facet_col_spacing=0.02,
  #category_orders={"reddito":["reddito_0_10000","reddito_10001_20000","reddito_20001_30000","reddito_30001_50000","reddito_50000_"]},
  #barmode="group"
  height=2000
  )
  #fig2.update_layout(showlegend=False)
  fig2.update_layout(legend=dict(
    yanchor="top",
    y=-0.1,
    xanchor="left",
    x=0
  ))
  for axis in fig2.layout:
    if type(fig2.layout[axis]) == go.layout.YAxis:
        fig2.layout[axis].title.text = ''
    if type(fig2.layout[axis]) == go.layout.XAxis:
        fig2.layout[axis].title.text = ''

  fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].replace(' / ',",")[0:15]+'...'))
  
  return fig2

def plotAspettativePerSettore(df):
  reddit = df.groupby(['role','activity','prospettive'])['anagrafica.provincia'].count().reset_index()
  tot = reddit['anagrafica.provincia'].sum()
  reddit['perc'] = reddit['anagrafica.provincia'] / tot
  fig = px.histogram(reddit, x="prospettive", y="perc", color="role", 
  category_orders={"prospettive":["catastrofe","paludosa","tristina","tengo_botta","ok","alla_grande"]},
  barmode="group")
  return fig

def plotAspettativePerAttivita(df):
 
  reddit2 = df.groupby(['role','activity','prospettive']).agg({"anagrafica.provincia" : "count"}).groupby(level=1).apply(lambda x: 100*x/x.sum()).reset_index()
  reddit2['perc'] = reddit2['anagrafica.provincia'] 
  #reddit2


  fig2 = px.histogram(reddit2, x="prospettive", y="perc", color="role", facet_col="activity",facet_col_wrap=4, facet_row_spacing=0.05,facet_col_spacing=0.02,
  category_orders={"prospettive":["catastrofe","paludosa","tristina","tengo_botta","ok","alla_grande"]},
  #barmode="group"
  height=2000
  )
  fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].replace(' / ',",")[0:15]+'...'))
  # dfz = px.data.tips()
  # dfz
  # fig2 = px.histogram(dfz, x="total_bill", y="tip", color="sex", facet_row="time", facet_col="day",
  #      category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

  return fig2

main()
