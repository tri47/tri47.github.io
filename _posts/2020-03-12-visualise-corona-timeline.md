---
layout: post
title: "Global CoVid-19 dashboard"
subtitle: ""
date: 2020-03-12
image: "coronatimeline.png"
color: "#3b7c82"
width: "1000px"
summary: "A look into events leading up to the worldwide crisis, statistics for all countries, with a particular focus on Australia and our response."
featured: False
tags: tableau health python
---
**This post contains interactive charts which are best viewed on a large screen.**

<div class='tableauPlaceholder' id='viz1586685144652' style='position: relative; display: block; margin-bottom: 30px;'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CoVtrends&#47;CoV-19-story&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='CoVtrends&#47;CoV-19-story' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CoVtrends&#47;CoV-19-story&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                

<script type='text/javascript'>                    var divElement = document.getElementById('viz1586685144652');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1000px';vizElement.style.height='827px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);             
   </script>

#### The Python code

If you're interested in building this in Tableau yourself, below is my script to download and collate all the csv data files into a format Tableau understands.

The raw files are in wide format (each column represents a day) so the code will perform the trasformation to long format as well.


```python
import numpy as numpy
import pandas as pd

import urllib.request

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
print ("download start!")
filename, headers = urllib.request.urlretrieve(url, filename="Confirmed.csv")
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
filename, headers = urllib.request.urlretrieve(url, filename="Deaths.csv")
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
filename, headers = urllib.request.urlretrieve(url, filename="Recovered.csv")


print ("latest data download complete!")

print ('Processing data for Tableau ....')
dataSets = ['Confirmed', 'Recovered', 'Deaths']
#prefix = 'time_series_19-covid-'
prefix = ''
datas = {}

for data_name in dataSets:
    df = pd.read_csv(prefix + data_name + '.csv')
    max_ix = df.shape[0]
    # add all countries that are not US
    df.loc[max_ix,:]= df[df['Country/Region'] != 'US'].sum(axis=0)
    df.loc[max_ix,'Country/Region'] = 'Rest of the World'
    df.loc[max_ix+1,:]= df[df['Country/Region'] != 'Rest of the World'].sum(axis=0)
    df.loc[max_ix+1,'Country/Region'] = 'Worldwide'
    Cases = pd.melt(df, id_vars= ['Province/State','Country/Region','Lat','Long'],var_name='dateString',value_name=data_name)
    Cases[['month','day','year']] = Cases.dateString.str.split('/',expand=True)
    Cases['date'] = Cases.day + '/' + Cases.month + '/' + Cases.year
    Cases.drop(['month','day','year','dateString'],axis=1,inplace=True)
    datas[data_name] = Cases

# Join all 3 data sets
df_cd = datas['Confirmed'].merge(datas['Deaths'].loc[:,['Country/Region','Province/State','date','Deaths']], on=['Country/Region','Province/State', 'date'], how='left')
df = df_cd.merge(datas['Recovered'].loc[:,['Country/Region','Province/State','date','Recovered']], on=['Country/Region', 'Province/State','date'], how='outer')

df.sort_values(by=['Country/Region','Province/State'],inplace=True)
# Fill null values with previous data
df['Recovered'] = df['Recovered'].ffill()
df.to_csv('test.csv',index=False)
print(df.describe())
print('Lastest date: ')
print(df.loc[:,'date'].unique()[-1])

df.to_csv('CovData.csv',index=False)
```

#### Background
During the early phase of the Covid-19 outbreak, the media focused mostly on the number of new cases and deaths, and not much on recovery. It prompted me to go digging to understand the severity of the pandemic.

I found [this dataset](https://github.com/CSSEGISandData/COVID-19) maintained by Johns Hopkins University and made a [visualisation](https://public.tableau.com/profile/tri1422#!/vizhome/CoVtrends/CoV-19-story?publish=yes) of this information in Tableau, with a focus on the reactions by the Australian government during this early phase of the pandemic. 

There were certainly questionable decisions, e.g., incoming flights from Italy were only banned after the Grand Prix was canceled, despite the number of cases in Italy already exceeded 11,000 then. Travelers from Italy were banned after South Korea and Iran, which only had 4,300 and 1,000 cases respectively at the time they were banned. 

It was also interesting that an economic stimulus package was announced in mid February, one month before those by the US and UK, and also before any significant border restrictions were enacted. Some suggest that the government had a better foresight of the economic downturn than our Western counterparts.

Link to the [Python code](https://github.com/tri47/CoVid-19-trends) I used to preprocess the data. 

Link to the [visualisation.](https://public.tableau.com/profile/tri1422#!/vizhome/CoVtrends/CoV-19-story?publish=yes) 
