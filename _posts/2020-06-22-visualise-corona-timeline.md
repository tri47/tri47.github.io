---
layout: post
title: "My CoVid-19 dashboard"
subtitle: ""
date: 2020-06-22
image: "coronatimeline.png"
color: "#663399"
width: "1000px"
---
**This post contains interactive charts which are best viewed on a large screen.**

<div class='tableauPlaceholder' id='viz1586685144652' style='position: relative; display: block; margin-bottom: 30px;'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CoVtrends&#47;CoV-19-story&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='CoVtrends&#47;CoV-19-story' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CoVtrends&#47;CoV-19-story&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                

<script type='text/javascript'>                    var divElement = document.getElementById('viz1586685144652');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1000px';vizElement.style.height='827px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);             
   </script>

During the early phase of the Covid-19 outbreak, the media focused mostly on the number of new cases and deaths, and not much on recovery. It prompted me to go digging to understand the severity of the pandemic.

I found [this dataset](https://github.com/CSSEGISandData/COVID-19) maintained by Johns Hopkins University and made a [visualisation](https://public.tableau.com/profile/tri1422#!/vizhome/CoVtrends/CoV-19-story?publish=yes) of this information in Tableau, with a focus on the reactions by the Australian government during this early phase of the pandemic. 

There were certainly questionable decisions, e.g., incoming flights from Italy were only banned after the Grand Prix was canceled, despite the number of cases in Italy already exceeded 11,000 then. Travelers from Italy were banned after South Korea and Iran, which only had 4,300 and 1,000 cases respectively at the time they were banned. 

It was also interesting that an economic stimulus package was announced in mid February, one month before those by the US and UK, and also before any significant border restrictions were enacted. Some suggest that the government had a better foresight of the economic downturn than our Western counterparts.

Link to the [Python code](https://github.com/tri47/CoVid-19-trends) I used to preprocess the data. 

Link to the [visualisation.](https://public.tableau.com/profile/tri1422#!/vizhome/CoVtrends/CoV-19-story?publish=yes) 