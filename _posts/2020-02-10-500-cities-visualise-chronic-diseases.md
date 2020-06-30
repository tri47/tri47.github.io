---
layout: post
title: "Disease and prevention in 500 US cities: an interactive visualisation"
subtitle: "A study of the relationship between chronic diseases, risk factors, and preventions in 500 US cities"
date: 2020-02-10
image: "500cities.png"
color: "#663399"
width: "1000px"
htmlwidgets: TRUE
summary: Analysis and visualisation of diseases, preventative measures, and their correlation.
---
**This post contains interactive charts which are best viewed on a large screen.**

<div class='tableauPlaceholder' id='viz1590373971060' style='position: relative; display: block; margin-bottom: 30px;'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Vi&#47;VizExamples2019_1_2_15543742823730&#47;Story&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='VizExamples2019_1_2_15543742823730&#47;Story' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Vi&#47;VizExamples2019_1_2_15543742823730&#47;Story&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object>

</div>                

This is a team project that I did as part of my Master's at Georgia Tech.

We analysed the data from CDC's [500 Cities project](https://www.cdc.gov/500cities/index.htm) which provides census-track level data.(a small geographical area defined for census-taking purpose). 

The data include estimates of diseases' prevalence, risk factors such as drinking, lack of sleep, etc. and preventative measures. The diseases are mostly non-communicable diseases such as heart conditions or arthritis. Some preventions covered by the data include cholesterol screening, mammography, and routine medical checkup.

We picked this dataset for our project because it was the first time that a public health dataset at such a granular level was made available to the public. It also allowed us to aggregate them into cities' and states' statistics, which would enable some interesting comparisons.

We used Scikit-learn to do the majority of the analysis. We distilled it to 5 most relevant factors for each disease, which we then visualise it Tableau, along with other findings.

Link to view the [visualisation](https://public.tableau.com/profile/tri1422#!/vizhome/VizExamples2019_1_2_15543742823730/Story). 

Link to view the [poster](/assets/files/500cities.pdf).


<script type='text/javascript'>                    var divElement = document.getElementById('viz1590373971060');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1016px';vizElement.style.height='991px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>


<embed style="width:100%; height:850px" src="/assets/files/500cities.pdf" type="application/pdf" />

