---
layout: post
title: "Disease and prevention in 500 US cities: an interactive visualisation"
subtitle: "A study of the relationship between chronic diseases, risk factors, and preventions in 500 US cities"
date: 2020-02-10
image: "500cities.png"
color: "#663399"
htmlwidgets: TRUE
---

![](/assets/images/500cities_screenshot.png)

This is a team project that I did as part of my Master's at Georgia Tech.

We analysed the data from CDC's [500 Cities project](https://www.cdc.gov/500cities/index.htm) which provides census-track (a small geographical area defined for census-taking purpose) level estimates of diseases' prevalence, risk factors such as drinking, lack of sleep, etc. and preventative measures. The diseases are mostly non-communicable diseases such as heart conditions or arthritis. Some preventions covered by the data include cholesterol screening, mammography, and routine medical checkup.

We picked this dataset for our project because it was the first time that a public health dataset at such a granular level was made available to the public. It also allowed us to aggregate them into cities' and states' statistics, which would enable some interesting comparisons.

We used Scikit-learn to do the majority of the analysis. We distilled it to 5 most relevant factors for each disease, which we then visualise it Tableau, along with other findings.

The visualisation can be viewed [here](https://public.tableau.com/profile/tri1422#!/vizhome/VizExamples2019_1_2_15543742823730/Story). For the best result, view this on a large screen.

Link to view the [poster](/assets/files/500cities.pdf).
<embed style="width:100%; height:850px" src="/assets/files/500cities.pdf" type="application/pdf" />

