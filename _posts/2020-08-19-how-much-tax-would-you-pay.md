---
layout: post
title: "Do we pay too much tax in Australia?"
subtitle: "Maybe."
date: 2020-08-19
image: "tax.png"
color: "#4cadc7"
htmlwidgets: TRUE
summary: Calculate how much tax you would pay if you were living in the UK, Belgium or Japan.
tags: finance tableau analysis
---

![](/assets/images/tax_time.jpg)

If you're like me, then you may have bemoaned how much income tax you have to pay.

To make myself feel better, I create a Tableau dashboard to show how much tax I would pay if I were living in another country.

Adjust your annual income in the box below (click on the graph if you're reading on the phone to hide the slider).

<div class='tableauPlaceholder' id='viz1597900964307' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ta&#47;Taxaroundtheworld&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Taxaroundtheworld&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ta&#47;Taxaroundtheworld&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /><param name='filter' value='publish=yes' /></object></div>                

<script type='text/javascript'>                    
var divElement = document.getElementById('viz1597900964307');                    
var vizElement = divElement.getElementsByTagName('object')[0];                    
if ( divElement.offsetWidth > 800 ) { vizElement.style.width='550px';vizElement.style.height='527px';} 
else if ( divElement.offsetWidth > 500 ) 
{ vizElement.style.width='550px';vizElement.style.height='600px';} 
else { vizElement.style.width='100%';vizElement.style.height='880px';}                     
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script>


Most countries have a similar tax bracket system like Australia. 

Taxation in Canada and the US is more complicated with federal and state (provincial) taxes. I might add them later when I have time.

UK's tax rates and public health insurance charges are very similar to Australia's. However, its GST is double that of Australia.

Monaco has no tax on individual income, but it has a GST of 19.6%. Russia has a flat tax rate of 13%.

The devil is in the details. Different countries have different ways to apply GST/VAT, which can make your daily expenses or property purchases much more or less expensive. Might be a good topic for another post.

Unlike Australia and the UK where the lowest bracket has a tax rate of zero, Belgium will tax you regardless of how much you earn.

![belgium_tax](/assets/images/belgium_tax.png)

The idea of not charging income tax like Monaco, and focusing on having different rates for goods and services is an interesting one. Necessities like food and medicine may enjoy lower rates while luxurious goods can attract more.

It's a debate that we rarely have in Australia, but a new tax regime - whatever that is - might be something we should discuss given the fast-changing economic environment. 

**Caveats:** 
- I consider individual income tax only.
- Medicare levy and UK National insurance are taken into account.
- It does not taking into account other charges like medicare surcharge or low and mid income tax offset.
- The exchange rates at the time of writing were used.

