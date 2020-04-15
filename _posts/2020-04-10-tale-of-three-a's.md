---
layout: post
title: "A tale of three A's: Australian stock market vs Gold and Silver"
subtitle: "An analysis of ASX 200 performance versus Gold and Silver over the last 30 years"
date: 2020-04-10
image: "au.png"
color: "#4cadc7"
htmlwidgets: TRUE
---
**\*NOTE: This post contains interactive charts which are best viewed on a large screen.**

This post was originally an [ipython notebook](https://github.com/tri47/ASX_vs_Metals/blob/master/analysis.ipynb) I created to explore the relative movement of ASX stocks (using ASX 200 index) vs Gold (Au) and silver (Ag) over the last 30 years. Data was collated from:
- Exchange rate: rba.gov.au
- ASX, Gold and Silver price: marketindex.com.au  
- Recent Gold and Silver price: LBMA.org.uk  

### ASX 200 index stock price to Gold and Silver ratio
Precious metals are seen as safe havens when there's high inflation or a downturn in the economy.  
In the graph below, we can see the relative price between ASX stocks and gold/silver.  
Gold's relative value to stocks peaked in 2009 during the GFC.  
At the point of writing (April 2020), Gold's relative value to stocks already exceeded that during the GFC. This suggests there may not be a lot of room for Gold to gain values relative to stocks.  
Silver on the other hand still has not reached its peak in 2011. This might suggest a buying opportunity.
<div>
    <a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/1/?share_key=roU3OOFzFGjFgivzCHw7RF" target="_blank" title="asx_gold_silver_ratio" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/1.png?share_key=roU3OOFzFGjFgivzCHw7RF" alt="asx_gold_silver_ratio" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:1" sharekey-plotly="roU3OOFzFGjFgivzCHw7RF" src="https://plotly.com/embed.js" async></script>
</div>
<br/>

### Cumulative return of the three asset classes from 1992
Over the last ~30 years, gold saw a 450% return, compared to 350% for Silver and 300% for Australian stocks (prior to the Corona's downturn).  
This does not take into account devidends from stocks and holding costs for Gold and Silver.

<div>
    <a a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/3/?share_key=L5r806qkmRwWvJg47Hi6ya" target="_blank" title="acc_return_1992" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/3.png?share_key=L5r806qkmRwWvJg47Hi6ya" alt="acc_return_1992" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:3" sharekey-plotly="L5r806qkmRwWvJg47Hi6ya" src="https://plotly.com/embed.js" async></script>
</div>
<br/>

### Cumulative return of the three asset classes from 2000
Zooming into the last 20 years, silver significantly outperforms ASX and Gold in terms of growth as seen below. Next, we take a deeper look at each asset class.
<div>
    <a a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/5/?share_key=rRQR38y0qhkUnr4sJKTHP6" target="_blank" title="acc_return_2000" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/5.png?share_key=rRQR38y0qhkUnr4sJKTHP6" alt="acc_return_2000" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:5" sharekey-plotly="rRQR38y0qhkUnr4sJKTHP6" src="https://plotly.com/embed.js" async></script>
</div>
<br/>

### ASX price movement
A commonly used indicator for a stock's price movement is the Bollinger bands, where we show the running average plus/minus 2 standard deviations. The actual stock price is also shown, and it is theorised that a stock price will tend to go back to the running average after it exceeds the upper or the lower band (deviates more than 2 standard deviations).  
As shown below, ASX stock price has already dropped below the lower band due to the recent Corona bear market.
<div>
    <a a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/7/?share_key=zqr4dXDmOxXvUnbMAndO3V" target="_blank" title="ASX_bollinger" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/7.png?share_key=zqr4dXDmOxXvUnbMAndO3V" alt="ASX_bollinger" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:7" sharekey-plotly="zqr4dXDmOxXvUnbMAndO3V" src="https://plotly.com/embed.js" async></script>
</div>
<br/>

### Gold price movement
As can be expected, when the stock price drops, gold picks up, and it already exceeds the upper Bollinger band as seen below.
<div>
    <a a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/9/?share_key=mXEMw2wmQPhagjVBT5Hvkx" target="_blank" title="Gold_bollinger" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/9.png?share_key=mXEMw2wmQPhagjVBT5Hvkx" alt="Gold_bollinger" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:9" sharekey-plotly="mXEMw2wmQPhagjVBT5Hvkx" src="https://plotly.com/embed.js" async></script>
</div>
<br/>

### Silver price movement
Silver price on the other hand is still very close to the moving average.  
If the past is an indicator, during the GFC, we saw that the peak of Silver's relative value to stocks lagged the Gold's peak by 6 months.  
Therefore, it appears that Silver has not caught up to the growth in precious metal's price due to the recent Corona downturn, and is likely to do so in the near future. 
<div>
    <a a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/11/?share_key=bAKdoju2hVHQvkllTSLykd" target="_blank" title="Silver_bollinger" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/11.png?share_key=bAKdoju2hVHQvkllTSLykd" alt="Silver_bollinger" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:11" sharekey-plotly="bAKdoju2hVHQvkllTSLykd" src="https://plotly.com/embed.js" async></script>
</div>
<br/>

### Endnotes
We have only looked at the asset classes from a technical perspective and is therefore not a complete picture. But the trend indicates that the best time to buy into Gold for a quick return might have already past, and the next option is Silver.  

It would be interesting to conduct the same analysis for properties as well, which I might explore in another post.

