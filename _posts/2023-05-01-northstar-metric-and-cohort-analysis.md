---
layout: post
title: "Using cohort analysis to fix misleading product metrics"
subtitle: "Why mixing users in your North Star (product) metric can be a bad idea"
date: 2023-05-02
image: "cohort.png"
color: "#3d61ac"
htmlwidgets: TRUE
summary: "Why mixing users in your North Star (product) metric can be a bad idea"
tags: discussion analysis
---


![](/assets/images/cohort/cohort_top.png)

#### The Boring North Star Metric

Do you often turn up to your monthly product/business review meetings, check your product dashboard, and never find anything new with those charts?

Many product teams have North Star metrics that look something like the chart below - kinda random most of the time.

If you are lucky, the Monthly Active Users may go up in the same month you release that new widget, and everyone is celebrating! Until it drops again.

And someone (maybe you) will promise to monitor it to see if there's a reversal next month.

![](/assets/images/cohort/important_metric.png)
>  Did we improve the product or not?

The situation I described is very common, and it can lead to a declining interest in being data-driven since the only thing data seem to tell you is that nothing happens.

#### Creating a Good Metric

You most likely have come across these advices on how to track good metrics or KR's:

- Avoid vanity metrics (page views, number of active users, etc.).
- Embrace ratio metrics (MAU, MEU, etc.).
- Embrace a single North Star Metric that your organisation or team can all get behind.

These advices usually culminate into a ratio metric like "% of Logged in Users", "% of Engaged Users" or "% User Making a Repeat Purchase".

While there's some solid foundation for these principles, following them can still lead to misleading and useless metrics.

One of the major culprits of these unhelpful metrics is mixing users from different cohorts in the denominator. 

The standard approach to dealing with it is to use Cohort Analysis, which I will discuss in detail.

#### How North Star Metric Turns Bad: Mixing Users in the Denominator

Imagine you are a product manager of an investment app. You have defined a metric called **Engaged Users %** which is the % of users over all paid users who interact with the app in a meaningful way like buying/selling stocks, i.e. not just logging in.

In Table 1, you can see when users join the app, and whether they are engaged in each month following their signup.

![](/assets/images/cohort/full_table.png)
>  Table 1: an example of calculating **Engaged Users %** metric

The first three rows include users who join in Month 1. The first user engaged with the app in the first three months then stopped, the second user engaged for two months then stopped, and the third user signed up but never engaged.

The subsequent cohorts who joined in months 2, 3, and so on, each also include three users who behave the same way - the first user engaged for three months then stop, the second engaged for two months then stop, etc.

The important thing to be cognizant of in this table is there's no change between the behaviours and types of users in each of the cohorts. We do not show the data beyond month 5, but the usage pattern repeats for all cohorts.

Now, if we try to calculate the **Engaged Users %** metric in this example, you will find that it starts to drop from month 2.

That's the insidious consequence of mixing user cohorts in the denominator. After some time, your new users become old users and become more disengaged (even though they may still have a subscription). This increasing proportion of old users can hurt your overall Engagement Metric.

Potentially, even if you improve your product, the above situation can mask that improvement in your metric and you will wrongly conclude that there was no improvement at all.

Ultimately, the best way to know if your Product change improves is via experimentation. But the reality is not many organisations have the necessary process and capacity to run experiments regularly.

Therefore, we turn to Cohort Analysis to find more meaningful signals from the user engagement data like in Table 1.

#### Cohort Analysis View 1: The Marketing View

Table 2 shows the Engaged Users % for each cohort over time. This is the standard view that comes readily available in many analytics or billing software.

![](/assets/images/cohort/cohort_view_1.png)
>  Table 2: Marketing View: Cohorted **Engaged Users %**

This view is great at showing you the gradual change in the behaviours of user groups who joined at different times. This view is great for checking the impact of marketing and promotion campaigns. What you need to do is to look at the cohort who joined at the time the campaign was run.

Splitting your metric like this is already useful because you're confusing your audience by presenting a combined view for your old and new users.

However, what if you still want to see the performance of your product over time in terms of user engagement? This view does not translate well into that.

#### Cohort Analysis View 2: The Product View

Another way to present the same information is shown in Table 3.

![](/assets/images/cohort/cohort_view_2.png)
>  Table 3: Product View: Cohorted **Engaged Users %**

In this view, each row represents the metric for users of different tenures (months from signup). The first row, for instance, shows us that over five months, **new users** (those who have been on the app for one month or less) always engage with the app at the same rate.

With this view, it is easier to see the engagement change over time for users of similar tenure. An example is shown in Table 4 where a new onboarding process introduced in Month 4 leads to improved engagement for new users. 

![](/assets/images/cohort/cohort_view_2_change.png)
>  Table 4: A change in new (one-month-tenured) users due to a product change in month 4.

While Cohort Analysis is the suggested approach by many data and product practitioners, it has one major flaw - information overload.

Except in very data-matured organisations, an analyst would find it difficult to convince business stakeholders to review these cohort matrices with any regularity.

For that reason, let's look at a few other more digestible options.

#### Alternative Approach 1: Simple Composite Metric (an Okay Approach)

Recall that the problem with mixing all users in the denominator is the changing size of the mixed cohorts: in the beginning, you only have **highly engaged, new users**, while after a few years, many of your users become **less engaged, old users**.

Cohort Analysis can deal with this, but it causes information overload. So another option is to remove the effect of  cohort sizes - by calculating the metric for each cohort separately and taking the (non-weighted) average (Table 5).

![](/assets/images/cohort/average.png)
>  Table 5: Taking a non-weighted average of the Engaged Users % for all cohorts.

As can be seen in the last row, this approach yields a composite metric that reflects the average engagement across all user tenures (from 1 month to 12 months).

The downside of this is that you almost always want to split it out again anyway when you observe a change. Your stakeholders would want to understand if the drop/increase in the metric is attributed to the new or older cohort.

While I do not like this approach a lot, it's still a way to succinctly describe and track a single product metric, in the spirit of North Star metric ideals.

#### Alternative Approach 2: Replace your North Start metric with (shocking!) Two or More Metrics

This is a middle-of-the-road approach. Instead of having one composite metric, or technically having many if you use Cohort Analysis, you can have, say, two metrics for two distinct groups of users.

One metric can be New User Engagement which is the % of engaged users in their first 7 days after signup.

The second metric can be Pre-renewal Engagement which is the % of engaged users 28 days before renewal.

This allows you to separate two distinct groups of users and optimize your onboarding and re-engagement experience.

What I like about this is that each metric also acts as a guardrail metric for the other. If you implement a new user guide button, does it improve **New User Engagement** while annoying old users and driving down **Pre-renewal Engagement**?

In some companies, there might be two Product Teams looking after these two experiences, and it makes sense that they would try to optimise a different metric while ensuring the other is not impacted.

This is currently my preferred approach. There is no information overload while eliminating the problems of mixing different user groups. But, check back with me in a year, I might have changed my mind then!

#### Overall Approach: Holistic Business Views

If there's one thing I learned dealing with misleading metrics, it is that it is always better to review several metrics that give you a holistic business view.

There are three types of Metrics that you should pay attention to:

- Growth Metric: this is especially important during the early stage of your product. You may sacrifice Retention Rate and User Engagement as long as your User Growth outpaces your Attrition.
- Retention Metric: this becomes more important later as your growth slows down and your product becomes more mature.
- Engagement Metric: this is mostly what we talked about in this article. Engagement is a good leading indicator for Retention. It's also something the Product team can better optimize.

Examining these in tandem gives you a more complete picture of your revenue source, leak (via churn), and a leading indicator (engagement) of future performance.

#### Beyond Cohort: Other Confounding Factors

So far we mainly talked about the difference in behaviour between groups of users who joined at different points in time, but it's certainly not the only possible delineation.

For example, due to a recent expansion, you start to have more users from a certain country, but these users have a higher engagement rate than your normal user base.

This leads to a higher **Engaged Users %** number since the expansion, and can make us mistakingly attribute the positive change to something else.

Unfortunately, there are many ways this can happen, and it's very hard to split the users across all possible delineations.

The good news is as long as the demographics of your users are relatively stable, most of the time you only need to care about user age on the platform.

#### Final Remarks

While the idea of a ratio-based North Star metric has helped tech companies identify and track meaningful success metrics, a fundamental understanding of basic analysis techniques like Cohort Analysis is still important. 

I advocate for a more holistic approach where each team, in addition to optimising their dedicated Product Metric, should always keep an eye on the company's Growth and Retention metric, both as guardrail metrics and as auxiliary goals they try to improve.

