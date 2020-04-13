---
layout: post
title: "Scraping an Instagram public profile - 2020"
subtitle: "(ideally your profile to avoid ethical dilemma)"
date: 2020-03-05
image: "instagram.png"
color: "#3366cc"
---
Instagram has become an important platform for restaurants and cafes to run advertisements and connect with their patrons. I was working on improving custoemr interactions on Instagram for a restaurant, and naturally had to extract and anlyse past performance somehow.

As a result, I developed some to scrape interactions data on Instgram posts and deployed it as a microservice on AWS Lambda to send a regular report by email to me. The code can be found [here](https://github.com/tri47/instaScraper).

This post's goal is to explain how to find the information required to run the code, as a lot of the tutorials on the web are outdated due to recent Instagram's updates.

## Can't I just scrape the page source?
You can. The problem is the infinite scroll feature, which returns the new posts via Javascript and won't update the page source, so you will only be able to retrieve a maximum of 12 posts from the page source alone. 

The following steps will help us get around that.

## The steps 
Firstly, Instagram uses graphQL API (at least) for profile pages. This process will be different for a hashtag page as it uses a different query structure.

To get the query structure, open a public profile page in Firefox. Right click and select Inspect.

Next, select Network, you will find a list of queries that were sent to Istagram's server. Scroll on the profile page to retrieve a few more posts while keeping the Network window open.

![screenshot](/assets/images/scrape1.png)

You should see a new GET query now. Click on that to open a new window. 

![screenshot](/assets/images/scrape2.png)

Click on XHR, and copy the request URL. If you paste this to your browser, you will receive the raw respond from Instagram, you can explore the JSON structure here.

Let's inspect the query string itself.

> https://www.instagram.com/graphql/query/?query_hash=d496eb541e5c7892ezcaee&  
>variables={"id":"25025320","first":12,  
>"after":"QVFFUMmlKZDla0VaOHhEZ1VreUJKNmpJNA=="}

I took the liberty to shorten the random strings a bit to make it more digestible. There are three important parameters here:

1. query_hash: a good explaination of it is [here](https://stackoverflow.com/questions/54238696/what-is-query-hash-in-instagram). For our purpose, we can simple make note of it as one of our inputs.
2. id: a unique identifier for the profile page you're querying.
3. after: an identifier for the last post in the previous query, basically it tells Instgram to find the next post after that post ID.

And that's it. Those are all the elements you need to provide the [Python script]() to scrape the data. I have extracted the post URL, description, likes and date.

This has helped me tremedously to be able to analyse pattern in user interactions, useful hashtags, etc. 

## End notes
It's a shame that I had to jump through all these hoops to get the data that should be readily available for users who created those in the first place.

It just shows how crucial the control of data is, as there is so much insights and values you can derive from it.

![pancake](/assets/images/pancake.png)






