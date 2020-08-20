---
layout: post
title: "Scraping an Instagram public profile"
subtitle: "Because they won't give you your data that easily"
date: 2020-03-05
image: "instagram.png"
color: "#3366cc"
summary: A guide on Instagram scraping and dealing with infinite scroll.
tags: data-mining social-media python tutorial
---
![](/assets/images/pancake1.png)
<br/> <br/>
Instagram has become an important platform for restaurants and cafes to run advertisements and connect with their patrons. I was working on improving customer interactions on Instagram for a restaurant in Melbourne. Naturally, I wanted to analyse past performance. Unfortunately, I couldn't find a (free) way to extract the information easily, so I resorted to web scraping.

I deployed the script as a microservice on AWS Lambda to send a regular report by email to me. The code can be found [here](https://github.com/tri47/instaScraper).

This post is to explain how to find the information required to run the code, as a lot of the tutorials on the web are outdated due to recent Instagram updates.

## Can't I just scrape the page source?
You can. The problem is the infinite scroll feature.

When you open an Instagram profile page, you get 12 posts by default, then you have to scroll down to load new ones. This process returns the new posts via Javascript and won't update the page source, so you will only end up with the original 12 posts from the page source. 

This also rules out using Selenium to do the scrolling for you (yeah I tried that), as the page source still won't contain new posts.

The following steps will help us get around that.

## Getting all those ID's! 
Instagram uses graphQL API for profile pages. The following does not apply to a hashtag page as it uses a different query structure.

To get the query structure, open a public profile page (I use Firefox). Right-click and select Inspect.

Next, select Network, you will find a list of queries that were sent to the Instagram's server. Scroll on the profile page to retrieve the next 12 posts while keeping the Network window open.

![screenshot](/assets/images/scrape1.png)

You should see a new GET query now. Click on that to open a new window. 

![screenshot](/assets/images/scrape2.png)

Click on XHR, and copy the "Request URL". If you paste this to your browser, you will receive the raw response from Instagram, you can explore the JSON structure here to identify where the data you want is stored.

Let's inspect the query string.

> https://www.instagram.com/graphql/query/?query_hash=d496eb541e5c7892ezcaee&  
>variables={"id":"25025320","first":12,
"after":"QVFFUMmlKZDVreUJKNmpJNA=="}

I shortened the random strings a bit to make it more digestible. There are a few important parameters here:

1. query_hash (d496eb...): for an explanation, read [here](https://stackoverflow.com/questions/54238696/what-is-query-hash-in-instagram). Make note of it as one of our inputs.
2. id (25025320): a unique identifier for the profile page you're querying.
3. first: the number of posts to retrieve next, default to 12.
4. after: an identifier for the last post in the previous query, i.e. post number 12 when you first open the profile page. It tells Instagram to find the next posts after that post ID.

Those are all the elements you need to provide the [Python script](https://github.com/tri47/instaScraper) to scrape the data. I have extracted the post URL, description, likes and date.

This helped me get all the data to analyse patterns in user interactions, useful hashtags, etc.

## Endnotes
I am yet to look at the API provided by Instagram for developers. I am guessing it will make things easier. It's a bit of a shame that we have to jump through all these hoops to get the data that should be readily available for users who created those in the first place. There are a lot of debates going on about the right to our data, and it undoubtedly will become a very crucial decision for us to make as a society.

![](/assets/images/vienna_meal.jpg)

 <p class= 'image-caption'>A meal on a hiking trip in Puchberg am Schneeberg, Austria. </p>





