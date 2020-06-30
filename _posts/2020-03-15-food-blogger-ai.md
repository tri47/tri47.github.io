---
layout: post
title: "Training an AI to write food blogs"
subtitle: "What could possibly go wrong?"
date: 2020-03-15
image: "foodblog.png"
color: "#89306f"
summary: Adapting OpenAI GPT-2 language model to write food blogs.
---
> Happy Sunday Melbourne! I’m here to spoon a delicious home-made Guacamole and Butternut Smoked Salmon.

> Fried Croissant with sourdoughnuts and creamy pancake batter. My favourite breakfast in New York City and @eriqulotopos some stupid puns.

> Available for breakfast, lunch and dinner just like Puget Sound’s Puget Sound Cafe. The buns at Somerbay South Burger & Chips are also good, especially the coconut and dill flavour.

![R2D2](/assets/images/r2d2.png)

## Intro
I mentioned in one of my last posts that I was looking at improving customer interactions on Instagram for a restaurant in Melbourne, and was experimenting with writing more engaging daily posts. Naturally, I couldn't help but wonder if I could automate the writing process (typical lazy programmer!).

As a result, this robot was born. This is my attempt to train OpenAI's GPT-2 language model to write snippets for an (Instagram) food blog. This model made waves earlier last year when OpenAI released it, but held back on the full version because it was deemed too [humanlike and dangerous](https://techcrunch.com/2019/02/17/openai-text-generator-dangerous/). OpenAI subsequently released the full version in Nov 2019, and new applications proliferated. They included models that create poems or D&D dialogues.

## The training data
I needed to provide the model with a lot of samples from actual food blogs so that it can learn the tone, jargons, and structure of a social media post on foods and dining. I repurposed the code I used to scrape an Instagram profile page to extract sample posts. I ended up with about 15,000 lines of text to feed the model.

## Training the model
I wrote more about how to train and deploy the model in my [Github repo](https://github.com/tri47/food-blog-AI-writer). It's not super polished as it was more for my reference and experiments.

## Deploying to production
To deploy the model to production, I used [Cortex](https://github.com/cortexlabs/cortex), an open-source machine learning deployment platform to deploy the trained model to AWS Lambda as an API web service. This gives me the flexibility of wrapping it around a web app later if I so desire.

## The results
### Without prompts

> The combination of grilled beef and lime, fresh lime slaw, mustard seeds, zucchini flowers, fresh lime, guacamole, leek and kim chi soft serve. Perfect epic meal #meturbalpourri #tcbd #sydneybars #sydneyeats #brewedchampagne #sydneyfood #sausageroll #menwithcuisines #sydneyfoodblogger #sydneyfoodblog #petecbarracuda

> A Danish outlandish pop-up pub hidden under the bridge, bar or bridge northwest of Sydney. When dining out at this bar/bistro, you know you’ll end up dining off the menu. Don’t judge by the menu when you can order a greasy caffelato pizza from the menu. I shared a handful of the toppings the other night so if you haven’t had them yet, I’ll share them in a later post #saint_daneeagle #fcba #barsons #bistro #edoublewine #publife #peninsulife #barbancake #borrancafe #sydneybars #sydneyeats #sydneyfood #steak #skewers #fries #ncafe #foodlovers

> One of my favourite places for good food photography is the The Kingfish Cafe in Thornbush. An eclectic bar/restaurant just outside the mainstream Sydney food scene. An absolute must visit place for foodists and locals alike. Try as this cafe might not look a view from the windows the food on display looks absolutely stunning!  #borrusco1 #sutherland #aldistillery #sydneyfood  #cocktails #invited #chocolate #sunflowerapples #sydneyfoodshare #scones #blacksesame #foodpics


### With prompts
#### Prompt: Burger

> I had a blast on Tuesday morning devouring the classic Pork and Prawn Toast from the very cool @theburgercollectiveapp.

> What an amazing food experience and @theburgencecollectiveau
My favorite fried chicken sandwich from the @sydneyburgercollectiveau is finally here!!
#burger #burgersofmelbourne #melbourneburgers #sydneyburgers #sydneyfood #

> A beautiful breakfast dish for the weather. Don’t get me started on the potato fried chicken in blue jeans @bishopsgatemarket  #melbournefoodblogger #melhotornot

The "burger" prompt makes the AI think "fried chicken"! This is largely because the training data does not come with a 'header' like 'vegan chicken burger' followed by a review/description of the dish. Therefore, the model does not complete the rest of the prompt very well. This is a recurring theme I have encountered during my journey learning Data Science. Most of the time the hardest part is sourcing required training data in the right format. Unsurprisingly, data is rightly considered the new oil by many corporations.

### Endnotes
The model performs pretty well. I like how it can pick up the tone and vocabulary from the training data with ease. It also picks up on how to write relevant hashtags. It is by no means a replacement for a human writer, but it has the potential to be a powerful virtual assistant who can help provide the starter lines/prompts when a writer is stuck. It will make an excellent prompt generator for writing practice.

The future is bright, right?
