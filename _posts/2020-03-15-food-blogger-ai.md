---
layout: post
title: "Training an AI to write food blogs"
subtitle: "What could possibly go wrong?"
date: 2020-03-15
image: "foodblog.png"
color: "#89306f"
---
> Happy Sunday Melbourne! I’m here to spoon a delicious home-made Guacamole and Butternut Smoked Salmon.

> Fried Croissant with sourdoughnuts and creamy pancake batter. My favourite breakfast in New York City and @eriqulotopos some stupid puns.

> Available for breakfast, lunch and dinner just like Puget Sound’s Puget Sound Cafe. The buns at Somerbay South Burger & Chips are also good, especially the coconut and dill flavour.

I trained an AI model (OpenAI's GPT-2 model) to write food blog entries. I hosted the model on AWS Lambda as a serverless microservice (I love AWS!). The above are samples written by the model after training.

![R2D2](/assets/images/r2d2.png)

## Intro
I mentioned in one of my last posts I was looking at improving customer interactions on Istagram for a restaurant, and was experimenting with writing more engaging daily posts. Naturally, I couldn't help but wonder if I could automate the writing process :)).

As a result, this robot was born. This is my attempt to train OpenAI's GPT-2 language model to write snippets for a (instagram) food blog. OpenAI made waves earlier last year when they released this model, which was deemed [too dangerous](https://techcrunch.com/2019/02/17/openai-text-generator-dangerous/) for the full version to be released. It subsequently released the full version in Nov 2019, and saw many new applications.
This repository is used to outline the steps I took to tune the model to write food blog snippets and deploy to AWS.


## The training data
I repurposed the code I used to scrape an Instagram profile page to extract sample food blog posts from Instagram as raw data. I ended up with about 15,000 lines of text.

## Training the model
I wrote more about how to train and deploy the model in the [Github repo](https://github.com/tri47/food-blog-AI-writer). It's not super polished as it was more for my reference and experiments.

## Sample output
### Without prompts

> The combination of grilled beef and lime, fresh lime slaw, mustard seeds, zucchini flowers, fresh lime, guacamole, leek and kim chi soft serve. Perfect epic meal #meturbalpourri #tcbd #sydneybars #sydneyeats #brewedchampagne #sydneyfood #sausageroll #menwithcuisines #sydneyfoodblogger #sydneyfoodblog #petecbarracuda

> A Danish outlandish pop-up pub hidden under the bridge, bar or bridge northwest of Sydney. When dining out at this bar/bistro, you know you’ll end up dining off the menu. Don’t judge by the menu when you can order a greasy caffelato pizza from the menu. I shared a handful of the toppings the other night so if you haven’t had them yet, I’ll share them in a later post #saint_daneeagle #fcba #barsons #bistro #edoublewine #publife #peninsulife #barbancake #borrancafe #sydneybars #sydneyeats #sydneyfood #steak #skewers #fries #ncafe #foodlovers

> One of my favourite places for good food photography is the The Kingfish Cafe in Thornbush. An eclectic bar/restaurant just outside the mainstream Sydney food scene. An absolute must visit place for foodists and locals alike. Try as this cafe might not look a view from the windows the food on display looks absolutely stunning!  #borrusco1 #sutherland #aldistillery #sydneyfood #sydneyfoodblogger #cocktails #invited #chocolate #sunflowerapples #sydneyfoodshare #scones #blacksesame #foodpics


### With prompts
#### Prompt: Burger

> I had a blast on Tuesday morning devouring the classic Pork and Prawn Toast from the very cool @theburgercollectiveapp.

> What an amazing food experience and @theburgencecollectiveau
My favorite fried chicken sandwich from the @sydneyburgercollectiveau is finally here!!
#burger #burgersofmelbourne #melbourneburgers #sydneyburgers #sydneyfood #

> A beautiful breakfast dish for the weather. Don’t get me started on the potato fried chicken in blue jeans @bishopsgatemarket #melbournefood #melbournefoodblogger #melhotornot

Somehow the "burger" prompt makes the AI think "fried chicken"! This is because the training data does not come with a 'header' like 'vegan chicken burger' followed by a review/description of the dish. Therefore, the model does not complete the rest of the prompt very well.

### Outro
This is really exciting. I like how the model can pick up the tone and vocabulary from the training data with ease. It is by no means a replacement for a human writer, but I can see it being a powerful assistant who can help provide the starter lines/prompts when a writer is stuck. It will make an excellent prompt generator for writing practice.

The future is bright, right? Right?
