---
layout: post
title: "Building TL-DR: an online <br/>text summariser"
subtitle: "An extractive approach"
date: 2020-03-20
image: "textsum.png"
color: "#3366cc"
summary: "Adapting latent sentiment analysis for automatic text summarisation. Or how I discovered machine learning doesn't solve everything."
---

![TextSummariserDemo](/assets/images/textsumdemo.gif)

This is an [online Text Summariser](http://www.thetextsummarizer.com/) application I built recently. The app's website was built with [Flask](https://flask.palletsprojects.com/en/1.1.x/), [Bootstrap](https://getbootstrap.com/), and hosted on [AWS EC2](https://aws.amazon.com/ec2/).

## Intro
I have always been fascinated by the Natural Language Processing side of machine learning, partly because I like to read, so I am very curious about how an AI interprets languages, create original texts or converses.

I was very excited when Georgia Tech (where I am doing a part-time online Master in Data Science) announced a new Deep Learning course in collaboration with Facebook. While I am waiting for that course to be available for online students, I decided to work on a project myself. 

## Some thoughts
Text summarization is a difficult problem for many reasons. Despite the resurgence and advancements in deep learning techniques, text summarization is still a big challenge, because:

1. **Variable-length input and output**: a good summarizer must be able to handle input with varying length, and adjust the output's length as necessary to capture the essential information in the summary. This unpredictability makes the problem exponentially more challenging.

2. **Training time and computation**: This is a direct ramification of point 1. A more sophisticated recurrent neural network model is usually used to track important information regardless of whether it appears at the beginning or the end of an article. That is, you cannot just extract a collection of words from the text (bag of words), but have to pay attention to the context within the flow of the text. It requires a lot of memory to train, and I constantly ran out of memory while trying to do it.

3. **Domain dependency**: Unfortunately, a universal deep learning text summariser will require a lot of training data from many different domains. In reality, most models are trained on data from a particular source, such as Google news, or Amazon product reviews, making them biased towards those domains. It's also impractical for a regular person to have access to a large body of textual data in the required format.

After some experiments and tinkering with deep learning models, I took a step back from the rabbit hole and pursed an extractive approach (does not require training a deep learning model) for the summariser.

I wrote more about the algorithm in the About section on the [app's website](http://www.thetextsummarizer.com/about).

## Reference
I based my algorithm on the following papers.  
*[1] Allahyari, M, Pouriyeh, S. et al (2017). Text Summarization Techniques: A Brief Survey.*  
*[2] Steinberger, J, Poesio, M. et al (2007). Two uses of anaphora resolution in summarization.*