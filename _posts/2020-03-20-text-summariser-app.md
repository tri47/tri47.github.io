---
layout: post
title: "Building Tl;dr: a Text Summariser app"
subtitle: "An extractive approach"
date: 2020-03-20
image: "textsum.png"
color: "#663399"
---

![TextSummariserDemo](/assets/images/textsumdemo.gif)

The above is an online Text Summariser App that I built recently.

[Link here](http://www.thetextsummarizer.com/)

The app was built with [Flask](https://flask.palletsprojects.com/en/1.1.x/), [Bootstrap](https://getbootstrap.com/), and hosted on [AWS EC2](https://aws.amazon.com/ec2/).

## Intro
I have always been fascinated by the Natural Language Processing side of machine learning, partly because I like to read, so naturally I am curious about how an AI intepretes languages, create original texts or converses.

I was very excited when Goergia Tech (the university where I am doing a part-time online Master in Machine Learning) announced a new Deep Learning course in collaboration with Facebook. 

While I am waiting for that course to be available for online student, I decided to work on a project myself. 

## Some thoughts
Text summarization is a difficult problem for many reasons. Despite the resurgence and advancement in deep learning techniques, text summarization is still a big challenge, because:

1. Variable-length input and output: a good summarizer must be able to handle input with varying length, and adjust the output's length as necessary to capture the essential information in the summary.
2. Training time and computation (for deep learning approach): This is a direct ramification of point 1, a more sophisiticated recurrent neural network model is required to be able to track important imformation regardless of whether the info appears at the beginning or at the end of an article. However, such a model will require a lot of memory to train, and I constantly ran out of memory while trying to do it.
3. Data, and domain dependency. Unfortunately, a universal deep learning text summariser will require a lot of training data from many different domains. In reality, most models are trained on data from a certain source, such as Google news, or Amazon product reviews, making them biased towards those domains.

For those reason, I had to take a step back from the rabbit hole, and pursed an extractive approach (does not require training a deep learning model) for the summariser.

I wrote more about the algorithm in the About section on the [app's website](http://www.thetextsummarizer.com/about).

## Reference
I based my algorithm on the following papers.  
*[1] Allahyari, M, Pouriyeh, S. et al (2017). Text Summarization Techniques: A Brief Survey.*  
*[2] Steinberger, J, Poesio, M. et al (2007). Two uses of anaphora resolution in summarization.*