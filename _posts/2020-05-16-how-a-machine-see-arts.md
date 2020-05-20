---
layout: post
title: "Van Gogh and Deep Learning:<br>How a machine see arts"
subtitle: "Field notes from using Deep Learning to recognise artists from paintings"
date: 2020-05-16
image: "deepart.png"
color: "#993366"
htmlwidgets: TRUE
---
<div class="post-image-container">
     <img src="/assets/images/deeplearning/title.png"> 
</div>
<br/>
You're rich. You just moved and need to decorate the new place. You give your 6-year-old a few hundred million dollars and tell him to run to the gallery to buy a couple of Picasso paintings. 

He asks, "how do I know which ones are Picasso"? 

Educational opportunity! "Well, look for paintings of people where each part is from a different angle and look like they are popping out of the wall, also there's this particular shade of yellow ..."

I hope you already see how that would end up a disaster when your kid brought home some Kandinsky's ...

Rules are hard to make, especially when they will be followed to the letter. That is how we tell machines to do tasks for us for a very long time. Following rules is the life purpose of machines.

But machines don't just blindly follow orders anymore. They can learn now.

Back to your hypothetical son. After selling all the paintings he bought at a loss, you decide to raise him cultured. You feed him artisan cheese and vintage wine and take him to art galleries on the weekends. He grows up and can immediately recognise a Picasso's painting at a glance. When asked how he does it, he shrugs, "I don't know, it just looks like one".

That's how we naturally think as humans. We unconsciously form concepts of subjects in the world, rather than creating strict rules about what is what. 

Who painted this?
<div class="post-image-container">
     <img src="/assets/images/deeplearning/example.jpg"> 
</div>
<br/>

Picasso, of course. You probably already guessed. Or at least, you shouldn't feel surprised now that I tell you.

But what made it Picasso? You probably didn't even think about it. You already have a rough feeling, an impression that it was. 

It comes from years of seeing Picasso's works in the pop culture, on the news. We have formed an internal concept of his distinguished style.

We can teach machines in the same way. These techniques are called Machine Learning. Neural Network is a specific Machine Learning technique that has become dominant in recent years. 

Deep Learning, which is what I will use to recognise artists from their paintings is the more advanced form of Neural Network.


### Overview of the process

I took a [collection of paintings](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) by some of the most influential artists like Van Gogh, Titian, and Da Vinci and feed them to a Deep Neural Network model (the machine). We tell the model which artist created which paintings so that it learns the styles and subjects of each artist.

After a few hours of looking at them, the model was able to look at a painting it has never seen before and tell me who painted it.

Below are 9 randomly drawn paintings and what the Deep Learning model think was the painter.

![](/assets/images/deeplearning/test2.png) 

The accuracy is roughly 85%. Definitely way better than me.


#### How does Deep Learning work?
To explain, we look at a specific type of Deep Learning - Convoluted Neural Network. It is the most popular for image recognition task.

Think of it as a company.

![](/assets/images/deeplearning/company.png) <br/> <br/>
At the lowest levels are workers who will look at a specific part of the painting, making note of the color and shape. 

The workers will report their findings to their managers. These managers take the reports and form a bigger picture of what is going on. They will collectively decide who they think the painter was.

The managers then send their guess up to the CEO, who will then tell them if they were correct.

Whether they get it right or not, they learn something. At first, the managers will be terrible at guessing, but they get better after thousands of rounds, thanks to the CEO who is always there to tell them when they're wrong.

A Deep Learning model will have a lot more layers, with many workers or managers at each layer.

In the classic Neural Network architecture, each worker will report to every single manager. It turns out this is not terribly efficient at learning local patterns. So in Convoluted Neural Network, a manager only manages a specific group of workers.

With the application of clever maths and more computing power, in recent years we have been able to add more layers to make the model more sophisticated and capable of learning more complex concepts - hence the name Deep Learning. It means Neural Networks with more layers.

### What does it see?

Take this painting by Rembrandt as an example.

<div class="post-image-container">
     <img src="/assets/images/deeplearning/rembrandt.jpg"> 
</div>
<br/>

Below is what an earlier layer of the model sees.

![](/assets/images/deeplearning/low_layer.png)

Each of these images is a channel, an aspect of the painting that the "workers" see. They can represent colours, shapes, edge etc. But they still very closely resemble the original image.

A few levels up, what the upper "managers" see is shown below. They don't look at the details anymore and pick out abstract ideas from the image.

![](/assets/images/deeplearning/high_layer.png)


If you look hard enough, you may find the concept of a face, a face with a Renaissance neckpiece, eyes and ears, and somewhere, perhaps with a weird indescribable blob is the concept of Rembrandt's art. 

Let's try another one, this time with a Van Gogh painting since everyone loves Van Gogh.

![](/assets/images/deeplearning/van_gogh_org.jpg)

The model was able to tell this was Van Gogh. To understand how, we ask the model to show us which part of the painting it believes to be uniquely Van Gogh.

![](/assets/images/deeplearning/van_gogh_heat.jpg)

The part that is highlighted red was the most "Van Gogh" part, according to the machine.

I ain't no art major so I can't tell you why it was important. What's with those weird trees in blue-purple-black colour???

Incidentally, this is Starry Night. Is that giant wavy black thing the same kind of tree?

<div class="post-image-container">
     <img src="/assets/images/deeplearning/starry_night.jpg"> 
</div>
<br/>

### Endnotes
The recent resurgence of Deep Learning was what enabled machines to [beat humans](https://deepmind.com/research/case-studies/alphago-the-story-so-far) at the game of Go, the most difficult game humans have ever invented. 

It was this power that gave birth to machines that [write poetry](https://www.theguardian.com/technology/2016/may/17/googles-ai-write-poetry-stark-dramatic-vogons), [make music](https://futurism.com/a-new-ai-can-write-music-as-well-as-a-human-composer), and help a government [track people](https://time.com/5735411/china-surveillance-privacy-issues/) by recognising every citizen's face.

It is fascinating and it is scary.

### Futher reading
For a more technical introduction to Neural Networks and Deep Learning, I highly recommend this site [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).

The book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by Google AI researcher François Chollet is a fantastic read. I am only halfway through it and already learned so much.



### Reference
The dataset was obtained from Kaggle's [Best artworks of all time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) dataset.

I also got many tips from this [Kaggle kernel](https://www.kaggle.com/supratimhaldar/deepartist-identify-artist-from-art) while building the model.