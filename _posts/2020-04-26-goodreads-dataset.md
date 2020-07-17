---
layout: post
title: "Goodreads' user rating data"
subtitle: "Most unfinished, most controverisial books, and books as a network of preference"
date: 2020-04-26
image: "goodreads.png"
color: "#663399"
htmlwidgets: TRUE
summary: How different are Harry Potter and 1984? What do the Bible and Twilight have in common?
---
**\*NOTE: This post contains interactive charts which are best viewed on a large screen.**


<iframe width="600" height="450" frameborder="0" scrolling="no" src="//plotly.com/~tri.qu.nguyen/90.embed"></iframe>

<br>
In this post, I analyse Goodreads's [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k). Goodreads is the most popular website for readers to share book reviews and maintain reading lists. As of 2020, Goodreads has more than 90 million users. The dataset contains 6 million user ratings for 10,000 most popular books. 

I conducted the analysis in Spark on Amazon's Elastic MapReduce (EMR). The visualisation was done in Plotly.

For the code, see my [Github repo](https://github.com/tri47/goodreads_10k_books).

### Most popular books

The most popular book according to the number of ratings is The Hunger Games, followed by Harry Potter. In the top 10, Pride and Prejudice is the oldest book, published more than 100 years before any of the others.

<div>
    <a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/82/?share_key=yZ8nirN13Feeihe73Yp4hN" target="_blank" title="top_goodreads_books" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/82.png?share_key=yZ8nirN13Feeihe73Yp4hN" alt="top_goodreads_books" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:82" sharekey-plotly="yZ8nirN13Feeihe73Yp4hN" src="https://plotly.com/embed.js" async></script>
</div>

<br>
### Most unfinished

Have you ever felt the guilt of not finishing something you started? You are not alone. Here, we summarised books that are most often given tags such as "unfinished", "just-cant-do-it", "half-finished" by readers.

<ul style='font-size:16px'>
<li>1. Catch-22, Joseph Heller</li>
<li>2. A Game of Thrones (A Song of Ice and Fire, #1), George R.R. Martin</li>
<li>3. The Book Thief, Markus Zusak</li>
<li>4. Anna Karenina, Leo Tolstoy</li>
<li>5. Lolita, Vladimir Nabokov</li>
<li>6. American Gods (American Gods, #1), Neil Gaiman</li>
<li>7. Jonathan Strange & Mr Norrell, Susanna Clarke</li>
<li>8. Pride and Prejudice, Jane Austen</li>
<li>9. Fifty Shades of Grey, E.L. James</li>
<li>10. 1984, George Orwell</li>
</ul>


### Most controversial

These are the books that have the highest variance in their ratings. I.e., people either give them very high or very low ratings.

<div>
    <a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/84/?share_key=SRQM6twAYTPi4czslCeVVg" target="_blank" title="top_controlversial_books" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/84.png?share_key=SRQM6twAYTPi4czslCeVVg" alt="top_controlversial_books" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:84" sharekey-plotly="SRQM6twAYTPi4czslCeVVg" src="https://plotly.com/embed.js" async></script>
</div>

<br>
So religious texts, Twilight, and Fifty Shade of Grey are the most polarising books. Who would have guessed?

### The book network 

Next, I turn the dataset into a graph problem. Studying the relationship between books via the readers' preference, we can identify relationships that may not be obvious.

In the graph below, books are linked if they share more than 2000 unique readers who gave them a 5-star rating. You can hover on the circles to see the titles.


In the centre are seven dark green circles - the seven Harry Potter books, along with To Kill a Mockingbird and The Hunger Games. Let's call this the Mainstream Centre.

<iframe width="600" height="450" frameborder="0" scrolling="no" src="//plotly.com/~tri.qu.nguyen/90.embed"></iframe>

<br>
On the bottom left, you find the notorious Game of Thrones (A Song of Ice and Fire) series. It connects back to the center via the first book in the series. 

At the top, the Tolkienian works are clustered together and linked to the Mainstream Center via The Hobbit. In graph theory, these connecting nodes are known as gatekeepers, since they establish the gateway between different groups.

On the right side, Georgie Orwell's dystopian classics Animal Farm and 1984 stand in solitude. We also see that Pride and Prejudice is more "mainstream" than Jane Eyre.

These linkages can be used as a recommendation system. We can also help readers get out of their comfort zone by skipping a few intermediate nodes from their favourite books.

Let's add more titles, reduce the threshold to form a connection, and zoom on two particular books as below.

<iframe width="600" height="450" frameborder="0" scrolling="no" src="//plotly.com/~tri.qu.nguyen/94.embed"></iframe>
<br>

To Kill a Mockingbird is the centre of more "deep" books. It connects the Mainstream Center to Shakespeare, John Steinback etc.. In other words, To Kill a Mockingbird is the gateway drug to serious literature.

1984 is also near the centre, and it forms a particular link with Fahrenheit 451. We know that both stories deal with a protagonist who lives and fights back in a dystopian world with a dictatorial government, hence the connection.

Without knowledge of the genres or authors of the books, simply by using the preference of readers, we were able to cluster books into groups of similar themes and genres.

Here we add even more titles. I made the graph below zoomable so you can explore the clusters on your own.

<iframe width="600" height="450" frameborder="0" scrolling="no" src="//plotly.com/~tri.qu.nguyen/97.embed"></iframe>

<br>
### Endnotes
Network diagrams as used above are a very powerful tool to study the relationships between entities. As we saw, it can establish implicit connections between entities, which is useful as a recommendation system or a clustering/segmentation tool. It has been used successfully in fraud detection as well as travel planning and optimisation. It is a topic that I would love to dig deeper into when I have time. For now, I need to reduce the number of unfinished books on my shelf!

### Bonus
This is what happened when I visualised all the connections within 10,000 books - an inscrutable mess.

![](/assets/images/book_network.png)