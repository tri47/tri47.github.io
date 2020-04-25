---
layout: post
title: "Analysing Goodbooks-10k dataset"
subtitle: "Most unfinished books, most controverisal books, and books as a network of preference"
date: 2020-04-26
image: "goodreads.png"
color: "#663399"
htmlwidgets: TRUE
---
**\*NOTE: This post contains interactive charts which are best viewed on a large screen.**

In this post, I analyse Goodreads's [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k). It contains 6 million user ratings for 10,000 most popular books on Goodreads. I did the majority of the analysis with [Spark](https://spark.apache.org/) on Amazon's [EMR](https://aws.amazon.com/emr/). The visualisation was done in [Plotly](https://plotly.com/).

### Most popular books

The most popular books (highest count of ratings) is shown below. The Hunger Games take the crown, followed by Harry Potter. In the top 10, Pride and Prejudice is the oldest book, published more than 100 years before any of the others.

<div>
    <a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/82/?share_key=yZ8nirN13Feeihe73Yp4hN" target="_blank" title="top_goodreads_books" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/82.png?share_key=yZ8nirN13Feeihe73Yp4hN" alt="top_goodreads_books" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:82" sharekey-plotly="yZ8nirN13Feeihe73Yp4hN" src="https://plotly.com/embed.js" async></script>
</div>

The next 10 books are below. The Harry Potter and the Hunger Games series dominate this list.


<ul style='font-size:16px'>
<li> The Diary of a Young Girl, Anne Frank </li>
<li> The Girl with the Dragon Tattoo, Stieg Larsson </li>
<li> Catching Fire, Suzanne Collins </li>
<li> Harry Potter and the Prisoner of Azkaban, J.K. Rowling </li>
<li> The Fellowship of the Ring, J.R.R. Tolkien </li>
<li> Mockingjay, Suzanne Collins</li>
<li> Harry Potter and the Order of the Phoenix, J.K. Rowling</li>
<li> The Lovely Bones, Alice Sebold</li>
<li> Harry Potter and the Chamber of Secrets, J.K. Rowling</li>
<li> Harry Potter and the Goblet of Fire, J.K. Rowling</li>
</ul>


### Most unfinished

Sometimes, we don't achieve all we set out to. The dataset also contains tags users have assigned to their books. Here, we look at books that have been given tags such as "unfinished", "just-cant-do-it", "half-finished" by Goodreads users.

|Rank| Cover      | Author           | Book name  |
| --- | :---------: | ------------- | ----- |
|1|![](https://images.gr-assets.com/books/1463157317m/168668.jpg)| Joseph Heller|Catch-22|
|2|![](https://images.gr-assets.com/books/1436732693m/13496.jpg)| George R.R. Martin|A Game of Thrones (A Song of Ice and Fire, #1)|
|3|![](https://images.gr-assets.com/books/1390053681m/19063.jpg)| Markus Zusak|The Book Thief|
|4|![](https://images.gr-assets.com/books/1352422904m/15823480.jpg)| Leo Tolstoy|Anna Karenina|
|5|![](https://images.gr-assets.com/books/1377756377m/7604.jpg)| Vladimir Nabokov|Lolita|

<ul style='font-size:16px'>
<li>6. Neil Gaiman, American Gods (American Gods, #1)</li>
<li>7. Susanna Clarke, Jonathan Strange & Mr Norrell</li>
<li>8. Jane Austen, Pride and Prejudice</li>
<li>9. E.L. James, Fifty Shades of Grey</li>
<li>10. George Orwell, 1984</li>
</ul>

I was more expecting War and Peace, Moby Dick and Ulysses. I suppose the aforementioned three already look like unfathomable tomes, which instantly kill any illusion that you might be able to finish them. On the other hand, Catch-22 and Lolita are not that thick, and as a result, are more deceiving about their mental digestibility.

### Most controversial

These are the books that have the highest variance in their ratings. I.e., people either give them very high or very low ratings.

<div>
    <a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/84/?share_key=SRQM6twAYTPi4czslCeVVg" target="_blank" title="top_controlversial_books" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/84.png?share_key=SRQM6twAYTPi4czslCeVVg" alt="top_controlversial_books" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:84" sharekey-plotly="SRQM6twAYTPi4czslCeVVg" src="https://plotly.com/embed.js" async></script>
</div>

<br>
Basically, religious texts, Twilight, and Fifty Shade of Grey are the most polarising books. WHo would have guessed?

### The book network 

The most fun I had with this dataset was to turn it into a graph problem. By studying the relationship between books via the readers' preference, we can identify relationships that may not be obvious.

In the graph below, books are linked if they share more than 2000 unique readers who gave them a 5-star rating. You can hover on the circles to see the titles.


In the centre are 7 dark green circles - the 7 Harry Potter books. This center also houses To Kill a Mockingbird and The Hunger Games. Let's call this the Mainstream Sphere, the Popular Clique, the Best-selling Black Hole, the Translated-to-2000-Languages Gang, or whatever you prefer.

<div>
    <a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/90/?share_key=GqVuJ9leYBDR55qUsX7aIM" target="_blank" title="graph_top_books" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/90.png?share_key=GqVuJ9leYBDR55qUsX7aIM" alt="graph_top_books" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:90" sharekey-plotly="GqVuJ9leYBDR55qUsX7aIM" src="https://plotly.com/embed.js" async></script>
</div>
<br>
On the bottom left side, you find the A Song of Ice and Fire series. It connects back to the center via the first book - A Game of Thrones. At the top, you see a similar pattern. The Tolkienian works are clustered together and connect to the Mainstream Center via The Hobbit and the first The Lord of The Rings book. In graph theory, these nodes are known as gatekeepers, since they establish the gateway between different groups.

On the right side, Georgie Orwell's dystopian classics Animal Farm and 1984 stand in solitude. We also see that Pride and Prejudice is more "mainstream" than Jane Eyre. Fans of The Hunger Games are also more likely to enjoy Twilight than fans of Harry Potter.

We now have a very interpretable book recommendation system. E.g., for a fan of The Hunger Games who wants to find something more serious to read, we can recommend To Kill a Mocking Bird and subsequently The Kite Runner. We can also help readers get out of their comfort zone by skipping a few intermediate nodes.

The clustering extend beyond grouping works by authors. To see how, let add more titles, reduce the threshold to form book connection, and zoom on two particular books as below.

<div>
    <a onclick="return false" href="https://plotly.com/~tri.qu.nguyen/94/?share_key=8UcvsWxKG628cCfwJuxRjH" target="_blank" title="graph_top_books2" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/94.png?share_key=8UcvsWxKG628cCfwJuxRjH" alt="graph_top_books2" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:94" sharekey-plotly="8UcvsWxKG628cCfwJuxRjH" src="https://plotly.com/embed.js" async></script>
</div>

To Kill a Mockingbird is the starting point of the more involved literature. It connects the Mainstream Center to Shakespeare, John Steinback and Mark Twain. I.e, To Kill a Mockingbird is the metaphorical gateway drug to books that explore and depict the human conditions and destinies in the universe.

1984 is also near the centre, and it forms a particular link with Fahrenheit 451. We know that both stories deal with a protagonist who lives and fights back in a dystopian world with a dictatorial government, hence the connection. However, 1984 is a lot more well-known and therefore closer to the Mainstream.

Without knowledge of the genres or authors of the books, simply by using the preference of readers, we were able to cluster books into groups of similar themes and genres.

Here we add even more titles. I made the graph below zoomable so you can explore the clusters on your own.

<div>
    <a href="https://plotly.com/~tri.qu.nguyen/97/?share_key=hyrHbtykWCMRH036VD1SS8" target="_blank" title="graph_top_books3" style="display: block; text-align: center;"><img src="https://plotly.com/~tri.qu.nguyen/97.png?share_key=hyrHbtykWCMRH036VD1SS8" alt="graph_top_books3" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="tri.qu.nguyen:97" sharekey-plotly="hyrHbtykWCMRH036VD1SS8" src="https://plotly.com/embed.js" async></script>
</div>

<br>
### Endnotes
Network diagrams as used above are a very powerful tool to study the relationships between entities. As we saw, it can establish implicit connections between entities, which is useful as a recommendation system or a clustering/segmentation tool. It has been used successfully in fraud detection as well as travel planning and optimisation. It is a topic that I would love to dig deeper into when I have time. For now, I need to reduce the number of unfinished books on my shelf :(.

### Bonus
This is what happened when I visualised all the connections among 10, books - an inscrutable mess.

![](/assets/images/book_network.png)