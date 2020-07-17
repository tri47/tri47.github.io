---
layout: post
title: "Infographic: the MoMA art collection"
subtitle: "Museum of Modern Art - New York"
date: 2020-05-13
image: "MoMA.png"
color: "#993366"
htmlwidgets: TRUE
summary: Asking pointy questions that will annoy the art gallery tour guide.
---


![Loading image ...](/assets/images/moma_info.png)


#### How this was made

The analysis was done in **Apache Spark**. The inforgraphic was created in **Adobe InDesign** with the great help of [Hanh La](https://www.linkedin.com/in/hanh-la-06886b128/).

The dataset was obtained from MoMA's [github's repository](https://github.com/MuseumofModernArt/collection). 


#### The code
There are two parts: analysing the dataset in Spark, and plotting the results in Plotly (Python).

See my [github repo](https://github.com/tri47/moma_collection_with_Spark) for the code.

**The Spark script for reference**

```python
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext
from operator import add
import pandas as pd
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, column, expr, lit

# Spark session configurations
conf = SparkConf().setMaster("local").setAppName("MoMA")
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()

# Read artworks
w = spark.read.csv(
    "Artworks.csv",mode="DROPMALFORMED", header='true', 
                      inferSchema='true'
).rdd
# Create the dataframe
w_df = sqlContext.createDataFrame(w, samplingRatio=0.2)


# Extract the created year of an artwork from Date column
w_df = w_df.withColumn('CreatedYear', F.regexp_extract(col('Date'),r"\d\d\d\d",0))
# Extract the obtained year of an artwork from DateAcquired column
w_df = w_df.withColumn('ObtainedYear', F.regexp_extract(col('DateAcquired'),r"\d\d\d\d",0))
w_df = w_df.withColumn('ObtainedYear', col('ObtainedYear').cast("int"))
# Extract the birth and death year
w_df = w_df.withColumn('BirthYear', F.regexp_extract(col('BeginDate'),r"\d\d\d\d",0))
w_df = w_df.withColumn('BirthYear', col('BirthYear').cast("int"))
w_df = w_df.withColumn('DeathYear', F.regexp_extract(col('EndDate'),r"\d\d\d\d",0))
w_df = w_df.withColumn('DeathYear', col('DeathYear').cast("int"))

# Cast width, height, and weight columns to DoubleType
w_df = w_df.withColumn("Width (cm)", col("Width (cm)").cast(DoubleType()))
w_df = w_df.withColumn("Height (cm)", col("Height (cm)").cast(DoubleType()))
w_df = w_df.withColumn("Area (cm^2)", col("Height (cm)")*col("Width (cm)"))
w_df = w_df.withColumn("Weight (kg)", col("Weight (kg)").cast(DoubleType()))

# Count number of credited artists on an artwork 
# ultising the gender column which specify gender for each artist
w_df = w_df.withColumn('CountArtist', F.size(F.split(col('Gender'),' ')) )

# Get the gender of first credited artist 
w_df = w_df.withColumn('FirstGender', F.regexp_extract(col('Gender'),r"\(([^()]+)\)",0))

# Get nationality of first mentioned artist
w_df = w_df.withColumn('FirstNationality', F.split(col('Nationality'),' ')[0])


# GET ALL PAINTINGS
w_df_pt = w_df.filter(col('Classification').isin(['Painting']))
w_df_pt.show(10)


### GENERATE OUTPUT CSV'S FOR VISUALISATION

# Sort by width to find widest painting
output = w_df_pt.sort(F.desc("Width (cm)")).select(['Title','Artist','Width (cm)'])
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('painting_widths.csv')

# Sum width of paintings with non-null width
output = w_df_pt.filter(col('Width (cm)').isNotNull())\
            .groupBy().sum('Width (cm)')

output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('painting_width_total.csv')


# Sum weight of Sculpture
# Cast data to double type
output = w_df.filter(col('Classification').isin(['Sculpture']))\
          .filter(col('Weight (kg)').isNotNull())\
            .groupBy().sum('Weight (kg)')
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('sculpture_weight_total.csv')

# Group paintings by countries
# Get the nationality of the first credited artist
w_dfs = w_df_pt.groupBy(['FirstNationality']).count()
output = w_dfs.sort(F.desc("count"))
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('painting_nationality.csv')

# group painting by artists
w_dfs = w_df_pt.groupBy(['Artist']).count()
output = w_dfs.sort(F.desc("count"))
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('painting_artists.csv')

# DISTRUBUTION BETWEEN GENDERS
# Group painting by gender obtained
w_dfs = w_df_pt.filter(col('CountArtist') == 1)
output = w_dfs.groupBy(['FirstGender']).count()
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('painting_gender_sum.csv')

# Group photograph by gender obtained
w_dfs = w_df.filter(w_df['Classification'].isin(['Photograph']))\
            .filter(col('CountArtist') == 1)
output = w_dfs.groupBy(['FirstGender']).count()
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('photos_gender_sum.csv')

# SHOW DISTRIBUTION OVER 30 YEARS OF ACQUISITION
# Group painting by gender and year obtained
w_dfs = w_df_pt.filter(col('CountArtist') == 1)
w_dfs = w_dfs.groupBy(['ObtainedYear','FirstGender']).count()
output = w_dfs.sort(F.desc("ObtainedYear"))
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('paintings_gender_each_year.csv')

# Group Photograph by gender and year obtained
w_dfs = w_df.filter(col('Classification').isin(['Photograph']))
w_dfs = w_dfs.filter(col('CountArtist') == 1)\
             .groupBy(['ObtainedYear','FirstGender']).count()
output = w_dfs.sort(F.desc("ObtainedYear"))
output.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('photos_gender_each_year.csv')


# GET THE YEAR AN ARTIST FIRST HAVE A PAINTING ACQUIRED BY MOMA
# Only works credited to an individual
w_dfs = w_df_pt.filter(col('CountArtist') == 1)
w_dfs= w_dfs.groupBy(['Artist','BirthYear','DeathYear']).min('ObtainedYear')\
          .withColumnRenamed('min(ObtainedYear)', 'ObtainedYear')\
          .withColumn('AgeFirstObtained', expr('ObtainedYear - BirthYear'))\
          .withColumn('AliveToSee', expr('DeathYear is Null or ObtainedYear < DeathYear'))
w_dfs.coalesce(1).write.option("header", "true")\
      .mode("overwrite")\
      .csv('painting_dead_vs_alive_artist.csv')

print('ENDING SESSION')
```





