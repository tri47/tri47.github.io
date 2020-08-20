---
layout: post
title: "Predicting Titanic survival with feature engineering and XGBoost"
subtitle: ""
date: 2020-08-15
image: "titanic.png"
color: "#993366"
htmlwidgets: TRUE
summary: How to reach top 4% of Kaggle Titanic dataset competition.
tags: machine-learning kaggle
---

<html>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div><div >
</div><div>
<div>
<img src="/assets/images/titanic_ship.png">
<br/>
<br/>
<p> <i>For my Jupyter notebook on the Kaggle competition, see <a href="https://www.kaggle.com/triqng/top-4-with-feature-engineering-and-xgboost">here.</a></i>
</p>
<p>The classic Titanic machine learning competition is an fantastic way to apply your machine learning skills and compare your work with others.</p>
<p>Even though the full dataset is available and you can cheat your way to perfect score, it is very satisfying to compete fairly and achieve good result.</p>
<p>In this notebook, I focus on the preprocessing aspect of machine learning, i.e., the cleaning and feature engineering of the dataset, which resulted in a top 4% position in the leaderboard with XGBoost.</p>
<p>This dataset is an excellent example to illustrate the power of understanding your dataset and making it more useful before diving into specific ML algorithms.</p>

<p>This post is divided into four parts:</p>

<p>
<b>Chapter 1: Missing values</b> <br/>
<b>Chapter 2: Feature engineering</b> <br/>
<b>Chapter 3: Assessing the features</b> <br/>
<b>Chapter 4: Prepare data for training</b> <br/>
<b>Chapter 5: Build a model to predict Titanic survival</b> <br/>
</p>
<p>Let's start by importing the necessary libraries.</p>
<br/>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sb</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">accuracy_score</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">KFold</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">cross_val_score</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">GridSearchCV</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">GaussianNB</span>
<span class="kn">from</span> <span class="nn">xgboost</span> <span class="kn">import</span> <span class="n">XGBClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="kn">import</span> <span class="n">RidgeClassifier</span>
<span class="n">sb</span><span class="o">.</span><span class="n">set</span><span class="p">(</span><span class="n">font_scale</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span> 
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We load the data and have a look. Wow, so many missing values (Null)!</p>
<p>What can I say, life is sad, and missing data is just a little dune in the desert of sadness.</p>
<p>But hey, a tiny bit of us data scientists is a naive bayesian, right? So we put our positive hat on and deal with those nulls.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Load data</span>
<span class="n">df_train</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;/kaggle/input/titanic/train.csv&#39;</span><span class="p">)</span>
<span class="n">df_train_org</span> <span class="o">=</span> <span class="n">df_train</span>
<span class="n">df_test</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;/kaggle/input/titanic/test.csv&#39;</span><span class="p">)</span>
<span class="n">df_full</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">df_train</span><span class="p">,</span><span class="n">df_test</span><span class="p">])</span>
<span class="n">df_train</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Chapter-1:-Missing-values">Chapter 1: Missing values</h4>


</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>There are two missing Embarked values in the train set. Let's check them out.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Embarked has 2 missing values</span>
<span class="n">df_train</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Embarked&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[3]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>62</td>
      <td>1</td>
      <td>1</td>
      <td>Icard, Miss. Amelie</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>829</th>
      <td>830</td>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>They are in the same cabin, and have no family otherwise (friends?).</p>
<p>We will try to guess where they embarked from based on the fare they paid and their fare class.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Show embarked values</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Embarked&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">()</span><span class="c1"># Show embarked values</span>
<span class="nb">print</span><span class="p">(</span><span class="n">df_full</span><span class="p">[</span><span class="s1">&#39;Embarked&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">())</span>
<span class="c1"># Distribution of Embarked vs fares</span>
<span class="n">sb</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s2">&quot;Embarked&quot;</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="s2">&quot;Fare&quot;</span><span class="p">,</span>
            <span class="n">hue</span><span class="o">=</span><span class="s2">&quot;Pclass&quot;</span><span class="p">,</span> <span class="n">palette</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;m&quot;</span><span class="p">,</span> <span class="s2">&quot;g&quot;</span><span class="p">],</span>
            <span class="n">data</span><span class="o">=</span><span class="n">df_full</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[&#39;S&#39; &#39;C&#39; &#39;Q&#39; nan]
</pre>
</div>
</div>

<div class="output_area">

    <div class="prompt output_prompt">Out[4]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0930912850&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>THe fare of 80 will likely land in Cherbourg (C). So we will Set the 2 missing embarked as 'C'. There is no embarked missing in test set.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Embarked&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span><span class="s1">&#39;Embarked&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="s1">&#39;C&#39;</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>There are a lot of cabin values missing, does it signify people who don't have private cabins? We change Null to a special class N.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_train</span><span class="o">.</span><span class="n">Cabin</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="s1">&#39;N&#39;</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">Cabin</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="s1">&#39;N&#39;</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Next, we deal with a missing Fare in the test set. We use the Ticket class (Pclass) to guess the fare value.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Missing fare in test set</span>
<span class="n">df_test</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[7]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152</th>
      <td>1044</td>
      <td>3</td>
      <td>Storey, Mr. Thomas</td>
      <td>male</td>
      <td>60.5</td>
      <td>0</td>
      <td>0</td>
      <td>3701</td>
      <td>NaN</td>
      <td>N</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="http://" alt="">Set the missing fare to be the mode of ll class 3 fares. Note we use all values from the train and the test sets. The passanger also had no companion on the ship.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#sb.boxplot(x=&quot;Pclass&quot;, y=&quot;Fare&quot;,</span>
 <span class="c1">#            palette=[&quot;m&quot;, &quot;g&quot;],</span>
  <span class="c1">#          data=df_full)</span>

<span class="c1">#guess_class_3_fare = df_full[(df_full[&#39;Pclass&#39;]==3) &amp; (df_full[&#39;Fare&#39;].notnull())][&#39;Fare&#39;].mean()</span>
<span class="c1">#print(guess_class_3_fare)</span>
<span class="c1">#df_test.Fare.fillna(guess_class_3_fare,inplace=True)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># See distribution of fare in class 3</span>
<span class="n">df_full</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">df_train</span><span class="p">,</span> <span class="n">df_test</span><span class="p">])</span>
<span class="n">class_3_fare</span> <span class="o">=</span> <span class="n">df_full</span><span class="p">[</span><span class="n">df_full</span><span class="o">.</span><span class="n">Pclass</span><span class="o">==</span><span class="mi">3</span><span class="p">][</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">sb</span><span class="o">.</span><span class="n">distplot</span><span class="p">(</span><span class="n">class_3_fare</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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xo20t7eTkZFBWVkZ8+bNY82aNWzYsIHFixcTCoV46qmnOH78OABr1qxh1apVAJecJyIiUy+q8hcRkelFb/iKiMQhlb+ISBxS+YuIxCGVv4hIHFL5i4jEIfPdJs+Eormr6VQoKyvjlVdeoba2lgMHDnDVVVeZIl9LSwvf//73qampISEhgdmzZ/PUU0/hdDpjng1g7dq1XLhwAavVSkpKCn//939PUVGRKbIBPPvss+zatSvyb2qGXHfeeScJCQkkJvbfmPC73/0uS5YsiXm2QCDAP/zDP/Db3/6WxMREvvjFL7Jt27aY57pw4QLr1q2LPO7o6MDv9/PGG2/EPNuIDBnVQw89ZLz88suGYRjGyy+/bDz00EMxyfHmm28adXV1xp/+6Z8a77//vmnytbS0GL/73e8ij//xH//R+Ju/+RtTZDMMw2hvb498/T//8z/G8uXLTZPt3XffNR599FHjK1/5SuTf1Ay5PvszNiDW2bZt22Y8/fTTRjgcNgzDMJqamkyR67O2b99ubN261TAM82UboPIfRXNzs3HjjTcawWDQMAzDCAaDxo033mh4vd6YZfr0L6YZ8x06dMj45je/acpsv/jFL4yvfe1rpsgWCASMb3zjG0ZNTU3k39QMuQxj+PKPdTa/32/ceOONht/vN1WuzwoEAsaXvvQl49133zVdtk/TsM8oor2raayYLV84HOZnP/sZd955p6my/eAHP+D48eMYhsG//du/mSLbM888w/33309hYWFkmhlyDfjud7+LYRjceOONfPvb3455tvPnz5OVlcWzzz7L66+/TmpqKn/1V39FUlKSab5nAK+++ip5eXksWrSId99911TZPk1v+MqE2rZtGykpKTz44IOxjjLI008/zWuvvcYTTzzBjh07Yh2Ht99+m3feeYfVq1fHOsqw/uM//oP//u//Zt++fRiGwVNPPRXrSASDQc6fP8/ChQv5r//6L7773e+yfv16urq6Yh1tkH379vH1r3891jFGpfIfRbR3NY0VM+UrKyvj448/5l/+5V+wWq2myjZg+fLlvP766+Tn58c025tvvslHH33EXXfdxZ133kl9fT2PPvooNTU1pvieDbxeQkICq1ev5g9/+EPM/z0LCgqw2+2UlJQAcN1115GdnU1SUpIpvmcADQ0NvPnmm9x3332AuX4/P0vlP4po72oaK2bJ98Mf/pB3332X3bt3k5CQYJpsnZ2deDyeyONXX32VzMzMmGd77LHHOHbsGK+++iqvvvoq+fn5/OQnP+HP/uzPYv496+rqoqOjA+j/TOrKykqKiopi/j1zOp186Utfitwgsrq6Gq/Xy5w5c2L+PRvwi1/8gjvuuIPs7P4PC4r19+xSdGO3KIx0V9Optn37dg4fPkxzczPZ2dlkZWVx8ODBmOc7c+YMJSUlzJkzh6SkJABmzZrF7t27Y56tubmZtWvX0t3djdVqJTMzkyeffJJFixbFPNun3Xnnnfz4xz/mqquuinmu8+fPs379ekKhEOFwmPnz5/N3f/d35ObmmiLb3/7t39La2ordbuev//qvueOOO2Kea0BxcTE/+MEPWLp0aWSaWbJ9lspfRCQOadhHRCQOqfxFROKQyl9EJA6p/EVE4pDKX0QkDqn8RUTikO7tI/IZd955J83NzZH7sQAcOnSIvLy8GKYSmVgqf5Fh/PjHP+ZP/uRPxvw8o/9OuVit+qNazE0/oSKjaGtr41vf+hZf/vKXufnmm/nWt75FfX19ZP5DDz3ED3/4Qx544AGuu+46zp8/z9mzZ/nzP/9zbrnlFoqLi6msrIzhFogMpfIXGUU4HGbFihX86le/4le/+hWJiYlD7nK5f/9+tm3bxh/+8AecTiePPPIIJSUl/OY3v2Hnzp1s3bqVM2fOxGgLRIbSsI/IMNatWxcZ87/lllvYs2dPZN5f/uVf8vDDDw9a/mtf+xpXXnklAEePHmXmzJmR2/ouWrSI4uJiXnnllcgyIrGm8hcZxu7duyNj/t3d3WzatImjR4/S1tYG9N8tNBQKRXYQn75Fb21tLSdPnuSmm26KTAuFQtx///1TuAUil6byFxnF888/T3V1Nf/5n/+Jy+Xi1KlTLF++nE/fE9FisUS+drvd3HzzzbzwwguxiCsSFY35i4yis7OTxMREMjIyaG1t5dlnn73k8l/5ylc4d+4cL7/8Mn19ffT19XHy5EnOnj07RYlFRqfyFxnFN7/5TQKBAF/+8pdZtWoVS5YsueTyaWlp/OQnP6GyspIlS5Zw++2380//9E/09vZOUWKR0el+/iIicUhH/iIicUjlLyISh1T+IiJxSOUvIhKHVP4iInFI5S8iEodU/iIicUjlLyISh/5/Jxgyp181N9oAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">guess_value</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span><span class="n">class_3_fare</span><span class="o">.</span><span class="n">mode</span><span class="p">())</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">Fare</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">guess_value</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">(),</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">guess_value</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Chapter-2:-Feature-engineering">Chapter 2: Feature engineering</h4><p>We create a bunch of features based on existing fields. The idea is to transform the information into more revealing features that would help with the classfication.</p>
<p>We do not necessarily use all the new features we create. Having them means we can test to see what is useful (for a particular algorithm).</p>
<p>It's a good spot to mention the concept of feature <strong>Relevance vs Usefulness</strong>.</p>
<p>Simply put, a feature is relevant if by itself it contains information that improves classification. On the other hand, a feature is useful if it improves classfication even though by itself it does not contain any information (e.g., a column of all zeros).</p>
<p>It certainly sounds strange that a feature with no information (correlation, entropy etc.) can be useful. This has to do with particular algorithms you use, which might benefits from the inclusion of a feature with little information.</p>
<p>Put it another way, <strong>relevance</strong> is about information, while <strong>usefulness</strong> is about effect on errors.</p>
<p>It's a very long way to say that when you create new features, it's hard to know what might be useful later. You can make an educated guess based on their correlation with the labels, but ultimately you want to systematically test them out along with your algorithms.</p>
<p>With all that over with, first, we create a few features to cover Family Size, Fare per person and a flag to indicate whether a passanger had no family on the trip.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Find family size</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;FamSize&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Parch&#39;</span><span class="p">]</span> <span class="o">+</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;SibSp&#39;</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;FamSize&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Parch&#39;</span><span class="p">]</span> <span class="o">+</span> <span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;SibSp&#39;</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span>
<span class="c1"># Fare per person</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;FarePp&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">/</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;FamSize&#39;</span><span class="p">]</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;FarePp&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">/</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;FamSize&#39;</span><span class="p">]</span>
<span class="c1"># IsAlone flag</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;IsAlone&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;FamSize&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">,</span><span class="s1">&#39;IsAlone&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;IsAlone&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;FamSize&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">,</span><span class="s1">&#39;IsAlone&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We extract the surnames of the passangers.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Get family/surname</span>
<span class="n">split_cols</span> <span class="o">=</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;,&#39;</span><span class="p">)</span>
<span class="n">surnames</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">row</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">split_cols</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]):</span>
    <span class="n">name</span> <span class="o">=</span> <span class="n">split_cols</span><span class="p">[</span><span class="n">row</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">surnames</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">name</span><span class="p">)</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">surnames</span>

<span class="n">split_cols</span> <span class="o">=</span> <span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;,&#39;</span><span class="p">)</span>
<span class="n">surnames</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">row</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">split_cols</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]):</span>
    <span class="n">name</span> <span class="o">=</span> <span class="n">split_cols</span><span class="p">[</span><span class="n">row</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">surnames</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">name</span><span class="p">)</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">surnames</span>
<span class="c1">#df_train[&#39;Surname&#39;] =df_train[&#39;Surname&#39;][0] </span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>One thing that's unclear is whether the fare is per person or the whole family, so we do some sanity check to decide which way to go.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Check if the fare is per person or for the whole family</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">nrows</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">10</span><span class="p">))</span>
<span class="n">sb</span><span class="o">.</span><span class="n">scatterplot</span><span class="p">(</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;FamSize&#39;</span><span class="p">,</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Fare&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Unmodified Fare values&#39;</span><span class="p">)</span>
<span class="n">sb</span><span class="o">.</span><span class="n">scatterplot</span><span class="p">(</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;FamSize&#39;</span><span class="p">,</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;FarePp&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span><span class="n">ax</span><span class="o">=</span><span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Per person Fare values&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Text(0.5, 1.0, &#39;Per person Fare values&#39;)</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmQAAAJiCAYAAABpSN6hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOzde3gTZfo38O/MJGmTFmyLFKsgvLKKCKsU2oILeCgIyqGKh21FkYq7iqDg6u5CERWLgCgecFHU36LVXRQPKCqggBwEVGjVIqCwICdRDqUQtDRpk8w87x9thqRNStqmnbT9fq7Ly2TmOdzP3E1zM5lJJSGEABEREREZRjY6ACIiIqKWjgUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBFRxPrggw9w66236s+Tk5Nx8OBBAEBZWRnGjh2LXr16YcKECfj4448xZsyYsMzTlDWntRC1JCajAyAiY3Xp0gUrV65Ex44d9W3/+te/cODAAcyZM8fAyKorLCzUH3/22WcoLi7G5s2bYTJV/CrLyMgI+5y//PILBgwYAJvNpm/r0KEDPv7447DPRUQtFwsyImqSDh06hE6dOunFWEMrKCio81wej6fR4iSipokfWRJRjTZv3owrrrgCr732Gi6//HL069cPixcv1vdPnjwZ06ZNw1/+8hckJycjKysLx44dw4wZM5Camoprr70WP/74o95+z549GDVqFFJSUjB06FCsXr1a32e32zF27Fj07NkTN998M37++We/WLp06YIDBw7ghRdewEsvvYRPP/0UycnJeO+996p9VLdnzx7ceeedSEtLw+DBg7F8+fKQ5wnF1q1bkZmZiZSUFPTr1w+5ublwuVx+sS5cuBCDBg3CoEGDAABr167F9ddfj5SUFGRlZWHnzp0Bx3700Ucxe/Zsv2333nsvXn/9dQDAq6++ioEDByI5ORlDhgzBqlWrAo7zyy+/oEuXLvB4PPq2UaNG4b333tOfv//++7juuuuQmpqKu+66C7/++isAQAiBmTNn4vLLL0evXr0wfPhw7Nq1q9bHiYhCw4KMiM6ouLgYJSUlWL9+PWbMmIHc3Fz89ttv+v5PP/0UDzzwADZt2gSLxYLMzEx069YNmzZtwuDBgzFr1iwAgNvtxtixY9G3b1989dVXmDp1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CVVjQUZERFQfLMgoZIosIzHe6rctMd4KReaPERERUX3wnZRCFhMlIyc7TS/KvNeQxUTxx4iIiKg+eFE/hczhcCMpPhqzxvWDqmlQZBkxUTIv6CciIqonFmRUKw6HGxIqf3BUFQ6HanBERERETR8/ayIiIiIyGAsyIiIiIoOxICMiIiIyGAsyIiIiIoOxICMiIiIyGAsyIiIiIoOxICMiIiIyGAsyIiIiIoOxICMiIiIyGAsyIiIiIoOxICMiIiIyGAsyIiIiIoOxICMiIiIyGAsyIiIiIoM1SkE2e/ZspKeno0uXLti1a5e+fd++fcjMzMTgwYORmZmJ/fv3N0Y4RERERBGlUQqyAQMGYOHChTjvvPP8tj/22GMYOXIkVqxYgZEjR+LRRx9tjHCIiIiIIkqjFGQpKSlISkry23b8+HH8+OOPGDZsGABg2LBh+PHHH3HixInGCImIiIgoYpiMmvjw4cNo164dFEUBACiKgsTERBw+fBgJCQm1GqtNm9iGCNFP27atGnwOCg1zETmYi8jBXEQO5iJyNKVcGFaQhdPx46egaaLBxm/bthWOHStpsPEpdMxF5GAuIgdzETmYi8gRibmQZSnoSSTD7rJMSkrC0aNHoaoqAEBVVRQVFVX7aJOIiIiouTOsIGvTpg26du2KpUuXAgCWLl2Krl271vrjSiIiIqKmrlE+snziiSewcuVKFBcX484770RcXByWLVuGadOmYfLkyXjppZfQunVrzJ49uzHCISIiIoookhCi4S6+aiS8hqzlYC4iB3MROZiLyMFcRI5IzEVEXkNGRERERBVYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZzGR0AACwdu1azJ07F0IIaJqG+++/H4MGDTI6rCbNajPDUa5B1TQosgxblAynw210WERERBSA4QWZEAL//Oc/sXDhQlx00UXYuXMnbr31VgwcOBCyzBN4dWG1mXHEXoZZefkosjuRGG9FTnYazomPZlFGREQUgSKi4pFlGSUlJQCAkpISJCYmshirB0e5phdjAFBkd2JWXj4c5ZrBkREREVEghp8hkyQJzz//PMaNGwebzYbS0lK88sortRqjTZvYBorutLZtWzX4HOFyuPiUXox5FdmdUDWBpCa0jmCaUi6aO+YicjAXkYO5iBxNKReGF2QejwevvPIKXnrpJfTq1Qvffvst/va3v2HZsmWIiYkJaYzjx09B00SDxdi2bSscO1bSYOOHm6IoSIy3+hVlifFWKLLUpNYRSFPLRXPGXEQO5iJyMBeRIxJzIctS0JNIhn8uuGPHDhQVFaFXr14AgF69esFqtWLPnj0GR9Z02aJk5GSnITHeCgD6NWS2KMPTTURERAEYfobsnHPOwZEjR7B3715ccMEF2LNnD4qLi3H++ecbHVqT5XS4cU58NGaN68e7LImIiJoAwwuytm3bYtq0aZg4cSIkSQIAzJo1C3FxcQZH1rQ5HW5IqEywqsLpUA2OiIiIiIIxvCADgIyMDGRkZBgdBhEREZEheFERERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAUZERERkcFYkBEREREZjAVZPSiKDKHI8EgShCJDUcJ3OG02M4SiVI6twGYzh21sIiIiiiwmowNoqhRFht3hxsy8fBTZnUiMt2JKdhribWaoqlavsW02Mw7byzDLZ+yc7DQkxUfD4XCHaQVEREQUKXiGrI48gF6MAUCR3YmZefnwhGHs0nJNL8a8Y8/Ky0dpef0KPSIiIopMLMjqSNWEXjB5FdmdUDURhrG1IGOzICMiImqOalWQffnll5gyZQrGjh0LANi2bRu+/vrrBgks0imyhMR4q9+2xHgrFFkKw9hykLFZPxMRETVHIb/D/+c//8G0adPQqVMnFBQUAACio6Mxd+7cBgsukpkATMlO0wsn7zVk4bgoLyZKRk6VsXOy0xATxYKMiIioOQq5fnjjjTeQl5eH9u3b4//+7/8AABdccAH27dvXYMFF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Bx98gFtvvVV/XtNfHPjiiy8wZMgQJCcno3///liwYEGtjxER1YEgIhJCXH311eLLL78UQghx6NAhMWTIEPHcc8+JI0eOiLS0NLFu3TqhqqrYuHGjSEtLE8ePHxdCCHH77beLK6+8UuzatUu43W7hcrkCjj106FBx6NAhYbfbRWZmpnj22WeFEEJs375d9OnTR2zZskV4PB7xwQcfiKuvvlqUl5frfTMyMsShQ4eE0+kUe/bsEVdccYU4cuSIEEKIgwcPigMHDgghhHj++efFLbfcIoqLi8Xx48dFZmameO6554QQQmzatEl07dpVPP/888Llcol169aJSy+9VJw8eTLg8bj99tvFu+++W237e++9J0pKSkR5ebl44oknREZGhr5v0qRJomfPnuKbb74RqqoKh8MhRowYIf71r3+J8vJy8fPPP4v09HSxfv36gHMOGDBAbNy4UX9+//33i1deeUUIIcS2bdtEYWGhcLvd4uDBg+Laa68Vr7/+ut72oosuEvv37w8Y++LFi0VWVpYQQojS0lJxxRVXiPfff1+43W6xfft2kZaWJnbt2iWEEKJv376ioKBACCHEyZMnxfbt2wPGSkThxTNkRKQbP348UlJSMHLkSKSmpmLs2LH46KOPcMUVV+DKK6+ELMvo27cvunfvji+++ELv5/1Ge5PJBLPZHHDs2267DUlJSYiLi8O9996LZcuWAQDeffddZGZm4rLLLoOiKBgxYgTMZjO2bNmi9x01ahSSkpIQHR0NRVHgcrmwZ88euN1utG/fHueffz4A4JNPPsH48ePRpk0bJCQkYPz48fj444/1cUwmE8aPHw+z2Ywrr7wSNpsN+/btC3o8nnjiCaSkpCAlJQUjRowAANx8882IjY2FxWLB/fffj507d6KkpETvM2DAAPTq1QuyLGPXrl04ceIE7rvvPlgsFnTo0AF//vOfg/4NzKFDh2Lp0qUAgFOnTmH9+vUYOnQoAKB79+7o0aMHTCYT2rdvj8zMTBQUFARPZhDr1q3T/+KAyWTy+4sD3mP0008/4dSpUzjrrLPQrVu3Ws9BRLXHi/qJSOf7TfZehw4dwmeffYa1a9fq2zweD3r37q0/9/1G+2B825x77rkoKirSx1+yZAn++9//6vvdbre+v2rfjh07YsqUKfjXv/6Fn376Cf369cPkyZPRrl07FBUV4dxzzw04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U4YpuY4UCZpKJjKe1RcM+9WRPlhXKGAqlfSuHgAAUL7DBrI///nP+pu/+RtJUiKR0NNPP62HHnpIp512mr7whS/o8ssvP6pA1t7erieeeELPPfdc4UNmjz322COebyA4jqt4IqMPDnRozOiQmppb1BpPKhYN65p50/Tk+i2af/4UTYiGPAllwbBfrfs6iupqbKhTbGyIUAYAwBBw2M8hy2QyikQikqRXX31Vo0aN0mmnnSZJOvHEExWPx4+qgHfffVdjx47VypUrdfHFF+uSSy7RH/7wh6Oas7990J7Sbc0tOi42phB6JKk1ntRPH92oWTMmqam5RYmU40l9HWmnW11NzS3qSHtTDwAA6JvDniE77rjj9NJLL2nmzJlat26dZs6cWVi3d+9ehcPhoyogm83q3Xff1Uc+8hF997vf1SuvvKIrrrhCv/nNbzR6dOnP6jhUT5/p0V9a4wm1xpOyHacQeg6uS6oq4u9c76q2pmpAayllx+4DJevyqp6BVjMMj2mooheVg15UDnpROYZSLw4byK6++mpdddVVmjhxorZu3apf/OIXhXVr167VX//1Xx9VAR/60Ifk8/k0Z84cSdIZZ5yhaDSqbdu2lT33QH8wrD/kVywalmWaikXDReEnFg2rLZHpXG948iF0lmWVrMuregZSJX7Q30hFLyoHvagc9KJyVGIvevtg2MNesjzvvPP0H//xH7riiiu0evVqnX766YV1J5xwgr797W8fVXHV1dWaOXOmXnjhBUnStm3btGfPHk2aNOmo5u1Px4wKaklDnf5f6/7cvVnR3FnB/D1kazdsV2NDnSLBsv8SVb8KBcxudTU21CkU8KYeAADQNxXxp5PeffddLVmyRPv27ZPP59O3vvUtffKTnyx7+8H400l797YXv8vSdWUZBu+yHGSV+BvPSEUvKge9qBz0onJUYi+O+E8ndfXee+9p5cqV2rRpkxKJRNG6NWvWHFWBEydOLLoUWols25EhKZXI/bfwxNmSISmZsD2rTZJSyczBumxbqaS39QAAgPKVHcgWL16sE044Qddcc41CodBA1gQAADCilB3Itm7dqkcffVSmyX1JAAAA/ansdPWpT31KLS0tA1kLAADAiFT2GbIbb7xR8+fP1/HHH69x48YVrWtqaur3wgAAAEaKsgNZY2OjLMvSiSeeqGAwOJA1AQAAjChlB7IX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oBeVg15UDnpROSqtF73VMyw+hwwAAGAo45IlAACAxwhkAAAAHiOQAQAAeIxABgAA4DECGQAAgMcIZAAAAB4jkAEAAHiMQAYAAOAxAhkAAIDHCGS92LZtm+bNm6fZs2dr3rx5evvtt70uaVhbvny56uvrdcopp+itt94qLO+tD/So/8XjcX3961/X7Nmz9dnPflZXX3219u7dK4leeOHKK6/U3LlzdeGFF2rBggXatGmTJHrhpZUrVxZ9n6IXg6++vl4XXHCBPve5z+lzn/uc1q9fL2mI98JFjy655BL3iSeecF3XdZ944gn3kksu8bii4W3Dhg3ue++9537qU59y33zzzcLy3vpAj/pfPB53X3zxxcLXt99+u9vY2Oi6Lr3wwv79+wuPf/Ob37gXXnih67r0wiuvvfaau2jRIvfcc88tfJ+iF4Pv0J8TeUO5FwSyHuzevdudPn26m81mXdd13Ww2606fPt3ds2ePx5UNf13/R+utD/RocPz61792Fy5cSC8qwOOPP+5edNFF9MIjqVTK/dKXvuS+8847he9T9MIbpQLZUO+Fz+szdJVqx44dGj9+vCzLkiRZlqVYLKYdO3aourra4+pGjt764LouPRpgjuPo4YcfVn19Pb3w0NKlS/XCCy/IdV39y7/8C73wyIoVKzR37lxNnDixsIxeeOc73/mOXNfV9OnTdd111w35XnAPGYAe3XzzzYpEIvrKV77idSkj2q233qrf/e53uvbaa3XHHXd4Xc6ItHHjRr366qtasGCB16VA0kMPPaT//M//1GOPPSbXdbVs2TKvSzpqBLIe1NbWaufOnbJtW5Jk27ZaW1tVW1vrcWUjS299oEcDa/ny5dq+fbvuuusumaZJLyrAhRdeqJdeekkTJkygF4Nsw4YN2rp1q2bNmqX6+nq9//77WrRokd555x164YH8cxgIBLRgwQL98Y9/HPLfowhkPRg3bpymTp2q1atXS5JWr16tqVOnVsypzZGitz7Qo4Hzk5/8RK+99pruueceBQIBSfTCC+3t7dqxY0fh63Xr1umYY46hFx64/PLL9fzzz2vdunVat26dJkyYoHvvvVef+cxn6MUgSyQSamtrkyS5rqunn35aU6dOHfL/Xxiu67peF1GptmzZohtuuEH79+/XmDFjtHz5cp1wwglelzVs3XLLLXrmmWe0e/duRaNRjR07Vk899VSvfaBH/W/z5s2aM2eOJk+erFAoJEk67rjjdM8999CLQbZ7925deeWVSiaTMk1TxxxzjL773e/q1FNPpRceq6+v16pVq3TyySfTi0H27rvv6pvf/KZs25bjODrxxBN14403KhaLDeleEMgAAAA8xiVLAAAAjxHIAAAAPEYgAwAA8BiBDAAAwGMEMgAAAI8RyAAAADzG37IEMGzU19dr9+7dhb9XJ0m//vWvNX78+H6Z/4YbbtDq1avl9/vl9/t16qmn6sYbb9SJJ57YL/MDGLk4QwZgWFm1apU2btxY+FduGHNdV47jHHbcokWLtHHjRj333HOqrq5WY2Pj0ZYMAJwhAzB8ffDBB7r++uv1yiuvyLZtffSjH9VNN92kCRMmSJIuueQSffSjH9VLL72kN954Q08++aSy2axuueUWvf7664pGo1q8eLE+85nPdJs7HA7rs5/9rK699lpJ0t13363NmzfLNE0999xzmjx5spqamjRlypRBPWYAQxNnyAAMW47j6OKLL9azzz6rZ599VsFgUMuWLSsa86tf/Uo333yz/vjHP6q6ulqXXXaZ5syZo//+7//Wj3/8Y910003avHlzt7nb29v15JNPaurUqYVla9eu1QUXXKCWlhbNmTNHV155pTKZzIAfJ4Chj0AGYFi56qqrdNZZZ+mss87S0qVLNXv2bIXDYY0ePVrf+MY3tGHDhqLxF110kU466ST5fD6tX79eH/7wh/X5z39ePp9Pp556qmbPnq01a9YUxt93330666yzdP7556u9vV233357Yd2pp56qCy64QH6/X5deeqnS6bReeeWVQTt2AEMXlywBDCv33HOPPv7xj0uSksmkvve972n9+vX64IMPJOXObNm2Xbjxv7a2trDtX/7yF/3pT3/SWWedVVhm27bmzp1b+Pqyyy4rXKY8VP5SqCSZpqnx48ertbW1/w4OwLBFIAMwbN13333atm2b/u3f/k01NTXatGmTLrzwQrmuWxhjGEbhcW1trWbMmKH777//iPb3/vvvFx47jqOdO3cqFosd+QEAGDG4ZAlg2Gpvb1cwGNSYMWO0b98+rVy5stfx5557rt5++2098cQTymQyymQy+tOf/qQtW7aUtb/XX39dzzzzjLLZrB544AEFAgGdccYZ/XEoAIY5AhmAYWvhwoVKpVI6++yzNW/ePH3iE5/odfzo0aN177336umnn9YnPvEJnXPOObrzzjuVTqfL2t+sWbP09NNPa8aMGfrVr36lu+++W36/vz8OBcAwZ7hdz90DAI7I3Xffre3bt+vOO+/0uhQAQxBnyAAAADxGIAMAAPAYlywBAAA8xhkyAAAAjxHIAAAAPEYgAwAA8BiBDAAAwGMEMgAAAI8RyAAAADz2/wFYeZs4b8OGrwAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's look at the very high fares on the right. We discover the second row (2 member family), where the Fare exactly match those of the individual fare. This suggests the fare is not total for the family?</p>
<p>This is still not compelling evidence, so we dig deeper.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Check the fare outliner</span>
<span class="n">df_train</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Fare&#39;</span><span class="p">]</span><span class="o">&gt;</span><span class="mi">500</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[14]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>FamSize</th>
      <th>FarePp</th>
      <th>IsAlone</th>
      <th>Surname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>N</td>
      <td>C</td>
      <td>1</td>
      <td>512.3292</td>
      <td>1</td>
      <td>Ward</td>
    </tr>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
      <td>2</td>
      <td>256.1646</td>
      <td>0</td>
      <td>Cardeza</td>
    </tr>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
      <td>1</td>
      <td>512.3292</td>
      <td>1</td>
      <td>Lesurer</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Contemplating removing the outliners. This might be beneficial or detrimental to the performance, so it's also best to test.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Remove the outliners</span>
<span class="c1">#df_train = df_train[df_train[&#39;Fare&#39;]&lt;500]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We use the first letter of the cabin number to group people from the same area on the ship.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Extract the first letter of the cabins</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;CabinClass&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Cabin&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">str</span><span class="p">)</span><span class="o">.</span><span class="n">str</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;CabinClass&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Cabin&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">str</span><span class="p">)</span><span class="o">.</span><span class="n">str</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Check fare sorted by cabin class. 
Manually check a few family, the fare in class B increases in line with number of cabins
The fare is probably a combination of number of people + number of cabins</p>
<p>Therefore, using fare per person is not perfect but will be a more reasonable way to classify spending on fare.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Check fare sorted by cabin class</span>
<span class="n">df_train</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;CabinClass&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;B&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">sort_values</span><span class="p">([</span><span class="s1">&#39;Cabin&#39;</span><span class="p">,</span><span class="s1">&#39;CabinClass&#39;</span><span class="p">,</span><span class="s1">&#39;Name&#39;</span><span class="p">])</span>
<span class="c1"># </span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[17]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>FamSize</th>
      <th>FarePp</th>
      <th>IsAlone</th>
      <th>Surname</th>
      <th>CabinClass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
      <td>1</td>
      <td>512.329200</td>
      <td>1</td>
      <td>Lesurer</td>
      <td>B</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0000</td>
      <td>B102</td>
      <td>S</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>Fry</td>
      <td>B</td>
    </tr>
    <tr>
      <th>329</th>
      <td>330</td>
      <td>1</td>
      <td>1</td>
      <td>Hippach, Miss. Jean Gertrude</td>
      <td>female</td>
      <td>16.0</td>
      <td>0</td>
      <td>1</td>
      <td>111361</td>
      <td>57.9792</td>
      <td>B18</td>
      <td>C</td>
      <td>2</td>
      <td>28.989600</td>
      <td>0</td>
      <td>Hippach</td>
      <td>B</td>
    </tr>
    <tr>
      <th>523</th>
      <td>524</td>
      <td>1</td>
      <td>1</td>
      <td>Hippach, Mrs. Louis Albert (Ida Sophia Fischer)</td>
      <td>female</td>
      <td>44.0</td>
      <td>0</td>
      <td>1</td>
      <td>111361</td>
      <td>57.9792</td>
      <td>B18</td>
      <td>C</td>
      <td>2</td>
      <td>28.989600</td>
      <td>0</td>
      <td>Hippach</td>
      <td>B</td>
    </tr>
    <tr>
      <th>170</th>
      <td>171</td>
      <td>0</td>
      <td>1</td>
      <td>Van der hoef, Mr. Wyckoff</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>111240</td>
      <td>33.5000</td>
      <td>B19</td>
      <td>S</td>
      <td>1</td>
      <td>33.500000</td>
      <td>1</td>
      <td>Van der hoef</td>
      <td>B</td>
    </tr>
    <tr>
      <th>690</th>
      <td>691</td>
      <td>1</td>
      <td>1</td>
      <td>Dick, Mr. Albert Adrian</td>
      <td>male</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>17474</td>
      <td>57.0000</td>
      <td>B20</td>
      <td>S</td>
      <td>2</td>
      <td>28.500000</td>
      <td>0</td>
      <td>Dick</td>
      <td>B</td>
    </tr>
    <tr>
      <th>781</th>
      <td>782</td>
      <td>1</td>
      <td>1</td>
      <td>Dick, Mrs. Albert Adrian (Vera Gillespie)</td>
      <td>female</td>
      <td>17.0</td>
      <td>1</td>
      <td>0</td>
      <td>17474</td>
      <td>57.0000</td>
      <td>B20</td>
      <td>S</td>
      <td>2</td>
      <td>28.500000</td>
      <td>0</td>
      <td>Dick</td>
      <td>B</td>
    </tr>
    <tr>
      <th>745</th>
      <td>746</td>
      <td>0</td>
      <td>1</td>
      <td>Crosby, Capt. Edward Gifford</td>
      <td>male</td>
      <td>70.0</td>
      <td>1</td>
      <td>1</td>
      <td>WE/P 5735</td>
      <td>71.0000</td>
      <td>B22</td>
      <td>S</td>
      <td>3</td>
      <td>23.666667</td>
      <td>0</td>
      <td>Crosby</td>
      <td>B</td>
    </tr>
    <tr>
      <th>540</th>
      <td>541</td>
      <td>1</td>
      <td>1</td>
      <td>Crosby, Miss. Harriet R</td>
      <td>female</td>
      <td>36.0</td>
      <td>0</td>
      <td>2</td>
      <td>WE/P 5735</td>
      <td>71.0000</td>
      <td>B22</td>
      <td>S</td>
      <td>3</td>
      <td>23.666667</td>
      <td>0</td>
      <td>Crosby</td>
      <td>B</td>
    </tr>
    <tr>
      <th>61</th>
      <td>62</td>
      <td>1</td>
      <td>1</td>
      <td>Icard, Miss. Amelie</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0000</td>
      <td>B28</td>
      <td>C</td>
      <td>1</td>
      <td>80.000000</td>
      <td>1</td>
      <td>Icard</td>
      <td>B</td>
    </tr>
    <tr>
      <th>829</th>
      <td>830</td>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0000</td>
      <td>B28</td>
      <td>C</td>
      <td>1</td>
      <td>80.000000</td>
      <td>1</td>
      <td>Stone</td>
      <td>B</td>
    </tr>
    <tr>
      <th>779</th>
      <td>780</td>
      <td>1</td>
      <td>1</td>
      <td>Robert, Mrs. Edward Scott (Elisabeth Walton Mc...</td>
      <td>female</td>
      <td>43.0</td>
      <td>0</td>
      <td>1</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B3</td>
      <td>S</td>
      <td>2</td>
      <td>105.668750</td>
      <td>0</td>
      <td>Robert</td>
      <td>B</td>
    </tr>
    <tr>
      <th>54</th>
      <td>55</td>
      <td>0</td>
      <td>1</td>
      <td>Ostby, Mr. Engelhart Cornelius</td>
      <td>male</td>
      <td>65.0</td>
      <td>0</td>
      <td>1</td>
      <td>113509</td>
      <td>61.9792</td>
      <td>B30</td>
      <td>C</td>
      <td>2</td>
      <td>30.989600</td>
      <td>0</td>
      <td>Ostby</td>
      <td>B</td>
    </tr>
    <tr>
      <th>369</th>
      <td>370</td>
      <td>1</td>
      <td>1</td>
      <td>Aubart, Mme. Leontine Pauline</td>
      <td>female</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17477</td>
      <td>69.3000</td>
      <td>B35</td>
      <td>C</td>
      <td>1</td>
      <td>69.300000</td>
      <td>1</td>
      <td>Aubart</td>
      <td>B</td>
    </tr>
    <tr>
      <th>641</th>
      <td>642</td>
      <td>1</td>
      <td>1</td>
      <td>Sagesser, Mlle. Emma</td>
      <td>female</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17477</td>
      <td>69.3000</td>
      <td>B35</td>
      <td>C</td>
      <td>1</td>
      <td>69.300000</td>
      <td>1</td>
      <td>Sagesser</td>
      <td>B</td>
    </tr>
    <tr>
      <th>487</th>
      <td>488</td>
      <td>0</td>
      <td>1</td>
      <td>Kent, Mr. Edward Austin</td>
      <td>male</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>11771</td>
      <td>29.7000</td>
      <td>B37</td>
      <td>C</td>
      <td>1</td>
      <td>29.700000</td>
      <td>1</td>
      <td>Kent</td>
      <td>B</td>
    </tr>
    <tr>
      <th>536</th>
      <td>537</td>
      <td>0</td>
      <td>1</td>
      <td>Butt, Major. Archibald Willingham</td>
      <td>male</td>
      <td>45.0</td>
      <td>0</td>
      <td>0</td>
      <td>113050</td>
      <td>26.5500</td>
      <td>B38</td>
      <td>S</td>
      <td>1</td>
      <td>26.550000</td>
      <td>1</td>
      <td>Butt</td>
      <td>B</td>
    </tr>
    <tr>
      <th>539</th>
      <td>540</td>
      <td>1</td>
      <td>1</td>
      <td>Frolicher, Miss. Hedwig Margaritha</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>2</td>
      <td>13568</td>
      <td>49.5000</td>
      <td>B39</td>
      <td>C</td>
      <td>3</td>
      <td>16.500000</td>
      <td>0</td>
      <td>Frolicher</td>
      <td>B</td>
    </tr>
    <tr>
      <th>194</th>
      <td>195</td>
      <td>1</td>
      <td>1</td>
      <td>Brown, Mrs. James Joseph (Margaret Tobin)</td>
      <td>female</td>
      <td>44.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17610</td>
      <td>27.7208</td>
      <td>B4</td>
      <td>C</td>
      <td>1</td>
      <td>27.720800</td>
      <td>1</td>
      <td>Brown</td>
      <td>B</td>
    </tr>
    <tr>
      <th>587</th>
      <td>588</td>
      <td>1</td>
      <td>1</td>
      <td>Frolicher-Stehli, Mr. Maxmillian</td>
      <td>male</td>
      <td>60.0</td>
      <td>1</td>
      <td>1</td>
      <td>13567</td>
      <td>79.2000</td>
      <td>B41</td>
      <td>C</td>
      <td>3</td>
      <td>26.400000</td>
      <td>0</td>
      <td>Frolicher-Stehli</td>
      <td>B</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
      <td>1</td>
      <td>30.000000</td>
      <td>1</td>
      <td>Graham</td>
      <td>B</td>
    </tr>
    <tr>
      <th>484</th>
      <td>485</td>
      <td>1</td>
      <td>1</td>
      <td>Bishop, Mr. Dickinson H</td>
      <td>male</td>
      <td>25.0</td>
      <td>1</td>
      <td>0</td>
      <td>11967</td>
      <td>91.0792</td>
      <td>B49</td>
      <td>C</td>
      <td>2</td>
      <td>45.539600</td>
      <td>0</td>
      <td>Bishop</td>
      <td>B</td>
    </tr>
    <tr>
      <th>291</th>
      <td>292</td>
      <td>1</td>
      <td>1</td>
      <td>Bishop, Mrs. Dickinson H (Helen Walton)</td>
      <td>female</td>
      <td>19.0</td>
      <td>1</td>
      <td>0</td>
      <td>11967</td>
      <td>91.0792</td>
      <td>B49</td>
      <td>C</td>
      <td>2</td>
      <td>45.539600</td>
      <td>0</td>
      <td>Bishop</td>
      <td>B</td>
    </tr>
    <tr>
      <th>730</th>
      <td>731</td>
      <td>1</td>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>1</td>
      <td>211.337500</td>
      <td>1</td>
      <td>Allen</td>
      <td>B</td>
    </tr>
    <tr>
      <th>689</th>
      <td>690</td>
      <td>1</td>
      <td>1</td>
      <td>Madill, Miss. Georgette Alexandra</td>
      <td>female</td>
      <td>15.0</td>
      <td>0</td>
      <td>1</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>2</td>
      <td>105.668750</td>
      <td>0</td>
      <td>Madill</td>
      <td>B</td>
    </tr>
    <tr>
      <th>632</th>
      <td>633</td>
      <td>1</td>
      <td>1</td>
      <td>Stahelin-Maeglin, Dr. Max</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>13214</td>
      <td>30.5000</td>
      <td>B50</td>
      <td>C</td>
      <td>1</td>
      <td>30.500000</td>
      <td>1</td>
      <td>Stahelin-Maeglin</td>
      <td>B</td>
    </tr>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
      <td>2</td>
      <td>256.164600</td>
      <td>0</td>
      <td>Cardeza</td>
      <td>B</td>
    </tr>
    <tr>
      <th>872</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
      <td>1</td>
      <td>5.000000</td>
      <td>1</td>
      <td>Carlsson</td>
      <td>B</td>
    </tr>
    <tr>
      <th>311</th>
      <td>312</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Emily Borie</td>
      <td>female</td>
      <td>18.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
      <td>5</td>
      <td>52.475000</td>
      <td>0</td>
      <td>Ryerson</td>
      <td>B</td>
    </tr>
    <tr>
      <th>742</th>
      <td>743</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Susan Parker "Suzette"</td>
      <td>female</td>
      <td>21.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
      <td>5</td>
      <td>52.475000</td>
      <td>0</td>
      <td>Ryerson</td>
      <td>B</td>
    </tr>
    <tr>
      <th>118</th>
      <td>119</td>
      <td>0</td>
      <td>1</td>
      <td>Baxter, Mr. Quigg Edmond</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
      <td>2</td>
      <td>123.760400</td>
      <td>0</td>
      <td>Baxter</td>
      <td>B</td>
    </tr>
    <tr>
      <th>299</th>
      <td>300</td>
      <td>1</td>
      <td>1</td>
      <td>Baxter, Mrs. James (Helene DeLaudeniere Chaput)</td>
      <td>female</td>
      <td>50.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
      <td>2</td>
      <td>123.760400</td>
      <td>0</td>
      <td>Baxter</td>
      <td>B</td>
    </tr>
    <tr>
      <th>820</th>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>Hays, Mrs. Charles Melville (Clara Jennings Gr...</td>
      <td>female</td>
      <td>52.0</td>
      <td>1</td>
      <td>1</td>
      <td>12749</td>
      <td>93.5000</td>
      <td>B69</td>
      <td>S</td>
      <td>3</td>
      <td>31.166667</td>
      <td>0</td>
      <td>Hays</td>
      <td>B</td>
    </tr>
    <tr>
      <th>671</th>
      <td>672</td>
      <td>0</td>
      <td>1</td>
      <td>Davidson, Mr. Thornton</td>
      <td>male</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>F.C. 12750</td>
      <td>52.0000</td>
      <td>B71</td>
      <td>S</td>
      <td>2</td>
      <td>26.000000</td>
      <td>0</td>
      <td>Davidson</td>
      <td>B</td>
    </tr>
    <tr>
      <th>520</th>
      <td>521</td>
      <td>1</td>
      <td>1</td>
      <td>Perreault, Miss. Anne</td>
      <td>female</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>12749</td>
      <td>93.5000</td>
      <td>B73</td>
      <td>S</td>
      <td>1</td>
      <td>93.500000</td>
      <td>1</td>
      <td>Perreault</td>
      <td>B</td>
    </tr>
    <tr>
      <th>257</th>
      <td>258</td>
      <td>1</td>
      <td>1</td>
      <td>Cherry, Miss. Gladys</td>
      <td>female</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>86.5000</td>
      <td>B77</td>
      <td>S</td>
      <td>1</td>
      <td>86.500000</td>
      <td>1</td>
      <td>Cherry</td>
      <td>B</td>
    </tr>
    <tr>
      <th>759</th>
      <td>760</td>
      <td>1</td>
      <td>1</td>
      <td>Rothes, the Countess. of (Lucy Noel Martha Dye...</td>
      <td>female</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>86.5000</td>
      <td>B77</td>
      <td>S</td>
      <td>1</td>
      <td>86.500000</td>
      <td>1</td>
      <td>Rothes</td>
      <td>B</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>Spencer, Mrs. William Augustus (Marie Eugenie)</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B78</td>
      <td>C</td>
      <td>2</td>
      <td>73.260400</td>
      <td>0</td>
      <td>Spencer</td>
      <td>B</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>1</td>
      <td>1</td>
      <td>Maioni, Miss. Roberta</td>
      <td>female</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>86.5000</td>
      <td>B79</td>
      <td>S</td>
      <td>1</td>
      <td>86.500000</td>
      <td>1</td>
      <td>Maioni</td>
      <td>B</td>
    </tr>
    <tr>
      <th>195</th>
      <td>196</td>
      <td>1</td>
      <td>1</td>
      <td>Lurette, Miss. Elise</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B80</td>
      <td>C</td>
      <td>1</td>
      <td>146.520800</td>
      <td>1</td>
      <td>Lurette</td>
      <td>B</td>
    </tr>
    <tr>
      <th>789</th>
      <td>790</td>
      <td>0</td>
      <td>1</td>
      <td>Guggenheim, Mr. Benjamin</td>
      <td>male</td>
      <td>46.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17593</td>
      <td>79.2000</td>
      <td>B82 B84</td>
      <td>C</td>
      <td>1</td>
      <td>79.200000</td>
      <td>1</td>
      <td>Guggenheim</td>
      <td>B</td>
    </tr>
    <tr>
      <th>139</th>
      <td>140</td>
      <td>0</td>
      <td>1</td>
      <td>Giglio, Mr. Victor</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17593</td>
      <td>79.2000</td>
      <td>B86</td>
      <td>C</td>
      <td>1</td>
      <td>79.200000</td>
      <td>1</td>
      <td>Giglio</td>
      <td>B</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0000</td>
      <td>B94</td>
      <td>S</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>Harrison</td>
      <td>B</td>
    </tr>
    <tr>
      <th>802</th>
      <td>803</td>
      <td>1</td>
      <td>1</td>
      <td>Carter, Master. William Thornton II</td>
      <td>male</td>
      <td>11.0</td>
      <td>1</td>
      <td>2</td>
      <td>113760</td>
      <td>120.0000</td>
      <td>B96 B98</td>
      <td>S</td>
      <td>4</td>
      <td>30.000000</td>
      <td>0</td>
      <td>Carter</td>
      <td>B</td>
    </tr>
    <tr>
      <th>435</th>
      <td>436</td>
      <td>1</td>
      <td>1</td>
      <td>Carter, Miss. Lucile Polk</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>2</td>
      <td>113760</td>
      <td>120.0000</td>
      <td>B96 B98</td>
      <td>S</td>
      <td>4</td>
      <td>30.000000</td>
      <td>0</td>
      <td>Carter</td>
      <td>B</td>
    </tr>
    <tr>
      <th>390</th>
      <td>391</td>
      <td>1</td>
      <td>1</td>
      <td>Carter, Mr. William Ernest</td>
      <td>male</td>
      <td>36.0</td>
      <td>1</td>
      <td>2</td>
      <td>113760</td>
      <td>120.0000</td>
      <td>B96 B98</td>
      <td>S</td>
      <td>4</td>
      <td>30.000000</td>
      <td>0</td>
      <td>Carter</td>
      <td>B</td>
    </tr>
    <tr>
      <th>763</th>
      <td>764</td>
      <td>1</td>
      <td>1</td>
      <td>Carter, Mrs. William Ernest (Lucile Polk)</td>
      <td>female</td>
      <td>36.0</td>
      <td>1</td>
      <td>2</td>
      <td>113760</td>
      <td>120.0000</td>
      <td>B96 B98</td>
      <td>S</td>
      <td>4</td>
      <td>30.000000</td>
      <td>0</td>
      <td>Carter</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Next, we create a flag indicating whether the rest of the family died. The theory for this feature is if your whole family died, you probably did not survive either.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Create a flag for people where the rest of the family died</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;AllDied&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;AllDied&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;AllSurvived&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;AllSurvived&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">df_train</span> <span class="o">=</span><span class="n">df_train</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>
<span class="n">df_test</span><span class="o">=</span><span class="n">df_test</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="n">df_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]):</span>
    
    <span class="n">name</span> <span class="o">=</span> <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">i</span><span class="p">,</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span>
    <span class="n">survives</span> <span class="o">=</span> <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span><span class="o">==</span><span class="n">name</span><span class="p">,</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">-</span> <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">i</span><span class="p">,</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span>
    <span class="k">if</span> <span class="n">survives</span> <span class="o">&gt;</span> <span class="mi">0</span><span class="p">:</span>
        <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">i</span><span class="p">,</span><span class="s1">&#39;AllDied&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span><span class="o">==</span><span class="n">name</span><span class="p">,</span><span class="s1">&#39;AllDied&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span> 
    <span class="k">elif</span> <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">i</span><span class="p">,</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">:</span> <span class="c1"># only the person survived</span>
        <span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span><span class="o">==</span><span class="n">name</span><span class="p">,</span><span class="s1">&#39;AllDied&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span> 
    <span class="k">if</span> <span class="n">survives</span> <span class="o">==</span> <span class="p">(</span><span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">i</span><span class="p">,</span><span class="s1">&#39;FamSize&#39;</span><span class="p">]</span> <span class="o">-</span> <span class="mi">1</span><span class="p">):</span> <span class="c1"># rest of family survived</span>
        <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">i</span><span class="p">,</span><span class="s1">&#39;AllSurvived&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
        <span class="k">if</span> <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">i</span><span class="p">,</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">:</span> <span class="c1"># The person survived as well</span>
            <span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span><span class="o">==</span><span class="n">name</span><span class="p">,</span><span class="s1">&#39;AllSurvived&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We extract the title of the passangers. This is a popular piece of code for the particular dataset, I don't know who wrote it originally.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">re</span>
<span class="k">def</span> <span class="nf">get_title</span><span class="p">(</span><span class="n">name</span><span class="p">):</span>
    <span class="n">title_search</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="s1">&#39; ([A-Za-z]+)\. &#39;</span><span class="p">,</span> <span class="n">name</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">title_search</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">title_search</span><span class="o">.</span><span class="n">group</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
    <span class="k">return</span> <span class="s2">&quot;&quot;</span>
<span class="n">all_data</span> <span class="o">=</span> <span class="p">[</span><span class="n">df_train</span><span class="p">,</span><span class="n">df_test</span><span class="p">]</span>
<span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">all_data</span><span class="p">:</span>
    <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">get_title</span><span class="p">)</span>

<span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">all_data</span><span class="p">:</span>
    <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">([</span><span class="s1">&#39;Lady&#39;</span><span class="p">,</span> <span class="s1">&#39;Countess&#39;</span><span class="p">,</span><span class="s1">&#39;Capt&#39;</span><span class="p">,</span> <span class="s1">&#39;Col&#39;</span><span class="p">,</span><span class="s1">&#39;Don&#39;</span><span class="p">,</span> <span class="s1">&#39;Dr&#39;</span><span class="p">,</span> <span class="s1">&#39;Major&#39;</span><span class="p">,</span> <span class="s1">&#39;Rev&#39;</span><span class="p">,</span> <span class="s1">&#39;Sir&#39;</span><span class="p">,</span> <span class="s1">&#39;Jonkheer&#39;</span><span class="p">,</span> <span class="s1">&#39;Dona&#39;</span><span class="p">],</span><span class="s1">&#39;Rare&#39;</span><span class="p">)</span>
    <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;Mlle&#39;</span><span class="p">,</span><span class="s1">&#39;Miss&#39;</span><span class="p">)</span>
    <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;Ms&#39;</span><span class="p">,</span><span class="s1">&#39;Miss&#39;</span><span class="p">)</span>
    <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;Mme&#39;</span><span class="p">,</span><span class="s1">&#39;Mrs&#39;</span><span class="p">)</span>
    
<span class="nb">print</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">crosstab</span><span class="p">(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">],</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;----------------------&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">df_train</span><span class="p">[[</span><span class="s1">&#39;Title&#39;</span><span class="p">,</span><span class="s1">&#39;Survived&#39;</span><span class="p">]]</span><span class="o">.</span><span class="n">groupby</span><span class="p">([</span><span class="s1">&#39;Title&#39;</span><span class="p">],</span> <span class="n">as_index</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">())</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Sex     female  male
Title               
Master       0    40
Miss       185     0
Mr           0   517
Mrs        126     0
Rare         3    20
----------------------
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.347826
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The tickets have a letter prefix, which might be useful to tell the different types of ticket. So we extract that.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_train</span><span class="o">.</span><span class="n">Ticket</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[20]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0           A/5 21171
1            PC 17599
2    STON/O2. 3101282
3              113803
4              373450
Name: Ticket, dtype: object</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">get_ticket_type</span><span class="p">(</span><span class="n">ticket</span><span class="p">):</span>
    <span class="n">type_search</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">search</span><span class="p">(</span><span class="s1">&#39;^([A-Za-z]+)&#39;</span><span class="p">,</span> <span class="n">ticket</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">type_search</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">type_search</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="k">return</span> <span class="s2">&quot;&quot;</span>
<span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">all_data</span><span class="p">:</span>
    <span class="n">data</span><span class="p">[</span><span class="s1">&#39;TicketType&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;Ticket&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">get_ticket_type</span><span class="p">)</span>
<span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">all_data</span><span class="p">:</span>
    <span class="n">data</span><span class="p">[</span><span class="s1">&#39;SpecialTicket&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">data</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">&#39;TicketType&#39;</span><span class="p">]</span> <span class="o">!=</span> <span class="s1">&#39;&#39;</span><span class="p">,</span><span class="s1">&#39;SpecialTicket&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
<span class="c1">#df_train[[&#39;SpecialTicket&#39;,&#39;TicketType&#39;]]</span>
<span class="n">df_train</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;temp.csv&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Remember all the missing ages in the dataset? We should fill that in somehow.</p>
<p>This is where I disagree with a lot of solutions out there. Some use a random gaussian generator with the mean and standard deviation of the age distribution to find the missing values.</p>
<p>Some others use decision tree with all the other features to guess the age.</p>
<p>I find the closest feature that we can use to predict the age while avoiding creating correlation with every other features is the Title. So here we assign the median age for each title to the missing values.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Use title to predict age</span>
<span class="n">df_full</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">df_train</span><span class="p">,</span><span class="n">df_test</span><span class="p">])</span>
<span class="n">df_full</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
<span class="n">titles</span> <span class="o">=</span> <span class="n">df_full</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">()</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
<span class="k">for</span> <span class="n">title</span> <span class="ow">in</span> <span class="n">titles</span><span class="p">:</span>
    <span class="n">df_title</span> <span class="o">=</span> <span class="n">df_full</span><span class="p">[</span><span class="n">df_full</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">==</span><span class="n">title</span><span class="p">]</span>
    <span class="n">age</span> <span class="o">=</span> <span class="n">df_title</span><span class="p">[</span><span class="n">df_title</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">notnull</span><span class="p">()][</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">()</span>
    <span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">==</span><span class="n">title</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()),</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">age</span>
    <span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">==</span><span class="n">title</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()),</span><span class="s1">&#39;Age&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">age</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">title</span><span class="p">,</span><span class="s1">&#39;: &#39;</span><span class="p">,</span><span class="n">age</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
Int64Index: 1309 entries, 0 to 417
Data columns (total 23 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   index          1309 non-null   int64  
 1   PassengerId    1309 non-null   int64  
 2   Survived       891 non-null    float64
 3   Pclass         1309 non-null   int64  
 4   Name           1309 non-null   object 
 5   Sex            1309 non-null   object 
 6   Age            1046 non-null   float64
 7   SibSp          1309 non-null   int64  
 8   Parch          1309 non-null   int64  
 9   Ticket         1309 non-null   object 
 10  Fare           1309 non-null   float64
 11  Cabin          1309 non-null   object 
 12  Embarked       1309 non-null   object 
 13  FamSize        1309 non-null   int64  
 14  FarePp         1309 non-null   float64
 15  IsAlone        1309 non-null   int64  
 16  Surname        1309 non-null   object 
 17  CabinClass     1309 non-null   object 
 18  AllDied        1309 non-null   int64  
 19  AllSurvived    1309 non-null   int64  
 20  Title          1309 non-null   object 
 21  TicketType     1309 non-null   object 
 22  SpecialTicket  1309 non-null   int64  
dtypes: float64(4), int64(10), object(9)
memory usage: 245.4+ KB
Mr :  29.0
Mrs :  35.0
Miss :  22.0
Master :  4.0
Rare :  47.5
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Chapter-3:-Assessing-the-features">Chapter 3: Assessing the features</h4><p>We plot the features against the labels to get a feel of what contribute to the survival on the ship.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;TicketType&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[23]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f09303b4ad0&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Title&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[24]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f093041a4d0&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Title&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[25]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f093022aa50&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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KTY2NsC9MtZQfzdu3Kh58+bp6NGjcrlcevDBBzVixAhJJ8PA4XDooosuksViUX5+vubOnStJuuOOOxQSEqKlS5cqJCREs2fP1tdff62jR49qwIABmjZtmqxWq+666y717dtX//73v3XRRRcpLy8vYOPgi+7du6tdu3YqLy/X5s2blZ+f7w3EqVOn6oYbbpB0cvYxduxYbdq0STExMcrKytLcuXP12Wef6fjx4+revbuysrJ08cXmX5uOkDhHAwYMUGFhoX788Ue9+eabGjVqlLZv3+59//PPP9cbb7xx2tnlzUVj+l9aWqqNGzequLhYISEh+vHHHwNYedM50xj8bOHChfrTn/6kUaNGyePxqLq6WpL07LPPyuFwyGaz6dixY3U25QSjhvrbq1cvFRYWymq16uDBgxozZowGDhwoSVq0aJE2btyoVq1aqaamRq1atVJmZqYKCwu1fPly7y/A6dOn67rrrtNf/vIXud1uPfHEE1q1apV+//vfS5L++9//atGiRWrRIvh/ff3rX//SpZdeqp49eyomJkZ2u10Wi0XffPONxo8fr48++sjb9sCBA1q6dKkkacGCBQoLC9PKlSslSXPmzFFeXp4mT55seh+Cf5SDnMVi0bBhw7R27VoVFRVp2bJldX5BDB48uNkGhNS4/sfExMjlcmn69OkaMGCAbrnllkCV3aTONAY/GzBggPLy8rR//37ddNNNio+PlyRdf/31mjZtmm699VYNGTJEMTExZnehURrqb1VVlZ588knt2bNHVqtVP/74o8rKynT11Vfr8ssv15QpUzRo0CANGTJEbdvWfzXS9evXa+vWrVq8eLEkqba2VlFRUd73R44cGfQB8cgjj8jj8Wjfvn3Kzc1VaGiovvrqKz3++OMqLy9XixYtdPDgQR04cEARERGSpFGjRnk/v379etXU1GjdunWSpGPHjqlnz54B6Utwj/R5YsyYMfrd736n/v3769JLL63zXiCmh2bztf9hYWFau3atSkpKtHHjRj333HN68803vT8k57OGxuBn48ePV0JCgj799FPNnDlTN910kyZPnqzc3Fxt27ZNmzZt0t13362srCzdfPPNJvegcYz6m5WVpYSEBOXm5spisSgxMVFHjx6V1WrVP/7xD33xxRfatGmTxowZo1dffbXeX3wej0cLFiwwDMs2bdr4rV9N5YUXXlD37t1VXFysKVOmaN26dXrssceUnp6uoUOHyu12Kz4+XkePHvV+5pf98ng8yszM9G6OCiRCognExMRo8uTJ3r8MLzS+9r+qqkpWq1WDBw/WTTfdpH/+85/at29fswgJX8agrKxMl19+uTp37qw2bdpo9erVOnHihPbv36/evXurd+/e2rt3r3bs2BH0IWHU3+rqanXs2FEWi0WffPKJ9uzZI0mqqanRkSNH1L9/f/Xv319ffvmldu7cqZ49e+riiy9WTU2N9w+KhIQE5eXlKSsrS1arVVVVVTp8+HDQz7DqM2zYMBUXFysvL0/V1dXq1KmTJGnlypUNHtGVkJCgJUuWqG/fvt7Nc+Xl5QHZV0VINJEL/UKEvvTf6XRqxowZOnHihFwulwYPHqw+ffqYUJ05zjQGS5cuVUlJiVq2bKnQ0FA99dRTcrvdSk9PV3V1tSwWi2w2mx5//HGTKj439fX38ccfV3Z2thYuXKgePXqoR48ekk6GxMMPP6za2lp5PB716tVLv/3tbyVJ9957r+6++261atVKS5cu1ZNPPqk5c+YoOTlZFotFLVu21JNPPnlehoR0ckzGjBmjGTNmaOLEiYqKilL//v11ySWXGH5mwoQJys3N1bhx42SxWGSxWDRp0qSAhAQX+AMAGOI8CQCAIUICAGCIkAAAGCIkAACGCAkAgCFCAvCDESNGqKSkxPD9u+6667y/OiguDJwnAZyFvn37ep//9NNPCg0NldVqlSRlZ2dr7dq13vd/vtDhc889Z3qdwLkiJICzsGXLFu/zhIQEPfPMM7rxxhsDWBHgH2xuAvzg52s0ffTRR3rllVdUXFysvn376vbbb6+3/cqVKzVs2DBdd911uu+++/Ttt9+aXDFQP0IC8KPBgwfrgQce0LBhw7Rlyxa9/fbbp7V577339Morryg3N1cbN27Utddee95cmgPNHyEBBNjy5cs1YcIExcbGqkWLFnrwwQe1Y8cOZhMICuyTAAJs//79mjVrlnJycryveTwelZeXq2PHjgGsDCAkAL+zWCwNvm+z2fTggw8a7q8AAonNTYCfhYeH69tvv5Xb7a73/TvuuEN5eXnauXOnpJP3ZCguLjazRMAQIQH4WVJSkqSTty8dPXr0ae/fdtttuv/++/XYY4/pmmuukd1ur3PvYyCQuJ8EAMAQMwkAgCFCAgBgiJAAABgiJAAAhggJAIAhQgIAYIiQAAAYIiQAAIYICQCAof8HTn2CDW2tnJoAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Pclass&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f09301acb90&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Sex&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[27]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f09301147d0&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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449r3bp1Wr16tX7++Wc9++yzat++vQ4cOKAHHnhAR44c0dGjR3XjjTdq6tSp1X7mvHnztHTpUoVCIcXFxemxxx6T0+m0+JvhbDFjxgxt2rRJBQUFWrRokR566CE9/fTTOnLkiCRp8uTJ6tevn/bu3avRo0frlltu0erVq1VWVqann35ab7zxhr766iudd955ysnJkdPp1I4dO5SVlaWffvpJR48e1S233KLbb7+9yrqPHTumZ555Rp9//rnKy8vVsWNHZWZm6sILmevtpAycEzp27Gjk5uYahmEYeXl5Rvfu3Y2VK1cahmEYc+fONR588EHDMAyjrKzMKC0tNQzDMI4dO2bcdtttxqpVqwzDMIznnnvOeOqppwzDMIz33nvPePjhh41QKGQYhmG8/vrrxgMPPGDlV8JZaNy4ccaKFSuMH3/80UhKSjIKCwsNwzCMwsJCo0+fPsaPP/5ofPfdd0bHjh3D2+e8efOMHj16GH6/3zAMw8jIyDBmzZplGIZhHD582Dh69KhhGIZRWlpqDBkyxPjPf/5jGIZhTJs2zXjttdcMwzCMOXPmGHPmzAnXkZ2dHf4M1IyexDnkv138zp07S5L69esnSerSpYv++c9/Sjo+r1Z2drY2b94swzB08OBBbd++XX379q30WStWrNC2bds0cuTI8HLR0cyQishs3rxZe/fuVWpqavg1m82m3bt3q3Xr1rrgggvC22fnzp3Vpk0bud3u8PN169ZJksrKypSZmakdO3bIZrOpqKhI27dvV/v27Sutb8WKFSotLdXSpUslHe9ZXHHFFRZ807MfIXEOad68uSQpKipKzZo1C78eFRWln3/+WZI0f/58HTp0SG+99ZaaN2+uRx55REePHq3yWYZhaMKECRozZow1xaNRMQxDnTp10uuvv17lvb1791bZPk98/t970kjSrFmz5HQ69dRTT6lJkya64447TLfXjIwMXX/99XXwbRo3zm5CJYcPH5bT6VTz5s1VWFio5cuXV9tuwIABWrRokX788UdJx/fMtm/fbmWpOItdffXV2r17tz777LPwa1u2bJFxirMEHT58WG3atFGTJk3073//Wxs3bqy23YABA7RgwQKVlZVJkkpLS5Wfn1/7L3AOoSeBSm677Tbdd999GjFihNq0aWO65zVixAiVlJRo3Lhxko7vqf3ud7+jC4+I/OpXv1JOTo5mzpypGTNmqLy8XBdffLFefPHFU/qcCRMmaOrUqXr//fd1ySWX6Nprr6223V133aXnn39eY8aMkc1mk81m06RJk6oMS6EqJvgDAJhiuAkAYIqQAACYIiQAAKYICQCAKUICAGCKkAAAmCIkgNO0ceNGjR07Vj169FCvXr00duxYbdmypb7LAs4ILqYDTkNpaanuueceZWZmasiQISovL9fGjRsrTSMBnM3oSQCnoaCgQJLk8Xhkt9t13nnnqXfv3uErz99++20NGTJE1157re68807t27dPkjR37lzdcsst4TmzFi1apGHDhlU77xBQnwgJ4DRcdtllstvtmjZtmlatWhWey0o6fhe2v//973r++ef16aefqkePHnrwwQclSePHj1fTpk31wgsvaNeuXXrmmWc0c+bM8CSMQEPBtBzAacrPz9e8efO0bt06HTx4UH379tXjjz+utLQ0JSYm6uabb5YkVVRU6Oqrr1ZeXp7atm2rvXv3atSoUXI4HBoxYoTuvvvuev4mQFWEBHAG5efna8qUKUpISND27dsVCARkt9vD7x87dkwLFizQNddcI0m69957tWrVKq1bt477caBBIiSAMyw3N1dvvvmmYmNjlZSUpJtuuqnadqtWrdL06dPVuXNnxcXF6dFHH7W4UuDkOCYBnIb8/Hy98sor2r9/vyQpEAjI5/OpW7duGjt2rObOnaudO3dKOn7vgw8//FCSVFxcrOnTp+uJJ57QU089pRUrVmjVqlX19j0AM5wCC5yG6OhoffXVV5o/f74OHz6sFi1aqH///po6daqio6N15MgRPfDAA9q3b59atGihG264QUOGDFF6eroGDBigG2+8UZL0xBNPaPr06VqyZIlat25dz98K+B+GmwAAphhuAgCYIiQAAKYICQCAKUICAGCKkAAAmCIkAACmCAkAgClCAgBgipAAAJj6fxKUtsDHkxoXAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Age and sex </span>
<span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">nrows</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span><span class="mi">8</span><span class="p">))</span>
<span class="n">sb</span><span class="o">.</span><span class="n">kdeplot</span><span class="p">(</span><span class="n">df_train</span><span class="p">[(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;male&#39;</span><span class="p">)][</span><span class="s1">&#39;Age&#39;</span><span class="p">],</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Male&#39;</span><span class="p">,</span>
    <span class="n">shade</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">ax</span><span class="o">=</span><span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">sb</span><span class="o">.</span><span class="n">kdeplot</span><span class="p">(</span><span class="n">df_train</span><span class="p">[(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">1</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;female&#39;</span><span class="p">)][</span><span class="s1">&#39;Age&#39;</span><span class="p">],</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Female&#39;</span><span class="p">,</span>
    <span class="n">shade</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">ax</span><span class="o">=</span><span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Survived&#39;</span><span class="p">)</span>

<span class="n">sb</span><span class="o">.</span><span class="n">kdeplot</span><span class="p">(</span><span class="n">df_train</span><span class="p">[(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">0</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;male&#39;</span><span class="p">)][</span><span class="s1">&#39;Age&#39;</span><span class="p">],</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Die, Male&#39;</span><span class="p">,</span>
    <span class="n">shade</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">ax</span><span class="o">=</span><span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="n">sb</span><span class="o">.</span><span class="n">kdeplot</span><span class="p">(</span><span class="n">df_train</span><span class="p">[(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">==</span><span class="mi">0</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;female&#39;</span><span class="p">)][</span><span class="s1">&#39;Age&#39;</span><span class="p">],</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Die, Female&#39;</span><span class="p">,</span>
    <span class="n">shade</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">ax</span><span class="o">=</span><span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s1">&#39;Die&#39;</span><span class="p">)</span>  
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[28]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Text(0.5, 1.0, &#39;Die&#39;)</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAAHkCAYAAAAQDHkbAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOzdeXhU133w8e+9d2a0jdbRSBotCMQqsxNsh4BxbAuEbWEREoJLHddNasc1tdO0Tk3Sviypm9Z+G6cxNWnixPHrOk5c4sYEQTDBK+CYxcZsEpvQgqTRNtJon+3e+/4xSEYI0MxoGS3n8zw8j2bmnDvnMstv7ll+R9J1XUcQBEEQgiSHuwGCIAjC6CQCiCAIghASEUAEQRCEkIgAIgiCIIREBBBBEAQhJCKACIIgCCERAUQQhtjGjRt54YUXBv24W7du5cknnxz04wpCoAzhboAghMvRo0f593//d86fP4+iKOTk5PC9732POXPmDOrzfP/73x/U4wnCSCECiDAutbe38+ijj7J582buvvtuvF4vR48exWQyBXUcXdfRdR1ZFhfzwvgj3vXCuFRWVgZAQUEBiqIQGRnJkiVLmDFjRp+uoaqqKqZPn47P5wPga1/7Gj/60Y+4//77mTt3Lv/1X//F6tWrex3/5Zdf5tFHHwVgw4YN/OhHPwLg7rvv5t133+0p5/P5uPXWWzl9+jQAn376Kffffz8LFy7kvvvu49ChQz1lL126xAMPPMD8+fP5y7/8S5qbm4fgf0YQAicCiDAuTZo0CUVReOqpp3j//fdpaWkJqv6OHTv453/+Zz755BO+9rWvUVZWRnl5ec/jO3fuZOXKlX3q3XvvvRQVFfXcPnDgAImJicycOZO6ujq++c1v8td//dccPnyYp556iieeeIKmpiYAnnzySWbOnMmhQ4d47LHH+N3vfhfayQvCIBEBRBiXzGYzr732GpIk8X/+z/9h0aJFPProozQ2NgZU/0tf+hJTp07FYDAQGxvLXXfd1RMYysvLuXjxInfeeWefeitXruSdd96hq6sL8AeagoICwB+Uli5dyu23344syyxevJhZs2bx/vvvU1NTw8mTJ/nWt76FyWTi5ptvvubxBWE4iQAijFuTJ0/m3/7t3/jggw/YuXMn9fX1/OAHPwiors1m63V75cqV7Nq1C4CioiLy8vKIiorqUy87O5vJkyfz7rvv0tXVxTvvvNNzpVJTU8OePXtYuHBhz7+PP/6YhoYG6uvriYuLIzo6uudY6enpoZ66IAwKMYguCPiDyerVq3n99de56aabcLlcPY9d66pEkqRetxcvXkxzczMlJSUUFRXx3e9+97rPVVBQQFFREZqmMWXKFLKzswF/UCosLOTpp5/uU6e6uprW1lY6Ozt7gkhNTU2fdgjCcBJXIMK4VFpayksvvURtbS0AdrudoqIi5s6dS25uLkeOHKGmpoa2tjZ++tOf9ns8g8FAfn4+zz77LC0tLSxevPi6Ze+55x4OHjzIr3/9657uK4D77ruPd999l/3796OqKm63m0OHDlFbW0tGRgazZs1i69ateDwejh492mswXhDCQQQQYVwym80cP36cNWvWMG/ePL761a8ybdo0NmzYwOLFi7nnnnu47777WL16NXfccUdAx1y5ciUffvghK1aswGC4/sV9SkoK8+bN49ixY9xzzz0999tsNrZt28ZPf/pTFi1axO23384vfvELNE0D4Ic//CHHjx/n1ltv5YUXXmDVqlUD+08QhAGSxIZSgiAIQijEFYggCIIQEhFABEEQhJCIACIIgiCERAQQQRAEISQigAiCIAghGRMLCZubO9C0sTeZzGIx43C0h7sZQ0ac3+gmzm/0sljMg3KcMRFANE0fkwEEGLPn1U2c3+gmzm98E11YgiAIQkhEABEEQRBCIgKIIAiCEJIxMQYiCMLYpqo+mpsb8Pk8w/ac9fVyTx6y0cpgMJGYaEVRhuarXgQQQRBGvObmBiIjo4mJSRu2FPYGg4zPN3oDiK7rdHS00tzcQHKyrf8KIRBdWIIgjHg+n4eYmDix/0kQJEkiJiZuSK/aAroCKSsrY8OGDTidThISEnjmmWeYOHFirzKqqvL000+zf/9+JEnikUceYc2aNQC88cYbvPzyy8iy/5JwzZo1PPjggwBs3bqV1157jZSUFAAWLFjApk2bBvEUhdFC13V8FcfwfPJ7FNt0Im5diySL3ziCnwgewRvq/7OAAsimTZtYt24dhYWF7Nixg40bN/LKK6/0KrNz504qKyvZu3cvTqeTVatWsWjRIjIzM8nPz2f16tVIkkR7ezsrV67klltuYcaMGQCsWrWKp556avDPThhVXO+/hO/cfqSoOLwn30JvdxB5xyNI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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;SibSp&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[29]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f09308c6190&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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YtN3t9tTZmamduzYoePHjwc7Sos++ugjffHFF0pJSVFycrKOHDmiSZMmqbS0NNjR/Pa//0+7dOkip9Opjz/+OMiJ/Ldnzx5VVVVp6NChkqShQ4eqW7du2r9/v6k5Qvf/NpP4OzkhAueZZ57R7t279fzzz6tLly7BjtMqp0+flsfjqX++efNmXXXVVYqKigpiKv888MADKi0t1ebNm7V582b16tVLK1eu1LBhw4IdzS9nzpypP57v8/lUUlIiu90e5FT+69Wrl44cOaIvvvhC0vn5744dO6a+ffuamoO5m/xgNDlhqJg/f742btyoY8eOqWfPnoqKitLbb78d7Fh+2bdvnxwOhxISEtS1a1dJUu/evfX8888HOZl/jh07pilTpuibb75ReHi4rrrqKs2cOVM33nhjsKO1WnJysl588UVdf/31wY7ily+//FLTpk2T1+tVXV2dEhMT9ac//UmxsbHBjua3devWacWKFfVfGpg+fbpSU1NNzUBJAAAMcbgJAGCIkgAAGKIkAACGKAkAgCFKAgBgiJIA2mjdunWaOHFi/fP+/fvr4MGDQUwEtB/mbgL8tGvXLv3lL3/Rvn37ZLFYdN1112n27NkaM2aMxowZ49c6vvvuOy1atEglJSU6deqUevbsqdTUVM2ePTvA6YGLQ0kAfqitrdVDDz2k3Nxc/fKXv9S5c+e0a9euVl8Bvnz5cu3evVvFxcWKjY3VV199pV27dgUoNdB2HG4C/FBRUSHp/FTxFotFXbt21bBhw3TDDTfob3/7m3796183GL9161alpKTo1ltvVV5enurq6iRJZWVlSk1NVVxcnMLCwtS7d29lZmbWL5ecnKxly5Zp9OjR+slPfqLHH39c3377rXlvFLgAJQH44dprr5XFYtHMmTO1detWff31182Of/fdd7VmzRq9+eab2rx5s9asWSNJGjx4sF5++WW98sor2rt3r5qa8GD9+vVauXKl3n33XVVUVCg/Pz8g7wnwByUB+KFHjx4qKipSWFiYnnjiCd1222166KGHdOzYsSbHu1wuRUVFKT4+Xr/5zW/qJ4h88MEH5XK5tH79et11110aPny43nzzzQbLTpgwQTabTVFRUZo8eXLIzLOFSxMlAfgpMTFRTz31lD744AOtX79eVVVVWrBgQZNjfziV/NVXX62qqipJ529aNWHCBL322mvatWuXJk+erNmzZzeY/vmHy8bHx9cvCwQDJQFchMTERI0bN0779u1r8uc/nB788OHDTc482rVrV02YMEGRkZH6z3/+06plAbNQEoAf9u/fr1WrVunIkSOSzn+Qu91uDR48uMnxK1eu1Ndffy2Px6PVq1dr9OjRkqSXX35ZO3bs0NmzZ/X999/rzTff1OnTpzVgwID6ZYuKinTkyBGdOHGi/iQ2ECx8BRbwQ48ePfTPf/5TBQUFOnXqlCIiIjRy5Eg99thj2rhxY6PxKSkpGjdunGprazV27FiNHz9e0vm9h//dZS8sLEwJCQlaunRpg3tIOxwOTZw4UVVVVUpJSdHkyZNNe5/AhbifBNCBJCcna/78+br99tuDHQWQxOEmAEAzKAkAgCEONwEADLEnAQAwREkAAAxREgAAQ5QEAMAQJQEAMERJAAAM/T9qTVPXjjpF/QAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Parch&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[30]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f09303ef250&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYkAAAEMCAYAAAAxoErWAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAa/UlEQVR4nO3deXDU9eHG8SdZDHIkhsQcG64UUGe9QksUoaZgiA0D0eDVOKu0RQ0MKKAtYKiYcCg2OKNSDhEVkIKOjbQoMUqKIILVqAVLaDwYDAZ1ybFAueTa3d8fjPsjkE+yIdndLLxfM8zsd/ez3312SfLs9w7zeDweAQDQgPBgBwAAtF2UBADAiJIAABhREgAAI0oCAGBESQAAjCgJAIBRu2AHaG379h2W282hHwDgi/DwMHXp0sn4+HlXEm63h5IAgFbC6iYAgBElAQAwoiQAAEaUBADAKCAlUVhYqPT0dF1xxRX6+uuvGxzjcrk0Y8YMZWRk6Oabb1ZRUVEgogEAGhGQkhgyZIhWrlyprl27GsesWbNGVVVVKi0t1euvv6558+bpu+++C0Q8AIBBQEoiNTVVVqu10TElJSW66667FB4erpiYGGVkZOjdd98NRDzAb+z2DoqPj/T+s9s7BDsS0Cxt5jgJh8OhpKQk77TVatWePXuaPZ/Y2M6tGQtokYiIM6fbKS4uMjhhgHPQZkqitTidhziYDm3GsmVSfPypUqipOShJqq0NYiDgDOHhYY1+uW4zezdZrVb98MMP3mmHw6HExMQgJgIAtJmSGDp0qIqKiuR2u7V3716tW7dOmZmZwY4FABe0gJTEE088oV/96lfas2ePRo0apeHDh0uScnNzVV5eLknKzs5Wt27d9Otf/1q/+c1v9OCDD6p79+6BiAcAMAjzeDzn1Qp8tkmgrTlzmwTQloTMNgkAQNtDSQAAjCgJAIARJQEAMKIkAABGlAQAwIiSAAAYURIAACNKAgBgREkAAIwoCQCAESUBADCiJAAARpQEAMCIkgAAGFESAAAjSgIAYERJAACMKAkAgBElAQAwoiQAAEaUBADAiJIAABhREgAAI0oCAGBESQAAjCgJAIARJQEAMKIkAABGlAQAwIiSAAAYURIAAKN2gXqhyspK5eXlaf/+/YqOjlZhYaGSk5PrjXE6nZo6daocDodOnDihG264QdOmTVO7dgGLCQA4TcCWJAoKCmS327V27VrZ7Xbl5+efNWbRokXq3bu31qxZozVr1ui///2vSktLAxURAHCGgJSE0+lURUWFsrKyJElZWVmqqKjQ3r17640LCwvT4cOH5Xa7dfz4cZ04cUIJCQmBiAgAaEBASsLhcCghIUEWi0WSZLFYFB8fL4fDUW/cuHHjVFlZqRtvvNH7r1+/foGICABoQJta2f/uu+/qiiuu0CuvvKLDhw8rNzdX7777roYOHerzPGJjO/sxIXDu4uIigx0BaLaAlITValV1dbVcLpcsFotcLpdqampktVrrjVuxYoVmz56t8PBwRUZGKj09XWVlZc0qCafzkNxuT2u/BaAFTpVDbe3BIOcAzhYeHtbol+uArG6KjY2VzWZTcXGxJKm4uFg2m00xMTH1xnXr1k0ffPCBJOn48eP66KOPdNlllwUiIgCgAWEejycgX7t37typvLw8HThwQFFRUSosLFSvXr2Um5urCRMm6JprrlFVVZUKCgpUV1cnl8ul/v3767HHHmvWLrAsSaCtiY8/tSRRU8OSBNqeppYkAlYSgUJJoK2hJNCWtYnVTQCA0ERJAACMKAkAgBElAQAwoiQAAEaUBADAiJIAABhREgAAI0oCAGBESQAAjCgJAIARJQEAMKIkAABGlAQAwIiSAAAYURIAACNKAgBgREkAAIwoCQCAESUBADCiJAAARpQEAMCIkgAAGFESAAAjSgIAYERJAACMKIkLgN3eQfHxkYqPj5Td3iHYcQCEEEoCAGAU5vF4PKYHJ0+erLCwsCZnMmfOnFYN1RJO5yG53ca3dMGKj4+UJNXUHAxykgsPnz3asvDwMMXGdjY/3tiTe/bsqR49eqhHjx6KjIzUunXr5HK5lJiYKLfbrffee09RUVGtHhoA0Da0a+zBhx56yHv7/vvv1+LFi5Wamuq977PPPtPzzz/vv3QAgKDyeZvE559/rpSUlHr3paSkaOvWra0eCgDQNvhcEldeeaWeeeYZHT16VJJ09OhRPfvss7LZbD49v7KyUjk5OcrMzFROTo527drV4LiSkhLdcsstysrK0i233KK6ujpfI+I8dPqeWeydBQReo6ubTvfUU09p0qRJSk1NVVRUlA4cOKCrr75aTz/9tE/PLygokN1uV3Z2tt58803l5+dr+fLl9caUl5dr/vz5euWVVxQXF6eDBw8qIiKiee8IANBqGt27qSEOh0M1NTWKi4tTUlKST89xOp3KzMxUWVmZLBaLXC6X+vfvr9LSUsXExHjH/fGPf9SAAQN05513Nu9d1Hst9m5qSKjvYRPK+UM5O85/Te3d5POShCTt27dPZWVlqq2tVW5urqqrq+XxeJSYmNjo8xwOhxISEmSxWCRJFotF8fHxcjgc9Upi586d6tatm+655x4dOXJEN998s8aOHevTbrhAa4qLiwyJedbWUjyNsds7aN26U3/mMjJO6tVXfwxyotDjc0l88sknGj9+vK6++mpt2bJFubm5+vbbb7VkyRItWrSoVcK4XC599dVXWrp0qY4fP64HHnhASUlJGjFihM/zaKwR4Z8/VIEU6vlbG59H405fWx0R0Y7P6xz4XBKzZ8/Wc889pwEDBui6666TdGrvpm3btjX5XKvVqurqarlcLu/qppqaGlmt1nrjkpKSNHToUEVERCgiIkJDhgzRtm3bmlUSrG4yOfXLEbrfPAObP1T+mITu/2dgLFv2/6v7li07qNra4OZpi1ptddP333+vAQMGSJJ39c9FF10kl8vV5HNjY2Nls9lUXFys7OxsFRcXy2az1VvVJElZWVnauHGjsrOzdfLkSX388cfKzMz0NSLgFyW/HdXCOSxtpfmcMmz50laZD+ALn3eB7d27tzZt2lTvvn/961+6/PLLfXr+9OnTtWLFCmVmZmrFihWaMWOGJCk3N1fl5eWSpOHDhys2NlbDhg3TiBEj1KdPnxZtxAYAtIzPSxJ5eXkaM2aMBg8erKNHjyo/P1/r16/XwoULfXp+7969VVRUdNb9L774ovd2eHi4pk6dqqlTp/oaCwDgRz4vSfTt21dvvfWW+vTpozvuuEPdunXTG2+8oWuvvdaf+QAAQeTzksQXX3whm82m3Nxcf+YBALQhPpfEqFGjFBMT4z1dRvfu3f2ZCwDQBvhcEh9++KE2bdrk3UPpsssuU1ZWloYNG6bY2Fh/ZgQABInPJWGxWDR48GDvhuv33ntPr732mgoLC7V9+3Z/ZrxghcJRv+ynD5zfmn350mPHjmnDhg0qKSnR9u3b611fAgBwfvF5SWLjxo1as2aN1q9frz59+mjYsGGaPn264uLi/JkPABBEPpdEYWGhhg8frtWrV6tHjx7+zIQG/HvOAy2cw0utNB+p35SXWjwPAKHB55IoKSnxZw4AQBvUaEk8//zzGjt2rCRp7ty5xnETJ05s3VQAgDah0ZLYs2dPg7cBABeGRkvip5PwSacuXwoAuLD4vAvsuHHj9M477+jYsWP+zAMAaEN8Lonrr79eL7/8sgYOHKhHH31UmzZtktvt9mc2AECQ+VwSv//97/XGG29o1apV6t69u2bPnq20tDQ98cQT/swHAAgin3eB/UlycrIeeughZWRkaM6cOVq5cqWmTZvmj2wIcZxWBAh9zSqJqqoqFRcX6+2339a+ffuUmZmpcePG+SsbACDIfC6JO+64Q7t27dKQIUM0ZcoU3XjjjbJYLP7MBgAIMp9KwuPxKCMjQyNHjlTnzp39nQnnod8vbekBl3NbaT7SslHmA0MB1OfThuuwsDC98MIL6tixo7/zAADaEJ/3brLZbKqsrPRnFgBAG+PzNonrr79eubm5uu2225SYmKiwsDDvY3feeadfwgEAgsvnktiyZYu6du2qTz75pN79YWFhlEQbN3HphHq35476SxDTAAglPpfEX//6V3/mAAC0QT6XRGOn4AgPb/ZVUBFALDkAOFc+l8SVV15ZbzvE6b744otWCwQAaDt8Lon33nuv3nRtba0WL16sm266qdVDAQDaBp9LomvXrmdNFxYW6s4779Rdd93V6sEAAMHXoo0Jhw4d0t69e1srCwCgjfF5SWLy5Mn1tkkcPXpUn376qW699Va/BAMABJ/PJdGzZ8960x07dtTdd9+tgQMHtnooAEDb0GRJbN++XREREXrooYckSU6nU7Nnz9aOHTvUt29fpaSkqFOnTn4PCgAIvCa3ScyePVt1dXXe6ccff1zffvutcnJytGPHDj399NN+DQgACJ4mS2Lnzp1KTU2VJB04cEAbN27U008/rXvuuUfPPPOMNmzY4NMLVVZWKicnR5mZmcrJydGuXbuMY7/55hulpKSosLDQt3cBAPCLJkvC5XLpoosukiR9/vnniouL089+9jNJktVq1YEDB3x6oYKCAtntdq1du1Z2u135+fnG1ysoKFBGRoav7wEA4CdNlkSfPn30zjvvSJJKSko0YMAA72PV1dWKjGz6msNOp1MVFRXKysqSJGVlZamioqLB3WcXL16swYMHKzk52df3AADwkyY3XE+aNEljx47V9OnTFR4erldffdX7WElJiX7xi180+SIOh0MJCQne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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;TicketType&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[31]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0918e340d0&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Embarked&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[32]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0918e3fc90&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;IsAlone&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[33]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0918c375d0&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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Qk8//bSuXLkSng8MfA0lAbRg1apVGjVqlA4dOqR33nlHM2bMCDz36aefqrKyUo888ogmTZqk7du36/PPP2/2ff7yl7+orKxMGzZs0M6dO3Xt2jU999xzQWP+8Y9/6K233tJrr72ml156SadPn5akwGtKSkq0d+9e3XnnnU1eC4QLJQG0ICYmRhcuXFBtba26dOmi1NTUwHM7duxQ586dNWrUKI0fP14+n0979uxp9n22bdum2bNnq0+fPrrjjjv0y1/+UhUVFfryyy8DY+bNm6fbbrtNgwYN0qBBg3Ty5ElJ0qZNm7RgwQIlJiaqc+fOmjdvniorK4NeC4QLJQG0YOHChfL7/Zo6daoefvhhbd68OfBceXm5fvjDHyomJkadO3fWgw8+qLKysmbfp7a2NuiWm71799aXX34pr9cbmPetb30rMN21a1ddu3ZNknThwgU99dRTSk1NVWpqqjIyMhQdHR30WiBcYiIdAOjI4uLitGTJEknS4cOH9dhjj+l73/ueunTpor///e86evSoduzYIemr3U+ff/656urqmlxaPT4+XufPnw88vnDhgmJiYmS323Xx4sUWMyQmJqqoqEj3339/O386oHVsSQAtePPNNwNf4nfeeaeioqIUHR2trVu3Kjk5WW+99ZbKy8tVXl6uyspKJSQkNHvpapfLpddee00ffPCBPvnkE61cuTKwFdKaH//4x3rhhRcCJVNXV6edO3e27wcFDNiSAFpw7NgxFRUVqb6+Xna7Xb/5zW/Up08flZWVafr06YqLiwsaP23aNJWVlWnmzJlB86dMmaKamhrNmDFDn332mUaPHq3FixeHlGHWrFny+/2aM2eOamtrZbfblZGRobS0tHb7nIAJ95MAABixuwkAYERJAACMKAkAgBElAQAwoiQAAEaUBADAiJIAABhREgAAI0oCAGD0f/2bbJpv9speAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;AllDied&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s1">&#39;Sex&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[34]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0918baba10&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYkAAAEPCAYAAAC3NDh4AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAfEElEQVR4nO3de0BUdf7/8dfMIOY1AwFBLQtLKRU32S5qN8WFNYysDBcr04LSzLbdUqsNUMsid1MzrTQ1XXTXpVZcCYufZaa24frNEiOyzMsaIyh4Q0RwmN8fbhTCgUGZi8zz8Zdz5jNz3keGefH5fM75HJPdbrcLAIA6mN1dAADAcxESAABDhAQAwBAhAQAwREgAAAwREgAAQy4JidTUVA0aNEg9evTQzp0762xjs9k0depURUZGasiQIUpPT3dFaQCAergkJAYPHqzly5erc+fOhm3WrFmjffv2KTs7WytXrtTcuXO1f/9+V5QHADDgkpCIiIhQcHBwvW2ysrI0YsQImc1m+fn5KTIyUh988IErygMAGPCYOQmr1aqQkJDqx8HBwTpw4IAbKwIA+Li7gKZ2+PAJVVWx0ggAOMJsNumSS9oYPu8xIREcHKyCggL16dNHUu2ehaOqquyEBAA0EY8ZboqOjlZ6erqqqqpUUlKidevWKSoqyt1lAYBXc0lIvPDCC7r55pt14MABjRkzRrfffrskKSEhQbm5uZKk2NhYdenSRb/5zW9077336rHHHlPXrl1dUR4AwICpuS0VXlxcWmu4yW636/Dhg6qoKJfUrA7XKSwWH7Vt20GtWhmPUwJoHsxmk/z92xo+7zFzEs5UWnpUJpNJQUFdZDJ5zAibR7Lb7aqsrNCRIwcliaAAvJxXfGOePFmqdu06EBAOMJlM8vVtqQ4dAlRaesTd5QBwM6/41qyqssli8YpOU5Np0cJXNttpd5cBwM285pvTZDK5u4QLijv+v+LjW2nduvo/kpGRp7VixUkXVQTAK3oSAIBz4zU9CXdZseKvWrt2jcxms+x2uxITx2vgwFvcXZZHOruHEBjYTpJUVHTcHeUAECHhVF9/vUMff/z/9Pbbf1XLli1VVnZCR44wGQzgwsFwkxMdOlSk9u3by9fXV5LUunUbhYR01rFjR5Wc/IwSEh7QmDHx2rjxE0nS66/P1ptvvi5J2rEjVw88EKdTp065q3wAoCfhTL/+9Q1aunSRfve7u3TttREaOPAW9e8/UHPm/EV33HGX+vX7tY4dO6bExNHq1+/XSkwcr4SE0erXL0Jz5vxFf/rTNLVs2dLdhwHAixESTtS6dWstXLhMO3bkatu2rXr11VTt3JmvnJx/a9eu76vbVVRUyGq1KjS0u557LlmJiQ/qgQfGqmfPMDdWDwCEhNNZLBaFh/dVeHhfXXfdDXr55emqqqrS/Plvq3Xr1rXa//DDLrVr116HDh10Q7UAUBNzEk60b98e7d27p/rxd9/tVKdOIbrhhv76xz9WVG//9tt8SVJRUaEWL16gt99ept27d+mzzza5umQAqIGehBOVlZ3UnDkzdezYMbVo4Ss/Pz899dQzatu2rWbNmqnRo0fKZqtSUFAn/fnPczRjxlQlJo5XUFAnPffcVD311ERdc00vXXxxB3cfCgAv5RWrwB44sFedOl3mpoouXO7+f+M6CcD5GloFluEmAIAhQgIAYIiQAAAYIiQAAIY4uwl18vExy2w2KyCgnbtL8YgafnLwIJPo8C5eFxKu+MLhiwRAc8Fw0wVq0aK39Prrs91dBoBmzut6Emi8/3vlYTft+W037/+MfpPeduv+AXfy6pCIn7S8yd5rxSujHG47cGCEEhLGaePGDTp69KgmT35OW7duUU7OZzp9+rSmT09Vt26Xq7j4kFJSntOJEydUUVGh/v0HaPz4J+p8z+XLl+qTTz6SzWZTx46Bmjz5Ofn7d2yqwwPgpRhucpO2bdvp7beXady4x/XMM39Unz59tWTJCkVH365lyxZXt0lNnaXFi9P0zjsrlJ//jT7//LNa7/Xhh1nav3+/3nrrHS1evFw33jiAoSgATcKrexLuNHjwbyRJPXr0lGRS//4D//c4TBs2rJek/60WO0e5udsl2VVcXKzvvtupG27oX+O9Nm36VPn532js2PskSTbbabVta3yZPQA4ipBwk5/uVmc2m+Xr26J6u9lsls1mkyStXLlcx48f04IF76hly5ZKTX1RFRW171Rnt9s1evRYxcTEuqZ4AF7Dq0OiMfMI7nD8+HH5+3dUy5YtdfBgkTZt2qA777y7VruBA29WevrfdfPNt6l9+/aqqKjQ3r17dOWVV7mhagDNiVeHhKcbMWKknn9+ssaMiVdgYJD69ft1ne2io2/X0aNH9PjjiZLODFMNHz6CkABw3rxuqXAupnOMj49ZBQV7dc01V7vtFNSIyWdOPd2a6jmnwDaHny3wSw0tFe51PQl+yQHAcZwCCwAwREgAAAwREgAAQ143JwHP9cSSidqc36fW9p8msCVpQM/tmjPmNVeWBXg1rwsJzm4CAMe5LCR2796tKVOm6MiRI+rQoYNSU1PVrVu3Gm2Ki4v1zDPPyGq1qrKyUjfccIP+9Kc/ycfH67LMK9FDADyPy+YkkpOTFR8frw8//FDx8fFKSkqq1ebNN99UaGio1qxZozVr1ujrr79Wdna2q0p0mU8//USjRt2jMWPitW/fHqfu68UXU/Teeyudug8AzZdL/kQvLi5WXl6elixZIkmKiYnR9OnTVVJSIj8/v+p2JpNJJ06cUFVVlSoqKlRZWamgoCCn1dWUF4k15p4Dq1f/Uw899KgGDYpssv0DgDO4JCSsVquCgoJksVgkSRaLRYGBgbJarTVCYvz48Xr88cc1cOBAnTx5UqNGjVK/fv0ata+6rhwsKjLLx8d1J3LVt6/Zs/+s7du36b//3auMjHc1fvzjmj9/rk6cKJUkJSaO04ABN6mgoEBjxtyn2Njh+vzzz3Tq1ClNnfqC/vnP9/T117lq2fIizZz5qvz9O+r777/TzJkv6eTJclVUnNKdd96lkSPPrEtlMplkNpvk42NWZWWl3nzzdW3b9oUqKysVGtpdkyY9q9atW7vk/6U58KT7bQOu4FGD/R988IF69OihpUuX6sSJE0pISNAHH3yg6Ohoh9+jrmU5qqqqdPp0VVOXa6i+fU2Y8Afl5+frd7+7X3369NXEiY9o5szX1LFjRx06dEgJCQ9o2bKVstmqdPToEfXqFa7ExMe0YsUyTZjwqObOfUuTJj2nP//5Za1c+XclJo5XYGAnzZo1X76+viorK1Ni4mhFRNygbt0ul91uV1WVXadPV2np0iVq1aqNFixYKkmaP/81LVmySI888litOl0ZqhcSTkpAc+MRy3IEBwersLBQNptNFotFNptNRUVFCg4OrtEuLS1NM2bMkNlsVrt27TRo0CDl5OQ0KiQuJDt2fCWrtUBPPTWxepvJZNKPP/5XF1/cQa1ata6+z8RVV/VUQECgrryyhySpZ8+e+s9/ciRJ5eXlev31l/X99ztlMpl16NBBff/9TnXrdnmN/W3e/KlOnDihTz75WJJUWVmh7t2vdMWhArhAuSQk/P39FRYWpszMTMXGxiozM1NhYWE1hpokqUuXLvr000/Vp08fVVRU6N///reGDBniihLdwm6XQkOv1Lx5C2s9Z7UW1LrPhK9vy188tlTfd+Ktt+bJz89fixcvl4+Pj5588jFVVFTUub8//nGK4WqyAHA2lw03paSkaMqUKZo/f77at2+v1NRUSVJCQoImTpyo3r1769lnn1VycrKGDRsmm82m66+/Xvfee6/TanL3De579eqj/fv36YsvturaayMkSd9887V69ry6Ue9TWnpcoaFXysfHRz/88L2++upLDRlSu/c1cODNWrlyuXr16q2WLS9SWdkJFRUV1epxAMBPXBYSoaGhSk9Pr7V94cKf/4q+9NJLq8+A8gbt27fXyy+/qnnz5mjOnL/o9OlKhYR0VmrqrEa9z+jRD2n69CRlZ69V586d1bfvr+psd999D2rRorf08MMPyGw2SzJp7NgEQgKAIe4n4QTNYXLTE+4n4Sm4nwSaM4+YuPYk/JIDgOM4zxEAYMhrQqKZjao5nd1eJZPJ3VUAcDevGG7y8fHViRPH1KZNe5n45quX3W6XzXZaR48eVZs2bdxdDuAR4uNbad26hr8uIyNPa8WKky6oyHW8IiQuuSRAhw8fVGnpEXeXckEwmy0KCPBXx44d3V0KADfzipCwWHzUsWNwww1RjTWKgJ/V1TsIDDzzO1JU1LxPhvGaOQkAQOMREgAAQ4QEAMAQIQEAMERIAAAMERIAAEOEBADAECEBADBESAAADBESAABDhAQAwJBXrN0EoPnwtHXFPKUeZ91QjZ4EAMAQIQEAMMRwE4AL1v+98rAb9/6222voN+ltp++DngQAwBAhAQAwREgAAAwREgAAQ4QEAMAQIQEAMERIAAAMERIAAEOEBADAEFdcA0ADnlgyUZvz+9T5XMTkn696HtBzu+aMec1VZbkEPQkAgCF6EgDQgObWO2gMl/Ukdu/erbi4OEVFRSkuLk579uyps11WVpaGDRummJgYDRs2TIcOHXJViQCAs7isJ5GcnKz4+HjFxsZq9erVSkpK0rJly2q0yc3N1euvv66lS5cqICBAx48fl6+vr6tKBACcpd6QePrpp2UymRp8k1deeaXe54uLi5WXl6clS5ZIkmJiYjR9+nSVlJTIz8+vut0777yjsWPHKiAgQJLUrp1n3PEJALxVvSFx2WWXVf/78OHDWrVqlW677TZ17txZBQUFWr9+vYYPH97gTqxWq4KCgmSxWCRJFotFgYGBslqtNUJi165d6tKli0aNGqWysjINGTJE48aNcyiofuLv39bhtkBjecqtKoGzOeuzWW9ITJgwofrfDz30kBYsWKCIiIjqbVu3btUbb7zRZMXYbDZ9++23WrJkiSoqKvTwww8rJCREd955p8PvUVxcqqoqe5PV5K34Mqybs+4jDMfx2azbuX42zWZTvX9cOzxx/eWXXyo8PLzGtvDwcG3btq3B1wYHB6uwsFA2m03SmTAoKipScHBwjXYhISGKjo6Wr6+v2rZtq8GDB2v79u2OlggAaGIOh8TVV1+tV199VeXl5ZKk8vJyzZo1S2FhYQ2+1t/fX2FhYcrMzJQkZWZmKiwsrMZQk3RmrmLTpk2y2+2qrKzU559/rp49ezbmeAAATcjhkHjppZe0bds2RUREqH///oqIiNAXX3yhl19+2aHXp6SkKC0tTVFRUUpLS9PUqVMlSQkJCcrNzZUk3X777fL399fQoUN15513qnv37rrnnnvO4bAAAE3BZLfbGzWAb7VaVVRUpICAAIWEhDirrnPGnETT+OW4r3tvNu9+v7zZPHMS7sdn82dN8dlssjkJ6cwZTjk5OdqyZYtCQkJUWFioAwcOnFNhAADP53BIbNmyRdHR0VqzZo3mz58vSdq7d69SUlKcVRsAwM0cDokZM2Zo9uzZWrRokXx8zpw5Gx4eztlHANCMORwSP/74o2688UZJqr64rUWLFtWntQIAmh+HQyI0NFQbN26sse2zzz7TVVdd1eRFAQA8g8ML/E2ZMkWPPPKIbr31VpWXlyspKUkff/xx9fwEAKD5cbgn0bdvX/3rX/9S9+7ddffdd6tLly5699131adP3XdrAgBc+BzuSXzzzTcKCwtTQkKCM+sBAHgQh0NizJgx8vPzq74ZUNeuXZ1ZFwDAAzgcEps3b9bGjRuVmZmp2NhYXXnllYqJidHQoUPl7+/vzBoBAG7icEhYLBbdeuut1RPXH330kf72t78pNTVVO3bscGaNAAA3afQ9rk+dOqX169crKytLO3bsqHF/CQBA8+JwT2LDhg1as2aNPv74Y3Xv3l1Dhw5VSkpK9a1GAQDNj8MhkZqaqttvv10ZGRm69NJLnVkTAMBDOBwSWVlZzqwDAOCB6g2JN954Q+PGjZMkzZkzx7DdE0880bRVAQA8Qr0h8ct7RXDfCADwPvWGxE+3GJXO3L4UAOBdHD4Fdvz48Vq7dq1OnTrlzHoAAB7E4ZC47rrrtGjRIvXv31+TJ0/Wxo0bVVVV5czaAABu5nBIPPjgg3r33Xf13nvvqWvXrpoxY4ZuuukmvfDCC86sDwDgRo2+4rpbt26aMGGCZs2apR49emj58uXOqAsA4AEcvk5Ckvbt26fMzEy9//77Onz4sKKiojR+/Hhn1QYAcDOHQ+Luu+/Wnj17NHjwYE2aNEkDBw6UxWJxZm0AADdzKCTsdrsiIyN1//33q23bts6uCQDgIRyakzCZTHrrrbfUunVrZ9cDAPAgDk9ch4WFaffu3c6sBQDgYRyek7juuuuUkJCg4cOHq1OnTjKZTNXP3XPPPU4pDgDgXg6HxBdffKHOnTtry5YtNbabTCZC4jzEx7fSunX1/xgiI09rxYqTLqoIAH7mcEj89a9/dWYdAAAP5HBI1LcEh9nc6Gvy8D9n9xACA9tJkoqKjrujHACoweGQuPrqq2vMQ/zSN99802QFAQA8h8Mh8dFHH9V4fPDgQS1YsEC33XZbkxcFAPAMDodE586daz1OTU3VPffcoxEjRjR5YQAA9zuvyYTS0lKVlJQ41Hb37t2Ki4tTVFSU4uLitGfPHsO2P/zwg8LDw5Wamno+5QEAzpPDPYmnn366xpxEeXm5/vOf/+iOO+5w6PXJycmKj49XbGysVq9eraSkJC1btqxWO5vNpuTkZEVGRjpaGgDASRwOicsuu6zG49atW2vkyJHq379/g68tLi5WXl6elixZIkmKiYnR9OnTVVJSIj8/vxptFyxYoFtvvVVlZWUqKytztDwAgBM0GBI7duyQr6+vJkyYIOnMF/6MGTP03XffqW/fvgoPD1ebNm3qfQ+r1aqgoKDqVWMtFosCAwNltVprhER+fr42bdqkZcuWaf78+edzXACAJtBgSMyYMUMTJkzQVVddJUl6/vnnVVRUpLi4OGVmZmrmzJlKSUk570IqKyv1/PPP66WXXjqvJcj9/ZvHKrUBAe3cXQLqwM8FnspZn80GQ2LXrl2KiIiQJB07dkwbNmxQZmamLr/8cg0aNEgjR45sMCSCg4NVWFgom80mi8Uim82moqIiBQcHV7c5ePCg9u3bp8TExOp92e12lZaWavr06Q4fUHFxqaqq7A639zxnftAHD7r3Yjq+DOvm7p8L+GwaOdfPptlsqveP6wZDwmazqUWLFpKkL7/8UgEBAbr88sslnfnyP3bsWINF+Pv7KywsTJmZmYqNjVVmZqbCwsJqDDWFhIQoJyen+vHcuXNVVlamyZMnN/j+AADnaPAU2O7du2vt2rWSpKysLN14443VzxUWFqpdO8dSPSUlRWlpaYqKilJaWpqmTp0qSUpISFBubu651A4AcLIGexJPPfWUxo0bp5SUFJnNZq1YsaL6uaysLF177bUO7Sg0NFTp6em1ti9cuLDO9o8//rhD7wsAcJ4GQyIiIkLr16/Xnj171K1btxq3L73llls0dOhQpxYIAHAfh66TaNu2rXr16lVr+xVXXNHkBQEAPAdrfAMADBESAABDhAQAwBAhAQAw5PACf97Ak67k9KRaAHgvehIAAEOEBADAEMNNBuInLXfTnke5ef9nrHhllFv3D8Az0JMAABgiJAAAhggJAIAhQgIAYIiQAAAYIiQAAIYICQCAIUICAGCIkAAAGCIkAACGCAkAgCFCAgBgiJAAABgiJAAAhggJAIAh7ifhZhveu1UFP3Sutf1vM3++n0PIFT/qlrs/cWFVAHAGPQkAgCF6Em5GDwGAJ6MnAQAwREgAAAwREgAAQ4QEAMAQIQEAMERIAAAMERIAAEMuu05i9+7dmjJlio4cOaIOHTooNTVV3bp1q9Fm3rx5ysrKksVikY+Pj5588knddNNNrioRAHAWl4VEcnKy4uPjFRsbq9WrVyspKUnLli2r0aZPnz4aO3asWrVqpfz8fN13333atGmTLrroIleVCQD4BZcMNxUXFysvL08xMTGSpJiYGOXl5amkpKRGu5tuukmtWrWSJPXo0UN2u11HjhxxRYkAgDq4pCdhtVoVFBQki8UiSbJYLAoMDJTVapWfn1+dr8nIyNCll16qTp06NWpf/v5tz7tewEhAQDt3lwDUyVmfTY9cu2nLli2aM2eOFi9e3OjXFheXqqrKfk775QsADTl48Li7S3Cp+PhWWreu/q+JyMjTWrHipIsq4vfUyLl+Ns1mU71/XLtkuCk4OFiFhYWy2WySJJvNpqKiIgUHB9dqu23bNj399NOaN2+errjiCleUBwAw4JKehL+/v8LCwpSZmanY2FhlZmYqLCys1lDT9u3b9eSTT+q1117TNddc44rSANTj7B5CYOCZv+KLiryrR+XNXHadREpKitLS0hQVFaW0tDRNnTpVkpSQkKDc3FxJ0tSpU1VeXq6kpCTFxsYqNjZW3377ratKBACcxWVzEqGhoUpPT6+1feHChdX/fu+991xVDgDAAVxxDQAwREgAAAwREgAAQx55nQSAmjzt2gBPqwfOQ08CAGCIkAAAGGK4CbjAxE9a7sa9j3J7DSteGeW2fXsjehIAAEOEBADAECEBADDEnAQAQxveu1UFP3Sutf1vM3+eFwi54kfdcvcnLqwKrkRPAgBgiJ4EAEP0EEBPAgBgiJAAABgiJAAAhggJAIAhQgIAYIiQAAAYIiQAAIYICQCAIUICAGCIkAAAGCIkAACGCAkAgCFCAgBgiJAAABgiJAAAhggJAIAhQgIAYIiQAAAYIiQAAIYICQCAIUICAGDIZSGxe/duxcXFKSoqSnFxcdqzZ0+tNjabTVOnTlVkZKSGDBmi9PR0V5UHAKiDy0IiOTlZ8fHx+vDDDxUfH6+kpKRabdasWaN9+/YpOztbK1eu1Ny5c7V//35XlQgAOIvJbrfbnb2T4uJiRUVFKScnRxaLRTabTddff72ys7Pl5+dX3S4xMVF33XWXoqOjJUnTpk1TSEiIHn74YYf3dfjwCVVVndsh+fu3PafXwXsUF5e6Zb98NtGQc/1sms0mXXJJG8Pnfc61oMawWq0KCgqSxWKRJFksFgUGBspqtdYICavVqpCQkOrHwcHBOnDgQKP2Vd/BAueLL2t4Kmd9Npm4BgAYcklIBAcHq7CwUDabTdKZCeqioiIFBwfXaldQUFD92Gq1qlOnTq4oEQBQB5eEhL+/v8LCwpSZmSlJyszMVFhYWI2hJkmKjo5Wenq6qqqqVFJSonXr1ikqKsoVJQIA6uCSiWtJ2rVrl6ZMmaJjx46pffv2Sk1N1RVXXKGEhARNnDhRvXv3ls1m07Rp07R582ZJUkJCguLi4lxRHgCgDi4LCQDAhYeJawCAIUICAGCIkAAAGCIkAACGCAkAgCFCArU4smIv4A6pqakaNGiQevTooZ07d7q7HK9ASKAWR1bsBdxh8ODBWr58uTp37uzuUrwGIYEaiouLlZeXp5iYGElSTEyM8vLyVFJS4ubKACkiIqLWcj5wLkICNdS3Yi8A70NIAAAMERKowdEVewF4B0ICNTi6Yi8A78ACf6jFaMVewN1eeOEFZWdn69ChQ7rkkkvUoUMHvf/+++4uq1kjJAAAhhhuAgAYIiQAAIYICQCAIUICAGCIkAAAGCIkgEaaMmWKZs2aJUnKycnRzTff3KjXJyUlad68eee9b8AVfNxdAODJ7r//fuXn52vz5s3y9fVtsP2gQYN06NAhWSwWWSwWde/eXbGxsYqLi5PZfOZvsmnTpjm7bKDJ0JMADOzfv19bt26VyWTSRx995PDr3nzzTW3btk3r169XQkKCFi5cqOeee86JlQLOQ0gABjIyMhQeHq7hw4crIyOj0a9v166dBg8erNmzZ2vVqlXVN8k5e8ho/fr1io2NVUREhEaOHKn8/Pzq5/Ly8jR8+HD96le/0u9//3udOnXq/A8MaARCAjCwevVqDRs2TMOGDdOmTZt06NChc3qfPn36qFOnTtq6dWut577++ms9++yzmjZtmnJychQXF6fx48eroqJCFRUVeuyxxxQbG6stW7YoOjpa2dnZ53tYQKMQEkAdtm7dqoKCAv32t79Vr1691LVr1+pFD89FYGCgjh49Wmv7P/7xD8XFxSk8PFwWi0XDhw9XixYt9OWXX+qrr75SZWWlRo8erRYtWig6Olq9e/c+n8MCGo2Ja6AOGRkZGjBgQPXqtzExMVq1apUefPDBc3q/wsJCXXzxxbW2FxQUKCMjQ2lpadXbKisrVVRUJJPJpKCgIJlMpurnQkJCzmn/wLkiJICzlJeXa+3ataqqqtKAAQMkSRUVFTp27FiN+QJHbd++XYWFherXr1+t54KDg/Xoo49q3LhxtZ7bsmWLCgsLZbfbq4OioKBAXbt2bXQNwLliuAk4y7p162SxWPT+++8rIyNDGRkZysrKUkRERKMmsEtLS7V+/Xr94Q9/0B133KEePXrUajNixAj9/e9/11dffSW73a6ysjJ98sknKi0tVd++feXj46Nly5bp9OnTys7OVm5ublMeKtAgehLAWVatWqW77rqr1tDOqFGj9OKLL+rGG2+s9/WPPvqoLBaLzGazunfvrjFjxmjkyJF1tu3du7emT5+uadOmae/evbrooot07bXXKiIiQr6+vpo7d66ef/55zZ49W7fccouGDBnSZMcJOIL7SQAADDHcBAAwREgAAAwREgAAQ4QEAMAQIQEAMERIAAAMERIAAEOEBADA0P8HWAJiebXieikAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;AllSurvived&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[35]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0918b34550&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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SXcvl26xWPSjH/1I48eP149//GNJTQ/jXLlyRUOGDNF3330n6e6oZe3atZo9e7ZiYmL0l7/8RUlJSY36eOeddzRv3jxJjQ97PfXUUzp69Ki73XfffacxY8bos88+kySdOXNGs2fPVmxsrKZPn66ioiJ323/961+aM2eOfvKTn+j555/XjRs3Wv2/Av6NkECnsWfPHiUkJCghIUEnT57U9evXm20/cOBAWSwWpaWl6dixY/ryyy/b1Ofrr7+ujz/+WHPnzlVpaWmjlf7y8/OVkJDQZLunn37aff0sSTp58qR69eqlYcOGqaKiQi+99JJefvllnTp1SmlpaVq4cKF7zY/Fixdr2LBhKioq0vz58/X++++3um7g3wgJdArFxcUqLy/XU089peHDh6tfv36NPoTvJzg4WNu2bVNAQICWLl2qsWPHat68eS2Gy/83Y8YMPfLIIwoKClKPHj00ZcoUd79lZWX6/PPP9eSTTzbZLiEhQUeOHNHXX38t6W6Y/HshqD179mjixImaNGmSAgMDNW7cOA0fPlzHjh1TeXm5zp49q1deeUVdu3bVT3/60/u+PuApQgKdQl5ensaNG+e+mq3dbvfoG3Z0dLR+//vf6/jx48rPz1dlZaWys7M97vfeS6wnJCS4L0jncDg0depUdevWrcl2AwYMUHR0tI4ePaqvv/5aR44ccY84ysvL9be//U2xsbHuv7///e+qqqpSZWWlQkJC1L17d/drRUZGelwvcC/ObsIDr66uTvv371d9fb3GjRsnSfrmm29UW1urCxcuePw60dHRSkpK0o4dOyRJ3bp1U11dnfv5+40wAgICGt0fN26cbty4ofPnz8vhcOh3v/udsT+73S6Hw6H6+noNHjxYAwYMkHQ3eBITE7VixYom21y9elW1tbW6ffu2OyjKy8ub1AF4ipEEHniHDh2SxWLRvn37lJeXp7y8PBUUFCg2NrbZCexLly7prbfe0rVr1yTdXdrV4XAoJiZGkmSz2fTRRx+pvLxcX331lTZs2NBiLUFBQYqLi9OqVav05ZdfukPrfuLj41VYWKjt27e7DzVJ0vTp03X06FGdOHFCLpdLd+7cUVFRka5du6a+fftq+PDhys3N1TfffKPi4uJGE+BAaxESeOC9//77SkpKUmRkpMLDw91/zz77rPLz891nI90rODhYn3zyiWbNmqVRo0bpF7/4hR599FGlp6dLujsqiI+P1/Tp05WUlKQnnnjCo3oSEhL04Ycfatq0aQoKMg/mIyIiNGrUKJ0+fVrx8fHux61Wq9avX68NGzZo7NixmjRpkjZv3uw+xfaPf/yjPvnkE40ZM0br1q3Tz3/+c0//VUATrCcBADBiJAEAMCIkAABGhAQAwIiQAAAYERIAACNCAgBgREgAAIwICQCAESEBADD6X3pJFqNoDvD+AAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;SpecialTicket&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[36]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0918a9fd90&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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PFgnTpzQ3LlztW/fPs2bN08fffQR003wCiMJNCshISHatm2bLBaLFi1apIEDB2r27Nm6fPmyJCk0NFQ///nP1aJFCyUmJqpHjx567733dPnyZR09elQLFy5Uq1atFBYWpunTp3uON+zYsUMzZ85UTEyMLBaLunXrVufT0aKjo9WvXz8FBwerc+fOSk5O1ocffnhX27Rjxw5NmDBBgwYNUlBQkCIiIhQVFeX5/J///KemTZumX/7yl0pOTpYk7d69W0OGDNHQoUMVFBSkQYMGKTo6WkeOHLmrWnDvYSSBZicqKkrLly+X9O3NChcsWKDs7GzFx8crIiJCFovF0zYyMlJlZWW6ePGivvnmG8XHx3s+c7vdnlukO51Ode3atcG+i4uLtXz5cp09e1bXr1+Xy+XSww8/fFfb43Q6NXToUOPne/bsUdeuXZWQkOBZdvHiRf35z3/W4cOHPcu++eYbPf7443dVC+49hASataioKI0fP17bt29XfHy8SktLVV1d7QkKp9Op4cOHq2PHjmrZsqU++OCDOqeRbDabSkpKGuwvKytLffr00W9/+1uFhITojTfe0P79++9qGxrqe86cOcrPz9f8+fOVm5srq9Uqm82mcePGaenSpXfVN8B0E5qVoqIibd68WZcuXZL0bQg4HA7FxsZKkiorK7VlyxbdunVL+/btU1FRkYYOHarw8HANGjRIy5cvV1VVldxut0pKSnTy5ElJ0sSJE7V582adPXtW1dXVunDhgr744ota/X/11Vdq3bq1WrduraKiIr311lt3vU0TJ07Url27dOLECbndbpWWlqqoqMjzeYsWLbR69Wpdv35dL774otxut8aOHavDhw8rPz9fLpdLN2/eVEFBged7eeCBB/Tvf//7rmtD80dIoFkJCQnR3/72N02aNEn9+vXTz372Mz344INKT0+XJMXExOjChQv60Y9+pJdffllr1qxR+/btJUkrVqzQrVu3lJiYqMcee0xz585VeXm5JGn06NGaPXu25s+fr/79++v555/Xf/7zn1r9p6WlyeFwqH///lq0aJESExPveptiYmK0bNkyZWdn69FHH9WUKVN08eLFGm1atmyp3/3ud6qoqNDChQsVERGh9evX69VXX9XAgQM1dOhQbdq0SW63W5I0bdo07d+/X4899hijDdSL50ngnrFr1y7l5eU1yq974F7BSAIAYERIAACMmG4CABgxkgAAGBESAAAjQgIAYERIAACMCAkAgBEhAQAw+j97Itn+GQYpFAAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;SibSp&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[37]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f0918a072d0&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYkAAAEMCAYAAAAxoErWAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAc6UlEQVR4nO3df3AU9eHG8Sc5QECCITdJuAiIhoqHIFRSHS1YSGhD8UIC0ok9tFPBU6GAX1oVpDUJP4oGOwiiKCDG0og6kcqPMw4gIBI6glStoUFKMYCVI4EEhIAoXu77B2PGkGyyIbm9HLxfM87cXT67+xwT77ndzX42IhAIBAQAQD0iQx0AANB6URIAAEOUBADAECUBADBESQAADFESAABDlAQAwFCbUAdoacePn1Z1NZd+AIAZkZER6tLlSsOfX3IlUV0doCQAoIVwuAkAYIiSAAAYoiQAAIYoCQCAIUoCAGCIkgAAGKIk0Kq53R0UFxdV85/b3SHUkYDLCiUBADAUcandma6iooqL6S5BcXFRkqTy8lMhTgJcWiIjI2S3dzL+uYVZAABhhpIAABiiJAAAhigJAIAhSgIAYIiSAAAYoiQAAIYoCQCAIctKorS0VJmZmUpNTVVmZqYOHDhQ77jCwkKlpaXJ5XIpLS1Nx44dsyoiAOAClt2+NDs7W263W+np6VqzZo2ysrK0YsWKWmOKi4v13HPP6a9//atiY2N16tQptWvXzqqIAIALWLInUVFRoZKSErlcLkmSy+VSSUmJKisra4175ZVXNG7cOMXGxkqSoqKidMUVV1gR8ZL2w0nymCAPQFNYUhI+n0/x8fGy2WySJJvNpri4OPl8vlrj9u/fry+++EJjx47VqFGjtHjxYl1iU0sBQFix7HCTGX6/X3v37lVeXp6+/fZb3X///UpISFBGRobpdTQ0UdXlauNGKSLi+8dtJEWFNE9zxMaGb3YgHFlSEg6HQ2VlZfL7/bLZbPL7/SovL5fD4ag1LiEhQcOHD1e7du3Url07paSk6NNPP21SSTALrJHzH65Hj4brLKrhnh9onVrFLLB2u11Op1Ner1eS5PV65XQ6FRMTU2ucy+VSUVGRAoGAzp07pw8++EA33HCDFREBAPWw7E9gc3JylJ+fr9TUVOXn52vmzJmSJI/Ho+LiYknSnXfeKbvdrhEjRigjI0O9evXSmDFjrIoIALgANx26TIT7TXvCPT/QWrWKw00AgPBESQAADFESAABDlAQAwBAlAQAwREkAAAxREgAAQ5QEAMAQJQEAMERJAAAMURIAAEOUBADAECUBADBESQAADFESAABDreoe16gtGPdzbul1cjtR4NLGngQAwBAlAQAwxOGmMPHPefc3cw0vtdB6pIGPvdTsdQAID+xJAAAMURIAAEOUBADAECUBADBESQAADFn2102lpaWaPn26Tpw4oejoaOXm5qpnz561xixatEgrV65UXFycJOnmm29Wdna2VREBABewrCSys7PldruVnp6uNWvWKCsrSytWrKgzLiMjQ9OmTbMqFgCgAZYcbqqoqFBJSYlcLpckyeVyqaSkRJWVlVZsHgBwkSzZk/D5fIqPj5fNZpMk2Ww2xcXFyefzKSYmptbYt99+W0VFRYqNjdXkyZP14x//uEnbsts7tVhuNC4Y80u1pu0Bl7tWdcX13XffrYceekht27bV9u3bNXHiRBUWFqpLly6m11FRUaXq6kAQU1onHD4QrZvgL8ri7QGXh8jIiAa/XFtyuMnhcKisrEx+v1+S5Pf7VV5eLofDUWtcbGys2rZtK0n66U9/KofDoX379lkREQBQD0tKwm63y+l0yuv1SpK8Xq+cTmedQ01lZWU1j/fs2aMvv/xS1157rRURAQD1sOxwU05OjqZPn67Fixerc+fOys3NlSR5PB5NmTJF/fr10/z58/Xvf/9bkZGRatu2rebNm6fY2FirIgIALmBZSSQmJqqgoKDO68uWLat5/H1xAABaB664BgAYoiQAAIYoicvAw3lT6n0MAI2hJAAAhlrVxXQIjoX3PRvqCADCFCWBoAnGFeMtvU6u4AYaxuEmAIAhSgIAYIjDTbDEb/MebuYaFrbQeqRX7lvY7HUAlwv2JAAAhigJAIAhSgIAYIiSAAAYoiSAIHK7OyguLqrmP7e7Q6gjAU1CSQAADPEnsEAQrVz5teLizl8lXl7O1d0IP+xJAAAMURIAAEOUBADAECUBADBESQAADFESAABDlAQAwFCD10k8+uijioiIaHQl8+bNa3RMaWmppk+frhMnTig6Olq5ubnq2bNnvWM///xzjRo1Sm63W9OmTWt03QCA4GhwT+Kaa65Rjx491KNHD0VFRendd9+V3+9X165dVV1drU2bNqlz586mNpSdnS23263169fL7XYrKyur3nF+v1/Z2dkaNmxY098NAKBFNbgnMWnSpJrH48eP19KlS5WUlFTz2q5du/TCCy80upGKigqVlJQoLy9PkuRyuTR79mxVVlYqJiam1tilS5dqyJAhOnPmjM6cOdOkNwMAaFmmp+X45JNP1L9//1qv9e/fXx9//HGjy/p8PsXHx8tms0mSbDab4uLi5PP5apXEZ599pqKiIq1YsUKLFy82G60Wu73TRS2HixMbGxXqCM1iZf5w/7fC5cl0SfTp00fz58/Xww8/rPbt2+vs2bN69tln5XQ6WyTIuXPn9MQTT+jJJ5+sKZOLUVFRperqQItkCrVw+FA5etR4PqJwz99yoizcFtA0kZERDX65Nl0STz75pB555BElJSWpc+fOOnnypPr27aunn3660WUdDofKysrk9/tls9nk9/tVXl4uh8NRM+bo0aM6dOiQHnjgAUnSyZMnFQgEVFVVpdmzZ5uNCQBoQaZLolu3bnr99dfl8/lUXl6u2NhYJSQkmFrWbrfL6XTK6/UqPT1dXq9XTqez1qGmhIQE7dixo+b5okWLdObMGf66CQght7uD3n33/MfEsGHfaeXKr0OcCFZr0nUSx48f144dO7Rz504lJCSorKxMR44cMbVsTk6O8vPzlZqaqvz8fM2cOVOS5PF4VFxc3PTkAICgM70nsXPnTk2ePFl9+/bVRx99JI/Ho4MHD+rll1/Wiy++2OjyiYmJKigoqPP6smXL6h0/efJks9GCjm9TuFz98H4Y/N5fnkzvScydO1cLFizQ8uXL1abN+Q/M/v3769NPPw1aOABAaJkuiS+//FK33XabJNVchd22bVv5/f7gJGtFfvgNim9TAC4npksiMTFR27Ztq/XaP/7xD11//fUtHgoA0DqYPicxffp0PfjggxoyZIjOnj2rrKwsbd68+aIvegMAtH6m9yQGDBigtWvXqlevXrrrrrvUrVs3vfnmm7rpppuCmQ8AEEKm9yT27Nkjp9Mpj8cTzDwAgFbEdEncd999iomJkcvlUlpamrp37x7MXACAVsB0SWzfvl3btm2ruWr6Rz/6kVwul0aMGCG73R7MjACAEDFdEjabTUOGDKk5cb1p0ya99tprys3N1e7du4OZEQAQIk2+fek333yjLVu2qLCwULt37651fwkAwKXF9J7E1q1btW7d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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;Parch&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[38]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f09189c0a90&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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GanzzjtPDz30kC699NJQR2uxjIwMLVq0KGx2gd2+fbsmTJggj8cjr9ervn37avr06UpISAh1NMu2b9+uP/3pT9q9e7c6dOigBx54QEOGDAlqBkoCAGDE6iYAgBElAQAwoiQAAEaUBADAiJIAABhREkA7MX/+fE2aNCnUMYBGOHcTYFFGRobq6+tls9nUqVMnDRkyRNOnT1eXLl1CHQ0IGJYkgBZYtGiRNm3apH/84x+qqKjQs88+a/m1Pp9PXq83gOmAtkdJAKchMTFR6enp+uabbzR27FhdffXVuvLKKzV27Fjt2LHDP2706NF66qmndNtttyk1NVXbt2/Xli1bNGbMGF111VUaPHiwFi1a5B9/5MgRTZkyRT//+c81cuRIVVRUhOLjAX6UBHAaXC6XPvjgA/Xs2VM33XST1q5dq7Vr16pjx44nnWn0zTff1OzZs7Vx40bFxcVpzJgxSk9P1/r161VWVqZBgwb5x65Zs0YjR47UZ599poyMjOBfhQw4AdskgBa49957ZbPZFB0drSFDhmjy5Mn+c0pJ0rhx4/Tb3/620WtuvPFGXXjhhZKk999/XxdccIHuvPNOSVLHjh2VmprqHztgwAD/uXlycnL06quvBvojAU2iJIAWWLhwoQYPHuy/f+DAAeXn52v9+vX63//+J+nYmV89Ho//QlXHn1be5XKpV69exukff+Wxc889V4cOHdLRo0fVoQO/qggNVjcBrbB48WJVVVXpb3/7mzZu3KilS5dKOraR+ifHX/bTbreruro66DmB00VJAK2wb98+dezYUd26ddPu3bu1YMGCJscPHTpU9fX1euWVV3T48GHt3btX//nPf4KUFmg5SgJohd/97nc6dOiQrr76auXm5io9Pb3J8V27dtXixYu1du1a/fKXv1RWVpbKy8uDlBZoOa4nAQAwYkkCAGBESQAAjCgJAIARJQEAMKIkAABGlAQAwIiSAAAYURIAACNKAgBg9H8QaJdmA+6TxwAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sb</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="s1">&#39;CabinClass&#39;</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">df_train</span><span class="p">,</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">capsize</span> <span class="o">=</span> <span class="o">.</span><span class="mi">05</span><span class="p">,</span>
            <span class="n">errcolor</span><span class="o">=</span><span class="s1">&#39;blue&#39;</span><span class="p">,</span>
            <span class="n">errwidth</span> <span class="o">=</span> <span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[39]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f09188f8a50&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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wsJC2e12Sad3UBcVFclsNp8z7vDhw877NptN7du3d0dEAEAd3FISgYGBslgsys3NlSTl5ubKYrHU2tQkSUOGDFFWVpZqamp09OhRbdq0STExMe6ICACog1t2XEvS/v37NWvWLJWWlqpNmzZKT09XeHi4rFarpkyZohtuuEF2u13z5s3TRx99JEmyWq1KSEhwRzwAQB3cVhIAgMsPO64BAIYoCQCAIUoCAGCIkgAAGHLLN64vF1FRUQoICNBbb70lX19f52NLlixRt27dPJarurpaGRkZysvLk5+fn2pqanTnnXdq2rRpatasmUcyRUVFyd/fX82b/+9bmy+88II6duzokTzS6eW0ZMkS5ebmys/PT35+furUqZOmTJmiLl26uD3Pz8vI399fp06dUpcuXWS1WnXzzTe7PcvZTpw4oX79+mns2LF64oknPB2n1uepsrJSkZGRSklJ8djnu65cktSnTx89/vjjHsszevRoVVVVqbq6WgcOHFDXrl0lST169NDChQsb5TUpibOUl5dr3bp1Gj58uKejOM2ePVuVlZXKzs5Wq1atVF1drddff11VVVUe/Ue0aNEij5bn2WbPnq2KigplZWWpTZs2cjgcWr9+vfbv3++RkpBqL6ONGzdqwoQJWrZsmXr27OmRPD/LyclRr1699Pbbb2vGjBny9/f3aB7pf8vKbrfrvvvu07vvvqvY2FhPx/Kqz/nPZ8b+7rvvNHLkSK1bt67RX5PNTWeZNGmSFi9erKqqKk9HkSQdOHBAmzZt0vz589Wq1elzvzRr1kwJCQlq2dJ7zlPlaT8vpz/96U9q06aNJMnHx0d3332313wh86677tLYsWO1bNkyT0dRdna2Jk6cqG7dumnLli2ejlNLZWWlKisrnb9HeBZrEmeJiIhQRESE1qxZo9/+9reejqP8/Hx16tRJV199taejnGPKlCnO1XCTyaTXX3/dY1m8eTmdqWfPnh7/n/LevXt14sQJ3XbbbSouLlZ2drbzzMue9PPn6dChQ+rXr5/69evn6UiSan/Op0+frv79+3s4kXtREnX4wx/+oAceeECjRo3ydBSv5k2r4Wf7z3/+o2nTpqmiokL9+/fXnDlzPB1JkuQN311du3at4uPj5ePjo7vuukvz589XYWGhQkJCPJrr589TZWWlJk+erJdfflkPPvigRzOdmaupYnNTHcLDw3XnnXcqsyHPPX2RevTooYMHD+rEiROejuLVfl5OpaWlkqQuXbpo3bp1uv/++3Xy5Lnn8feU3bt3O3c2ekJVVZVycnKUnZ2tqKgoxcbGqrq6Wm+88YbHMp2tefPmGjBggD7++GNPR4EoCUOTJ0/W6tWrVVZW5tEcYWFhioqKUnJysvN/dna7XStWrPB4Nm8SFhamQYMGac6cOfrxxx+dj5eXl3swVW2bNm3SmjVrlJSU5NEM4eHhev/997VlyxZt2bJFy5cv9+imwrPV1NTos88+U1hYmKejQGxuMtS+fXvFx8dr+fLlno6ip556Si+88IJGjhypZs2aOQ+B9fQRKWduq5Wk+fPn64YbbvBYnoULFyojI0OjRo2Sn5+f2rRpo+DgYE2YMMFjmaZMmeI8BLZz585aunSpevXq5bE8r7/+uu65555aj910003O/zHfeuutHkr2v89TdXW1unbtqkceecRjWfA/nOAPAGCIzU0AAEOUBADAECUBADBESQAADFESAABDlATggu7du+vgwYN1/uytt97SuHHj3PJagLvxPQk0KTk5OcrMzFRBQYFatmyp6667Tg899JAiIyMv+jmHDRumYcOGuTy+qKhIzz33nN5//32VlZUpJCREsbGxGj9+vAICAi46B9AYWJNAk5GZmakFCxbooYce0kcffaStW7cqMTFRmzdvdluG48ePa+zYsaqsrNSrr76qnTt3KjMzU6WlpTp06JDbcgAucwBNQGlpqaNXr16OvLy8On/+1VdfOcaMGeO45ZZbHL/+9a8daWlpjsrKSufPu3Xr5lixYoUjKirK0bt3b8dTTz3lsNvtDofD4cjOznaMHTu21tjVq1c7Bg8e7IiMjHSkpqY6ampqHA6Hw/Hss8864uLinHPr0q1bN8eBAwccDofDsXXrVkd8fLzjpptuctxxxx2ORYsWOcdVVFQ4pk2b5ujdu7fjlltucYwYMcJRXFzszBQVFeXo1auXY+DAgY5169Zd5JJDU8fmJjQJO3fuVGVlpQYPHlznz319fTV79mxFREToyJEjslqtWr16da2zkL777rvKzs5WeXm5kpKSFB4ertGjR9f5fO+9957Wrl2rkydPasSIERo4cKDuuOMOffLJJxo8eLDzyocX0qJFC6Wnp6tr16769ttvNW7cOFksFkVHR+uNN97QyZMn9d5778nf319ff/21rrrqKpWXl2v+/Plau3atwsPDVVRUxAkicdHY3IQm4fjx47rmmmvk51f330URERHq1auX/Pz81LFjRyUkJOizzz6rNcZqtapt27YKDQ3VAw88oNzcXMPXs1qtatOmjUJDQ9WnTx/t3bvXmSMoKMjl3H369FH37t3l6+ur6667TkOHDtWOHTskSX5+fjp+/LgOHjwok8mkiIgI54WpfH19tW/fPlVUVCg4ONijZ57F5Y01CTQJbdu21bFjx/TTTz/VWRQFBQV66qmntGfPHp06dUp2u13XX399rTFms9l5u0OHDioqKjJ8vTOLoEWLFs4z9rZt21bFxcUu5/7qq6/05z//Wfv27VN1dbWqqqqcFwiKj4/XkSNH9Mc//lGlpaUaNmyYpk6dqoCAAP31r3/V8uXL9cQTT+jmm2/WzJkz1blzZ5dfF/gZaxJoEm666SY1b95cmzZtqvPnqampCg8P14YNG/TFF19o6tSp51wgyGazOW8fPnxYwcHB9c7Rt29fvfvuu6qpqXFp/LRp0zRo0CBt27ZNn3/+ucaOHevM1axZM02aNEl5eXl69dVX9d577+nNN9+UJPXv31+ZmZn68MMPFR4errlz59Y7KyBREmgiWrdurSlTpmjevHnatGmTTp06perqam3btk1PP/20ysrK1LJlS7Vs2VL79+/XmjVrznmOZcuW6cSJE7LZbFq5cqViY2PrnSMpKUllZWWaOXOmvv/+e0lSYWGhFi5c6NwkdaaysjJdffXVat68uXbt2lVrE9enn36qb775Rna7Xa1atZKfn59MJpN++OEHbd68WeXl5fL391dAQIBMJlO9swISm5vQhCQlJSkwMFAZGRmaPn26WrZsqeuvv14PPfSQBg4cqLlz52rZsmWyWCyKjY3Vp59+Wmv+oEGDNGLECJ08eVLDhw+/qMvbtm3bVmvWrNFzzz2nMWPGqLy8XCEhIYqLi1OnTp3OGZ+SkqL09HTNmzdPvXv31t133+28+t4PP/yglJQUFRYWKiAgQLGxsRo2bJiOHj2qzMxMPfbYY/Lx8ZHFYlFKSsrFLTQ0eVxPAgBgiM1NAABDlAQAwBAlAQAwREkAAAxREgAAQ5QEAMAQJQEAMERJAAAMURIAAEP/B/qRxCBh8n8xAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Code the sex column</span>
<span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;male&#39;</span><span class="p">,</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">df_train</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;female&#39;</span><span class="p">,</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;male&#39;</span><span class="p">,</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;female&#39;</span><span class="p">,</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The features that show strong correlation with the labels are:</p>
<p>They are the first candidates to consider including in the training set. Again, it's just an educated guess and it's worthwhile to experiment with different combinations if you have the time.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">_</span> <span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">figsize</span> <span class="o">=</span><span class="p">(</span><span class="mi">14</span><span class="p">,</span> <span class="mi">12</span><span class="p">))</span>
<span class="n">mask</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros_like</span><span class="p">(</span><span class="n">df_train</span><span class="o">.</span><span class="n">corr</span><span class="p">(),</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">bool</span><span class="p">)</span>
<span class="n">mask</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">triu_indices_from</span><span class="p">(</span><span class="n">mask</span><span class="p">)]</span> <span class="o">=</span> <span class="kc">True</span>

<span class="n">sb</span><span class="o">.</span><span class="n">set_style</span><span class="p">(</span><span class="s1">&#39;whitegrid&#39;</span><span class="p">)</span>
<span class="n">sb</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">df_train</span><span class="o">.</span><span class="n">corr</span><span class="p">(),</span>
<span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
<span class="n">linewidths</span><span class="o">=.</span><span class="mi">2</span><span class="p">,</span> 
            <span class="n">ax</span> <span class="o">=</span> <span class="n">ax</span><span class="p">,</span>
            <span class="n">linecolor</span><span class="o">=</span><span class="s1">&#39;white&#39;</span><span class="p">,</span>
            <span class="n">cmap</span> <span class="o">=</span> <span class="s1">&#39;RdBu&#39;</span><span class="p">,</span>
            <span class="n">mask</span> <span class="o">=</span> <span class="n">mask</span><span class="p">,</span>
            <span class="n">fmt</span><span class="o">=</span><span class="s1">&#39;.2g&#39;</span><span class="p">,</span>
            <span class="n">center</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span>
            <span class="c1">#square=True</span>
            <span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[42]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.axes._subplots.AxesSubplot at 0x7f091894c310&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAz0AAALxCAYAAAB2NcgOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOzdeVxU9f7H8dcwArKKCxrugqJ209QsNM0UzRUFzXK7pXXV3MsyTa9LpGammeWamktumbkguN7ylpW5Zd78ZYrhiqISKvs+8/uDHCNQIbFhxvfz8ZjHY+bMZ855z2A9+PD5njMGs9lsRkRERERExE45WDuAiIiIiIjIvaSmR0RERERE7JqaHhERERERsWtqekRERERExK6p6REREREREbumpkdEREREROyamh4REREREbmnpk+fTmBgILVr1yYyMjLfmuzsbEJDQ2nTpg1PPfUU69evL7Ljq+kREREREZF7qnXr1qxevZpKlSrdsiY8PJxz586xa9cu1q1bx5w5c4iOji6S46vpERERERGRe6px48b4+Pjctmbbtm0888wzODg4UKZMGdq0acOOHTuK5PglimQvIiIiIiJyX0lISCAhISHPdk9PTzw9PQu9v5iYGCpWrGh57OPjw6VLl+4q4w1qekREREREbJBTwxetevz3XmzI3Llz82wfNmwYw4cPt0KiW1PT8zeJjU20doQ78vb2KPY5vb09gOL/edrCZwm2kdMWMoJyFiVbyAjKWZRsISMoZ1GyhYxw8/cOyV/fvn3p2rVrnu1/ZcoDOZOdixcvUr9+fSDv5OduqOkREREREZFC+6vL2G6lffv2rF+/nrZt23L9+nW++OILVq9eXST71oUMRERERETknpoyZQotWrTg0qVLvPDCC3Tq1AmAAQMGcPToUQCCg4OpXLkybdu25dlnn2Xo0KFUqVKlSI6vSY+IiIiIiNxT48ePZ/z48Xm2L1682HLfaDQSGhp6T46vSY+IiIiIiNg1TXpERERERGyQwcFo7Qg2Q5MeERERERGxa5r0iIiIiIjYIE16Ck6THhERERERsWtqekRERERExK5peZuIiIiIiA3S8raC06RHRERERETsmiY9IiIiIiI2SJOegtOkR0RERERE7JqaHhERERERsWta3iYiIiIiYoMMRi1vKyhNekRERERExK7ZVdMTHBxMWlpaoV8XGBhIZGTkPUgkIiIiIiLWZlfL28LCwqwdQUREREREihm7mvTUrl2b5ORkIGd688EHH9CjRw8CAwNZtWqVpe7QoUN07tyZ7t27M2XKFMxms+W5U6dO0b9/f55++mm6dOnChg0bANi8eTPPPPMMmZmZmEwm+vbty9q1a//eNygiIiIiIoVmV5OeP0tLS2PdunVER0fTuXNnunbtiqOjIyNHjmTmzJkEBASwbds2Vq5cCUBWVhajRo1ixowZ+Pn5kZSUxNNPP02DBg0ICQnhwIEDvPfee7i7u+Pl5UWvXr2s/A5FRERE5H7loO/pKTC7bno6duwIQOXKlfH09OTSpUtkZmbi4uJCQECApWbixIkAnDlzhqioKF599VXLPjIzMzl16hR+fn5MnDiRbt26kZWVxcaNG//+NyQiIiIiIoVm102Ps7Oz5b7RaCQ7O/u29WazmdKlS9/y3KDY2FhSUlIwGAwkJSXh7u5epHlFRERERKTo2dU5PQXh6+tLWloaBw8eBGDHjh0kJiYCUKNGDUqWLMnmzZst9VFRUSQlJZGRkcHIkSN5/fXXGTZsGCNHjiQrK8sq70FERERExOBgtOrNltj1pCc/Tk5OzJo1i9DQUJydnWnSpAkVK1YEoESJEixcuJC3336bjz/+GJPJRNmyZZk9ezYffPABdevWpVOnTgDs27eP2bNnM2rUKGu+HRERERERuQOD+Y+XLpN7JjY20doR7sjb26PY5/T29gCK/+dpC58l2EZOW8gIylmUbCEjKGdRsoWMoJxFyRYyws3fO4qrUoHjrHr8+N1vW/X4hXHfLW8TEREREZH7i5oeERERERGxa/fdOT0iIiIiIvbA4KD5RUHpkxIREREREbumSY+IiIiIiA2ytctGW5MmPSIiIiIiYtfU9IiIiIiIiF3T8jYRERERERuk5W0Fp0mPiIiIiIjYNTU9IiIiIiJi19T0iIiIiIiIXVPTIyIiIiIidk0XMhARERERsUG6kEHBadIjIiIiIiJ2TZMeEREREREbZDBq0lNQmvSIiIiIiIhdM5jNZrO1Q4iIiIiISOGU6/yOVY//W/gbVj1+YWh529/EqeGL1o5wRxk/LiU2NtHaMW7L29sDwCZyFveMYBs5bSEjKGdRsoWMoJxFyRYygnIWJVvICDd/7yiudCGDgtPyNhERERERsWua9IiIiIiI2CBNegpOkx4REREREbFranpERERERMSuaXmbiIiIiIgNctDytgLTpEdEREREROyamh4REREREbFranpERERERMSuqekRERERERG7pgsZiIiIiIjYIH1PT8Fp0iMiIiIiInZNTY+IiIiIiNg1LW8TEREREbFBWt5WcJr0iIiIiIiIXdOkR0RERETEBmnSU3Ca9NiYwT0C+X71RBL3f8SS0BetHUdEREREpNj7y01PYGAg7du3p0uXLgQFBbF169aizGVV0dHRBAQE3PL52rVrk5yc/Dcmuikm9jrTFoezPOxbqxxfRERERMTW3NXytg8//BB/f3+OHTtGz549adq0KWXKlCmqbFaRlZVl7Qi3tXn3YQAeebA6lSqUtnIaEREREbEWLW8ruCI5p+fBBx/Ezc2NkSNHkpSURGZmJqVLl+btt9+mUqVKxMXF8dprrxEXFwdA06ZNGTduHIcPH2by5MmYTCaysrIYPHgwQUFBJCUlMW3aNE6cOEF6ejoBAQGMHTsWo9HIc889x0MPPcSRI0e4cuUKHTp0YNSoUQD8+uuvjB07ltTUVOrUqcO5c+cYPHgwrVq14sqVK0yZMoWLFy+Snp5Op06dGDRoEJAztXr66afZt28fVapUYciQIbne365du5g1axZeXl60aNGiKD4yERERERH5mxRJ07Nv3z7S09N5//33LZOe9evXM3PmTN5//33Cw8OpWLEiy5cvByA+Ph6AxYsX07dvX0JCQjCbzSQmJgIwbdo0Hn30UaZOnYrJZGLUqFFs2LCBZ599FoCYmBhWr15NcnIybdq0oXv37lSvXp3Ro0fTt29fgoODOXr0qKUeYMyYMQwZMoRHH32UjIwM+vXrR7169WjWrBkAsbGxrFy5EshZ3nZDXFwcEyZMYO3atfj6+rJ48eKi+MhERERERO6KJj0Fd1dNz4gRI3B2dsbd3Z05c+awZ88e1qxZQ0pKSq5lYg8//DDLli1j+vTpPPbYYzRv3hyAgIAAFi1axMWLF2nWrBkPP/wwALt37+ann35i2bJlAKSlpVGhQgXL/tq3b4+DgwMeHh74+flx7tw5ypUrR2RkJJ07dwagXr161K5dG4CUlBQOHDjA1atXLftITk4mKirK0vSEhITk+x6PHDnCgw8+iK+vLwA9evRg5syZd/OxiYiIiIjI36hIzukBuHDhAq++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HICmTZuyceNGAK5du8YXX3zx19/wnzRr1ox+ffvSf+BA2nfshI+PD0MGD7I8P2ToMJZ8/DEAjo6OvP/+LMIjImje4kk2h23h/fdn4ejoeMd9ZWVmMW/efFoGtqZlq0DWfvops9+fRfXq1e0io4iIiIjIrRjMVl47dOHCBSZMmMCFCxcoWbIkJpOJPn360LNnTz7//HPmz59PpUqVCAgIYOXKlezfv5/o6Giefvpp9u/fn2tfFy9epHXr1nTr1o2pU6datteuXZvDhw/j5pZzXsn8+fP54IMPWL16NY0bN7bULV++nA0bNgA5051evXrRp08fEhMTGTduHFFRUVSqVIkKFSrg4eHBmDFjCvw+01KS71xkZSVd3Yp9zpK/nxsUG5to5SS35+3tUewzgm3ktIWMoJxFyRYygnIWJVvICMpZlGwhI+TkLM4ajNtm1eMfebujVY9fGFZveu4Xxb2ZADU9RcmW/mde3HPaQkZQzqJkCxlBOYuSLWQE5SxKtpARin/T0/Df2616/B+ndrhzUTFh9eVtIiIiIiIi91IJawcQEREREZHCs7XLRluTJj0iIiIiImLX1PSIiIiIiIhd0/I2EREREREbZNDytgLTpEdEREREROyamh4REREREbFranpERERERMSuqekRERERERG7pgsZiIiIiIjYIF3IoOA06REREREREbumSY+IiIiIiA1yMGjSU1Ca9IiIiIiIiF1T0yMiIiIiInZNy9tERERERGyQLmRQcJr0iIiIiIiIXTOYzWaztUOIiIiIiEjhNJnyhVWPv298G6sevzC0vO1vkp543doR7sjZw6vY53T28AIgbdsCKye5vZIdBxMbm2jtGHfk7e1R7HPaQkZQzqJkCxlBOYuSLWQE5SxKtpARcnIWZ1reVnBa3iYiIiIiInZNkx4RERERERvkoElPgWnSIyIiIiIidk1Nj4iIiIiI2DU1PSIiIiIiYtfU9IiIiIiIiF3ThQxERERERGyQwaALGRSUJj0iIiIiImLX1PSIiIiIiIhd0/I2EREREREbZND4osD0UYmIiIiIiF3TpEdERERExAY5OOhCBgWlSY+IiIiIiNg1NT0iIiIiImLXtLxNRERERMQGGbS8rcDU9BQDK1evZeknn5Celk6bwFaMHzsGJyenfGuPn4hk0uQpnD59hho1qhM6YTx1avsDsH3nLuZ/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ElOoV+0BPEo688WbA/hsVB8+G9WHuQODAVj7am/qVXvgL2UTERERsXUrVqygdevWeW4rVqzIVRcTE0OFChUsX0liNBopX748MTExueqioqI4f/48ffr0oWvXrsyfPz/XH9Lvhs0vb/uzkJAQLl++TI8ePTAajbi5ubF69eo8NV9++SWdOnWiQoUKPPLII6SnpwOwcuVK9u/fj6OjI05OTowfPx6TycQbb7xBYmIiBoMBHx8fXnvttSLJ2/zxprzw3HP8a9AQ0tNzvgNnyEsDLM8PHvEKjRo0YMCL/XB0dGT2zHd5c8pUPpg7nxrVq/W+KpgAACAASURBVDN75rs4OjoCEHXqNO/PmUtCQiKenh480exxXh46FMi5eMPcBQs5feYsRgcHqlevxgcz36VG9Wp2kTE/zepWp19gY/rP25DzPT0P12RIh5sN1JCPNtHItxL9n3qMjKxspm/6iui4eEoYHajlU465A4IpXypnmWQ5z5tL+dKzsgAo6+Gq5W4iIiJy3+rbty9du3bNs93T0/Mv7S87O5sTJ06wbNkyMjIy6N+/PxUrViQkJORuo2IwF1X7JLeVnnjd2hHuyNnDq9jndPbIudx12rYFVk5yeyU7DiY2tvhf8MDb26PY57SFjKCcRckWMoJyFiVbyAjKWZRsISPk5CzO2sz91qrH/2JYwb7GIy4ujnbt2rF//36MRiPZ2dkEBASwa9cuypQpY6l76aWX6NChg6XJWbx4MTExMUycOPGus+rP1CIiIiIics+ULVuWunXrWk4piYiIoG7durkaHsg51+fbb7/FbDaTmZnJvn37qFOnTpFkUNMjIiIiIiL31JtvvsmqVato164dq1atIjQ0FIABAwZw9OhRADp16kTZsmXp2LEjISEh1KxZk+7duxfJ8e3unB4RERERkfuB0Ya+p8fPz4/169fn2b548WLLfQcHB8aOHcvYsWOL/Pia9IiIiIiIiF3TpEdERERExAbZ0qTH2jTpERERERERu6amR0RERERE7JqWt4mIiIiI2CAtbys4TXpERERERMSuadIjIiIiImKDNOkpOE16RERERETErqnpERERERERu6blbSIiIiIiNkjL2wpOkx4REREREbFranpERERERMSuqekRERERERG7ZjCbzWZrhxARERERkcJ5eul+qx5/w4sBVj1+YehCBiIiIiIiNqiE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kdz2nLkentUxHqt0fieOXb6b1TEeqWIxF27ciMjqGI/l6uqonJkkO2QE5cxM2SEjpOR8kYVFxTy+6Dlytrd9fNEL4j/T07Nr165Uy61bt051306/fv3SrbWysqJfv36pakREREREJHv4zzR6RERERERyEk1ZnXH/2YkMRERERETkv0GNHhERERERydE0vE1EREREJBvS6LaMU0+PiIiIiIjkaOrpERERERHJhszN1NeTUerpERERERGRHE2NHhERERERea7Onz9Pu3btaNKkCe3atePChQsmNQaDgdGjR9OwYUMaNWrE6tWrM+38avSIiIiIiGRDZmZZ+/MkRo4cSYcOHfj555/p0KEDI0aMMKnZtGkTly5dYvv27axatYrp06cTHBycKddKjR4REREREXlubt26xalTp/D19QXA19eXU6dOcfv27VR1W7Zs4c0338Tc3Jw8efLQsGFDtm3blikZNJGBiIiIiEg2ZJacnKXnDw8PJzw83GS9k5MTTk5OxuWQkBAKFCiAhYUFABYWFuTPn5+QkBDy5MmTqq5w4cLG5UKFCnHt2rVMyapGj4iIiIiIPLElS5YwY8YMk/UfffQRffr0yYJE6VOjR0REREREnljHjh1p06aNyfqHe3kgpccmNDQUg8GAhYUFBoOB69evU6hQIZO6q1ev8uqrrwKmPT/PQo0eEREREZHsKDkpS0//72Fs6cmbNy9ly5bF398fPz8//P39KVu2bKqhbQBNmzZl9erVNG7cmLt377Jjxw6WL1+eKVk1kYGIiIiIiDxXo0aNYtmyZTRp0oRly5YxevRoALp168bJkycB8PPzo2jRojRu3Ji33nqL3r17U6xYsUw5v3p6RERERETkufLw8EjzuTvz5s0zvrawsDA2hjKbWXJyFk/7ICIiIiIiTyw2OipLz5/Lzj5Lz/8k1NPzPxIXfvvxRVnMxinPC5/Txill7GdkdEwWJ3k0BztbboRHZ3WMx3J1siPp3KGsjvFI5p7VSDi8PqtjPJZV1dbka/lVVsd4rJubBuP54dqsjvFI52a+zrHLd7M6xmNVLObCjRsRWR3jsVxdHV/4nNkhIyhnZsoOGSElp+QMavSIiIiIiGRDZlk8kUF2ookMREREREQkR1OjR0REREREcjQNbxMRERERyY40vC3D1NMjIiIiIiI5mnp6RERERESyIz15JsPU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+edNZwFQXV1Nb28vN9xwA3a7naqqKrZv347X6+XDDz80nSciV5lEu/hTkt+7777L0aNHAYhGo1RWVrJ48WKWLFlCZ2en4bqLvfjii0ydOhW3243f72fHjh1aJZVL0kqPxF1GRgYbNmwwnXFJ3d3dOBwOAFpaWpg3bx67du3i119/5amnnsLj8RguFJGrSaJd/CnJr6mpKXaIRltbGydOnGD//v389NNPsYMCrCAUCpGWlsbo6GjsvezsbLKzsxkdHWXKlCkG68SKNPRI3LW3t3PXXXeRkZFBXV0dhw4dYu3atZZ4fOzCuxo6OzspKioCYMaMGZa+GVtEklOiXfwpyS8lJSW2T+a7777D7XYzbdo0li1bxhtvvGG47r8efvhhmpubcTqd2Gw2xsfHJ/zf09NjOlEsRkOPxJ3P58PlcnHo0CH27dtHeXk5GzZsoLGx0XQaAIFAgKysLDo6OlizZk3sfaufMCciyUcnjonVRKNRwuEwqampHDx4kJUrV8a+Fw6HDZZN1NzcDEBvb6/hEkkUGnok7s4fvbp//348Hg/3338/u3btMlx1jtfrxe12k5qaSkFBQezRka6uLmbOnGm4TkRExKx7772XiooKsrOzsdlsOJ1O4Nyl3na73XDdxY4cOUJubi7p6ekA/Pnnn5w4cYIbb7zRcJlYjW1cOyclzkpLS6moqGDbtm34fD7y8vK47777+PTTT02nATA4OMjQ0BAOhyP2SFsgECAajWrwERGRq96ePXsIBAKUlJSQk5MDwOHDhxkeHqawsNBw3USlpaV88MEHsUfyQqEQjzzyCH6/33CZWI1WeiTuampq2LFjBx6Ph7y8PPr6+liyZInprJicnJzYD/Hzpk+fbqhGRETEWoqLiy9676abbjJQcmXRaDQ28ACkpaURjUYNFolVaeiRuLvtttvYunVr7PWcOXN4+eWXDRaJiIjIP7FmzZrLHuxTV1f3L9Zc2TXXXMPx48fJy8sD4NixY6SkpBiuEivS0CNx99prr/HMM88wZcoUysvL6e7u5pVXXmHFihWm00REROQy7r77btMJ/5PVq1ezatUqli1bBsDevXste22GmKU9PRJ3DzzwAK2trXz99de0tLSwbt06vF4vLS0tptNEREQkyfzyyy98++23ABQWFjJ79mzDRWJFWumR/5sffviB5cuXM336dN2BIyIikgBqa2sv+/0XXnjhXyr553Jycli0aJFl9x2JNUwyHSDJZ9q0adTU1NDe3s6dd95JJBLRpkIREZEEkJ6e/rf/uru7TeddZO/evZSUlPDss88C8OOPP/L0008brhIr0kqPxN3rr79Oa2srK1euJCsri/7+fh5//HHTWSIiInIFq1evnvD61KlTNDc34/f7seKOiDfffJOmpiYqKysBuOWWWzh27JjhKrEiDT0Sd9dddx0VFRWx17m5ueTm5poLEhERkX8sEonw1Vdf8dFHH9HV1UUkEmHnzp0sWrTIdNol/fUairS0NEMlYmUaeiTuTp48yaZNm+jt7eXs2bOx97/88kuDVSIiInIlGzdupK2tjfz8fB588EHq6upwuVyWHXjsdjtDQ0OxvcMHDhzg2muvNVwlVqShR+Ju/fr1uFwuenp62Lx5Mw0NDcyaNct0loiIiFxBQ0MDTqcTr9fL0qVLASx9GFF1dTWVlZX09/fz2GOP0dfXh8/nM50lFqShR+JueHgYj8dDfX09TqeThQsXTnjcTURERKxp3759fPLJJ9TW1hIMBnG73ZY+jOjWW2+lvr6egwcPAuB0OsnMzDRcJVak09sk7lJTU4FzJ8AMDAwQiUQYGBgwXCUiIiJXkpmZSVlZGX6/ny1bthAMBjlz5gxlZWU0NjaazrukcDjM2NgY4+PjRCIR0zliURp6JK5GRkbIyMggGAzy6KOPUlpaSlFREffcc4/pNBEREfkfOBwOampq+OabbygrK7Pk3tzPPvuM4uJidu/ezXvvvUdJSQlffPGF6SyxINu4Fc8flITU3t7OSy+9hN1uJxQK8dZbbzF79mz++OMP8vPzTeeJiIhIkikuLmbr1q3MnTsXgL6+PqqqqtizZ4/hMrEa7emRuPH5fDQ2NrJgwQK+//57tmzZwvvvv286S0RERJJUVlZWbOABmDNnDlOnTjVYJFalx9skbiZNmsSCBQsAWLp0KadPnzZcJCIiIsmssLAQn8/H4OAgp06d4p133mH58uWMjo4yOjpqOk8sRCs9EjfhcJijR4/GbmwOhUITXs+fP99knoiIiCSZt99+G4C6ujpsNlvsb47a2lpsNhs9PT0m88RCtKdH4uZyhxXYbDZLboAUERGRxBcMBuno6CAvLw+Hw2E6RyxIQ4+IiIiIJJTq6mqefPJJHA4HIyMjrFixgoyMDIaHh1m7di0ej8d0oliM9vSIiIiISELp7u6Orei0tLQwb9482tra8Pv97N6923CdWJGGHhERERFJKJMnT4593dnZSVFREQAzZszAZrOZyhIL09AjIiIiIgknEAhw5swZOjo6WLx4cez9s2fPGqwSq9LpbSIiIiKSULxeL263m9TUVAoKCmInxHZ1dTFz5kzDdWJFOshARERERBLO4OAgQ0NDOByO2CNtgUCAaDSqwUcuoqFHRERERESSmvb0iIiIiIhIUtPQIyIiIiIiSU1Dj4iIiIiIJDUNPSIiIiIiktQ09IiIiIiISFL7D0qXifzYLxSIAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Chapter-4:-Prepare-data-for-training">Chapter 4: Prepare data for training</h4><p>This is the standard step for machine learning, creating dummy variables, converting datatype etc, normalisation if required for certain algorithms.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cat_cols</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Embarked&#39;</span><span class="p">,</span><span class="s1">&#39;Title&#39;</span><span class="p">,</span><span class="s1">&#39;Pclass&#39;</span><span class="p">,</span><span class="s1">&#39;TicketType&#39;</span><span class="p">]</span>
<span class="n">df_train</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">df_train</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">cat_cols</span><span class="p">,</span> <span class="n">drop_first</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="n">df_test</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">df_test</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">cat_cols</span><span class="p">,</span> <span class="n">drop_first</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's remind ourselves what columns are available</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_train</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 891 entries, 0 to 890
Data columns (total 49 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   index             891 non-null    int64  
 1   PassengerId       891 non-null    int64  
 2   Survived          891 non-null    int64  
 3   Name              891 non-null    object 
 4   Sex               891 non-null    int64  
 5   Age               891 non-null    float64
 6   SibSp             891 non-null    int64  
 7   Parch             891 non-null    int64  
 8   Ticket            891 non-null    object 
 9   Fare              891 non-null    float64
 10  Cabin             891 non-null    object 
 11  FamSize           891 non-null    int64  
 12  FarePp            891 non-null    float64
 13  IsAlone           891 non-null    int64  
 14  Surname           891 non-null    object 
 15  CabinClass        891 non-null    object 
 16  AllDied           891 non-null    int64  
 17  AllSurvived       891 non-null    int64  
 18  SpecialTicket     891 non-null    int64  
 19  Embarked_C        891 non-null    uint8  
 20  Embarked_Q        891 non-null    uint8  
 21  Embarked_S        891 non-null    uint8  
 22  Title_Master      891 non-null    uint8  
 23  Title_Miss        891 non-null    uint8  
 24  Title_Mr          891 non-null    uint8  
 25  Title_Mrs         891 non-null    uint8  
 26  Title_Rare        891 non-null    uint8  
 27  Pclass_1          891 non-null    uint8  
 28  Pclass_2          891 non-null    uint8  
 29  Pclass_3          891 non-null    uint8  
 30  TicketType_       891 non-null    uint8  
 31  TicketType_A      891 non-null    uint8  
 32  TicketType_C      891 non-null    uint8  
 33  TicketType_CA     891 non-null    uint8  
 34  TicketType_F      891 non-null    uint8  
 35  TicketType_Fa     891 non-null    uint8  
 36  TicketType_LINE   891 non-null    uint8  
 37  TicketType_P      891 non-null    uint8  
 38  TicketType_PC     891 non-null    uint8  
 39  TicketType_PP     891 non-null    uint8  
 40  TicketType_S      891 non-null    uint8  
 41  TicketType_SC     891 non-null    uint8  
 42  TicketType_SCO    891 non-null    uint8  
 43  TicketType_SO     891 non-null    uint8  
 44  TicketType_SOTON  891 non-null    uint8  
 45  TicketType_STON   891 non-null    uint8  
 46  TicketType_SW     891 non-null    uint8  
 47  TicketType_W      891 non-null    uint8  
 48  TicketType_WE     891 non-null    uint8  
dtypes: float64(3), int64(11), object(5), uint8(30)
memory usage: 158.5+ KB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Drop columms that are not useful</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dropped_cols</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;index&#39;</span><span class="p">,</span><span class="s1">&#39;PassengerId&#39;</span><span class="p">,</span><span class="s1">&#39;Name&#39;</span><span class="p">,</span><span class="s1">&#39;Fare&#39;</span><span class="p">,</span><span class="s1">&#39;Ticket&#39;</span><span class="p">,</span><span class="s1">&#39;Cabin&#39;</span><span class="p">,</span><span class="s1">&#39;Surname&#39;</span><span class="p">,</span><span class="s1">&#39;SpecialTicket&#39;</span><span class="p">,</span><span class="s1">&#39;FamSize&#39;</span><span class="p">,</span><span class="s1">&#39;AllSurvived&#39;</span><span class="p">,</span><span class="s1">&#39;CabinClass&#39;</span><span class="p">]</span>
<span class="n">df_train</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">dropped_cols</span><span class="p">,</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">test_id</span> <span class="o">=</span> <span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;PassengerId&#39;</span><span class="p">]</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">dropped_cols</span><span class="p">,</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Split the data into train and test set</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">df_train</span><span class="o">.</span><span class="n">drop</span><span class="p">([</span><span class="s1">&#39;Survived&#39;</span><span class="p">],</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df_train</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span><span class="n">test_size</span> <span class="o">=</span> <span class="o">.</span><span class="mi">3</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">3</span>
<span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Consider normalisation/standardisation if required by the algorithm</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="kn">import</span> <span class="n">StandardScaler</span>
<span class="n">scaler</span> <span class="o">=</span> <span class="n">StandardScaler</span><span class="p">()</span>
<span class="c1">#X_train_scale = scaler.fit_transform(X_train)</span>
<span class="c1">#X_test_org = X_test</span>
<span class="c1">#X_test = scaler.transform(X_test)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">

<p>
<h4 id="Chapter-1:-Missing-values">Chapter 5: Build a model to predict Titanic survival</h4>

Create a model with XGBoost and plot the error and loss aganist number of estimators</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[48]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">XGBClassifier</span><span class="p">(</span><span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.02</span><span class="p">,</span> <span class="n">gamma</span><span class="o">=</span><span class="mf">0.3</span><span class="p">,</span> <span class="n">n_estimators</span><span class="o">=</span><span class="mi">140</span><span class="p">,</span> <span class="n">objective</span><span class="o">=</span><span class="s1">&#39;binary:logistic&#39;</span><span class="p">,</span>
                    <span class="n">silent</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">nthread</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span><span class="n">min_child_weight</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span><span class="n">max_depth</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
                    <span class="n">colsample_bytree</span><span class="o">=</span> <span class="mf">0.8</span><span class="p">,</span> <span class="n">subsample</span><span class="o">=</span> <span class="mf">0.7</span><span class="p">)</span><span class="c1">#learning_rate=0.05,max_depth=4, n_classifier)</span>

<span class="n">eval_set</span> <span class="o">=</span> <span class="p">[(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">),</span> <span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)]</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">,</span> <span class="n">eval_metric</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;error&quot;</span><span class="p">,</span> <span class="s2">&quot;logloss&quot;</span><span class="p">],</span> <span class="n">eval_set</span><span class="o">=</span><span class="n">eval_set</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>

<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="n">results</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">evals_result</span><span class="p">()</span>
<span class="n">epochs</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">&#39;validation_0&#39;</span><span class="p">][</span><span class="s1">&#39;error&#39;</span><span class="p">])</span>
<span class="n">x_axis</span> <span class="o">=</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">epochs</span><span class="p">)</span>
<span class="c1"># plot log loss</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">()</span>
<span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x_axis</span><span class="p">,</span> <span class="n">results</span><span class="p">[</span><span class="s1">&#39;validation_0&#39;</span><span class="p">][</span><span class="s1">&#39;logloss&#39;</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Train&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x_axis</span><span class="p">,</span> <span class="n">results</span><span class="p">[</span><span class="s1">&#39;validation_1&#39;</span><span class="p">][</span><span class="s1">&#39;logloss&#39;</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Validation&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Log Loss&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;XGBoost Log Loss&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="c1"># plot classification error</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">()</span>
<span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x_axis</span><span class="p">,</span> <span class="n">results</span><span class="p">[</span><span class="s1">&#39;validation_0&#39;</span><span class="p">][</span><span class="s1">&#39;error&#39;</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Train&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">x_axis</span><span class="p">,</span> <span class="n">results</span><span class="p">[</span><span class="s1">&#39;validation_1&#39;</span><span class="p">][</span><span class="s1">&#39;error&#39;</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Validation&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Classification Error&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;XGBoost Classification Error&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[01:39:09] WARNING: /workspace/src/learner.cc:480: 
Parameters: { silent } might not be used.

  This may not be accurate due to some parameters are only used in language bindings but
  passed down to XGBoost core.  Or some parameters are not used but slip through this
  verification. Please open an issue if you find above cases.


</pre>
</div>
</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Print the accuracy on the test (CV) set</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[49]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">accuracy_score</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="c1">#y_pred = y_pred[:,0].tolist()</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="p">[</span><span class="nb">round</span><span class="p">(</span><span class="n">value</span><span class="p">)</span> <span class="k">for</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">y_pred</span><span class="p">]</span>
<span class="c1">#predictions= predictions or X_test_org[&#39;Title_Master&#39;].tolist()</span>
<span class="c1">#predictions = [ x|y for (x,y) in zip(predictions, X_test[&#39;SpecialTicket&#39;].tolist() )]</span>
<span class="c1">#predictions= predictions or X_test_org[&#39;Title_Mrs&#39;].tolist()</span>

<span class="n">accuracy</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy: </span><span class="si">%.2f%%</span><span class="s2">&quot;</span> <span class="o">%</span> <span class="p">(</span><span class="n">accuracy</span> <span class="o">*</span> <span class="mf">100.0</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Accuracy: 80.97%
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>There are many avenue to explore to improve the model. You can:</p>
<ul>
<li>Fine tune the hyper parameter of the model.</li>
<li>Get more data.</li>
<li>Try different algorithms.</li>
<li>Use an ensemble of different algorithms.</li>
<li>Use piecewise prediction, i.e., use a different algorithm on a certain population of the data to account for local variation.</li>
</ul>
<p>However, to avoid blindly doing experiments, it is important that we look at what our current method is getting wrong.</p>
<p>This part export the examples that we got wrong in the test set. It can help with feature selection and tuning.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[50]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_check</span> <span class="o">=</span> <span class="n">X_test</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="n">df_check</span><span class="p">[</span><span class="s1">&#39;Real&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">y_test</span> 
<span class="n">df_check</span><span class="p">[</span><span class="s1">&#39;Pred&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">predictions</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[51]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">mismatch</span> <span class="o">=</span> <span class="n">df_check</span><span class="p">[</span><span class="n">df_check</span><span class="p">[</span><span class="s1">&#39;Real&#39;</span><span class="p">]</span> <span class="o">!=</span> <span class="n">df_check</span><span class="p">[</span><span class="s1">&#39;Pred&#39;</span><span class="p">]]</span>
<span class="n">mismatch</span> <span class="o">=</span> <span class="n">df_train_org</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">mismatch</span><span class="o">.</span><span class="n">index</span><span class="p">]</span>
<span class="n">mismatch</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;wrong.csv&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Finally, we apply the model on the submission set.</p>
<p>Note that since the submission set may not have all the variations in the categorical features (cabin class etc.) so we need to fill in the Null with zeros.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[52]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df_test</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">df_test</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[52]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>FarePp</th>
      <th>IsAlone</th>
      <th>AllDied</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>...</th>
      <th>TicketType_CA</th>
      <th>TicketType_F</th>
      <th>TicketType_LP</th>
      <th>TicketType_PC</th>
      <th>TicketType_PP</th>
      <th>TicketType_S</th>
      <th>TicketType_SC</th>
      <th>TicketType_SOTON</th>
      <th>TicketType_STON</th>
      <th>TicketType_W</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>...</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.636364</td>
      <td>29.663876</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>21.771256</td>
      <td>0.605263</td>
      <td>0.744019</td>
      <td>0.244019</td>
      <td>0.110048</td>
      <td>0.645933</td>
      <td>...</td>
      <td>0.019139</td>
      <td>0.014354</td>
      <td>0.002392</td>
      <td>0.076555</td>
      <td>0.002392</td>
      <td>0.016746</td>
      <td>0.028708</td>
      <td>0.023923</td>
      <td>0.009569</td>
      <td>0.014354</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.481622</td>
      <td>13.003851</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>35.603363</td>
      <td>0.489380</td>
      <td>0.436934</td>
      <td>0.430019</td>
      <td>0.313324</td>
      <td>0.478803</td>
      <td>...</td>
      <td>0.137177</td>
      <td>0.119088</td>
      <td>0.048912</td>
      <td>0.266203</td>
      <td>0.048912</td>
      <td>0.128474</td>
      <td>0.167185</td>
      <td>0.152994</td>
      <td>0.097471</td>
      <td>0.119088</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.634400</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.662500</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>35.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>25.982813</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>262.375000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 32 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Make prediction on test set</span>
<span class="n">df_test</span><span class="p">[</span><span class="s1">&#39;CabinClass_T&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">df_test</span> <span class="o">=</span> <span class="n">df_test</span><span class="o">.</span><span class="n">reindex</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">X</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span>
<span class="n">df_test_np</span> <span class="o">=</span> <span class="n">df_test</span> <span class="c1">#.to_numpy()</span>
<span class="c1">#X_final_test = scaler.fit_transform(df_test_np)</span>
<span class="n">X_final_test</span> <span class="o">=</span> <span class="n">df_test_np</span>
<span class="n">final_pred</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_final_test</span><span class="p">)</span>
<span class="c1">#final_pred = final_pred[:,0].tolist()</span>
<span class="n">final_predictions</span> <span class="o">=</span> <span class="p">[</span><span class="nb">round</span><span class="p">(</span><span class="n">value</span><span class="p">)</span> <span class="k">for</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">final_pred</span><span class="p">]</span>
<span class="c1">#predictions= predictions or X_test_org[&#39;Title_Master&#39;].tolist()</span>
<span class="n">submission_headers</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;PassengerId&#39;</span><span class="p">,</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span>
<span class="n">submissions</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="nb">list</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">test_id</span><span class="p">,</span><span class="n">final_predictions</span><span class="p">)),</span><span class="n">columns</span><span class="o">=</span><span class="n">submission_headers</span><span class="p">)</span>
<span class="n">submissions</span><span class="o">.</span><span class="n">to_csv</span><span class="p">(</span><span class="s1">&#39;submission_final.csv&#39;</span><span class="p">,</span><span class="n">index</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Feature engineering and selection is a very important and powerful step in developing machine learning model.</p>
<p>There are a lot of literature out there about this topic, and unfortunately it's rare that we have the time to explore them all for every problem.</p>
<p>This notebook got me into the top 5% of the leaderboard. Hopefully, you have found this useful in your journey of learning and practicing machine learning!</p>

</div>
</div>
</div>
    </div>
  </div>
</body>

 


</html>
