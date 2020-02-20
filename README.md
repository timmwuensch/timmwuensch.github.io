This is my knowledge glossar about topics on Machine Learning and Data Science. I use it for myself to recapture different topics, dicscriptions, solutions and technologies. 

#### Table of contents
1. [Machine Learning](#machine-learning)
    1.1 [Decision Tree](#decision-tree)
    1.2 [Random Forest](#random-forest)
2. [Data Science](#data-science)
    2.1 [Similarity Measures](#similarity-measures)
    2.2 [Recommendation Systems](#recommendation-systems)
    2.3 [Dimensionality Reduction](#dimensionality-reduction)

# Machine Learning
Machine Learning in general is the ability of computer systems to learn patterns and to generalize data. Based on this patterns, it is possible to make decisions and forecast future behavior for unknown data.

Infact Machnine Learning is a collection of various statistical and mathemetical methods to recognize patterns in a dataset. 

Here we differ between three types of Machine Learning problems, based on their learning method (suppervised or unsupervised) and the intention of the Machine Learning system.

![Overview of Machine Learning Categories](images/ml_overview.JPG "Machine Learning Overview")

**Classification** is a type of problem where the system predicts **discrete outputs**, like a certain number of classes. Based on its Supervised Learning strategy, the input data has to be pre-labeled.

**Regression** is a type of problem where the system predicts **continuous outputs**, like a certain value of interest. The input data has to be labeles as well, that means it requires a record of data including the value. A common example is the prediction of temperature on the basis of collected weather data.

**Clustering** is a type of problem where the system finds **unknown patterns and groups** in a dataset. It is a kind of explorative data analysis. A common example is categorization of fitting sizes into groups like small, medium, large or extra large. 

## Decision Tree
A Decision Tree is often used for classification and regression problems. It is a structured and directed presentation of certain decision rules. 

The tree consists of different nodes, branches and leafs. To define the nodes, esp. the root node, you have to find a feature that seperates the dataset the best. To find that feature, we make use of the so-called **Gini Impurity**. The lower the Gini Impurity value is, the better the feature seperates the data. 

In this example we want to create a Decision Tree from a small dataset. It should classify datapoints that consists of gender, weight and age into its status of obesity. 

![Decision Tree Impurity Calculation](images/decision_tree_impurity.JPG "Decision Tree Impurity Calculation")

The dataset has three feature and one label column. To build a Decision Tree, we have to find a feature with the lowest Gini Impurity for our data record. The first feature (*male*) is of Boolean Type and easy to handle. You just have to answer the question: How much is obesity dependent from gender? In the figure above, you can see a simple Gini Impurity calculation.

![Impurity Calculation for numeric values](images/decision_tree_impurity_weight.JPG "Decision Tree Impurity Calculation for numeric values")

To calculate the Gini Impurity of a non boolean feature like *weight* we have to do a couple of more steps. Firstly, sort the patients by ascending weight. Secondly, calculate the average weight for every adjacent patients. Finally, calculate the Gini Impurity for every average weight and take the lowest to define the Gini Impurity of the feature.

When it comes to ranked or classified data, you have to calculate the Gini Impurity for each combination of classes or rank intervals. 

```python
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=1)
dt.fit(train_X, train_y)
```

## Random Forest
A Random Forest is an ensemble of numerous Decision Trees which are randomly build. They are generated from different subsets of the original data which is also called Bootstraped Data. Instead of the entire dataset, these subsets consist of randomly selected features and data rows. The diffenrent subsets lead to *n* different Decision Trees which are combined in ensemble structure of the Random Forest. 

When inserting new data into the Random Forest, it runs the data through every single Decision Tree and takes the **major decision** as output. 

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=1)
rf.fit(train_X, train_y)
```

## Support Vector Machines
## k-Nearest Neighbor
## k-Means Clustering
## Deep Learning
## Convolutional Neural Networks
## Reinforcement Learning
## Natural Language Processing



# Data Science
The challenge behind Data Science is to turn data into information. These information could be used to support business processes with advanced analytics, predictions and decisions. The tasks of a Data Scientist partly overlaps with Machine Learning, but there are still differences between both field of research. 

The work of a Data Scientist could be divided into three segments. 

![Data Science Overview](images/data_science.JPG "Data Science Overview")

The first segment refers to the variety of **Data Sources and Technologies**. Here you have to identify valid data sources (databases or datastreams) and convert data into a usable structure for further analytics (Data Ware House). In this context Data Scientists make use of ETL-Pipelines (Extract-Transform-Load) to transform and convert data.

The second segment refers to so-called **Data Frameworks** (e.g. Pandas or Apache Spark). These frameworks or libraries provide data structures and methods to handle the data. The selection of an adequate Data Framework is often based on the amount, size and type of data. 

The third segment refers to variety of **analytical, statistical and mathematical methods** which are used by Data Scientists. These methods can range from statistical basics, over regression models to Machine Learning Models. 

## Similarity Measures
An important task in Data Science research is comparing different datapoints with each other to explore similarities and correlations. Here are some major measurement techniques:

### Euclidean Distance (L2-Norm)
The Euclidean Distance is a basic measurement of similarity. For the sake of simplicity, it is the length of the line segment connecting two points in n-dimensional euclidean space. See the following formula to calculate the Euclidean Distance between two points:

![Formula of Euclidean Distance Calculation](images/euclidean_formula.JPG "Formula of Euclidean Distance Calculation")

The lower the Euclidean Distance, the more similar are the vectors and the datapoints.

### Pearson Correlation
The Pearson Correlation emphasis the similarity and correlation between to vectors or datapoints. It results in a score between -1 (total correlation) and 1 (total positiv correlation). To calculate the Pearson Correlation, we make use of following formula:

![Formula of Pearson Correlation Calculation](images/pearson_formula.JPG "Formula of Pearson Correlation Calculation")

```python
from scipy.stats import pearsonr
pearsonr(v1, v2)
```

### Cosine Similarity
The Cosine Simularity is a common used measurement. It calculates the cosine value of the angle between two vectors or datapoints. A consine value of 1 indicates total similarity and a cosine value of -1 indicates total dissimilarity.

![Formula of Cosine Similarity Calculation](images/cosine_formula.JPG "Formula of Cosine Similarity Calculation")

```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(v1, v2)
```


## Recommendation Systems
Recommendation Systems or Recommendation Engines are ML-based systems to recommend a product (or datapoint) based on ratings, content, metadata or user and behavior similarities. At least, you can differ between three types of recommenders. The following examples are related to the IMDB Top 250 dataset, which you can find on [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset).

### Simple Ranking Recommender 
This type on Recommender is is very trivial. The recommendation is based on the rankings given by the users. To make the rankings comparable, you have to calculate the weighted rating for each movie in the dataset. Here we can used the following formula:

FORMEL Weighted rating with description

Based on the weigthed rankings (see score column in the example), we are now able to recommend best rated movies to the user. 

```python
import pandas as pd
data = pd.read_csv("../data/movies_metadata.csv")

# Number of votes garnered by the 80th percentile movie
m = data.vote_count.quantile(0.80)

# Only consider movies that have more than m votes
movies = data[movies.vote_count >= m]

C = data.vote_average.mean()

def weighted_rating(x, m=m, C=C):
    v = x.vote_count
    R = x.vote_average
    return (v/(v+m) * R) + (m/(m+v) * C)

movies['score'] = movies.apply(weighted_rating, axis=1)

# Sort data by descending score
movies = movies.sort_values('score', ascending=False)
```

### Knowledge Filtering
This type of Recommender uses explicit information given by the user on his favorites. Based on this input you can simply filter the dataset and recommend the movies with the best scores. 

```python
movies = gen_df.copy()
movies = movies[(movies[genre] == 1) &
               (movies.runtime >= low_time) &
               (movies.runtime <= high_time) &
               (movies.year >= low_year) &
               (movies.year <= high_year)]

# Compute the weightes score 
C = movies.vote_average.mean()
m = movies.vote_count.quantile(percentile)

q_movies = movies.copy().loc[movies.vote_count >= m]

q_movies['score'] = q_movies.apply(lambda x: (x['vote_count']/(x['vote_count'] + m) * x['vote_average']) + (m/(m + x['vote_count']) * C), axis=1) 

# Sort movies
q_movies = q_movies.sort_values('score', ascending=False)
```

### Data-based Recommender
This type of Recommender uses descriptions, contents or any other type of meta-information and calculates the similarity between the movies. Based on this similarity scores, the system recommends similar movies to the user.

In the following example we use a set of metadata to create a soup of information that can be compared.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a metadata soup containing all information we want to take into account
def create_soup(x):
    return ' '.join(x.keywords) + ' ' + ' '.join(x.cast) + ' ' + x.director + ' ' + ' '.join(x.genres)

data['soup'] = data.apply(create_soup, axis=1)
data.soup.head(3)

# Create the recommender with CountVectorizer and cosine_similarity
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data.soup)

# Compute cosine similarity scores
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Reset index and construct reverse mapping
data = data.reset_index()
indices = pd.Series(data.index, index=data.title)

# Function that takes the title of a movie as an input for recommendation
def content_recommender(title, cosine_sim=cosine_sim, data=data, indices=indices):
  idx = indices[title]

  # Get the pairwise similarity score of all movies withe that movie
  # Convert it into a list of tuples (position, similarity score)
  sim_scores = list(enumerate(cosine_sim[idx]))

  # Sort the list
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  # Reduce it to the first 10 movies; Ignore the first (itself)
  sim_scores = sim_scores[1:11]

  # Get the indices
  movie_indices = [i[0] for i in sim_scores]

  # Return the top 10
  return data.title.iloc[movie_indices]

content_recommender('The Lion King')
```

### Collaborativ Filtering


## Dimensionality Reduction
A lot of data and different features have a bad impact on the performance of the training and the model. This is also known as the curse of dimensionality and a big problem in Machine Learning. There are some techniques to reduce the number of dimensions in the data space. This might increase the overall performance of model.

### Principal Component Analysis (PCA)
PCA is an unsupervised feature extraction algorithm and describes the dataset in a limited number of Principle Components. The following description is taken from Wikipedia:

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set.

To sum it up, it combines our input variables in a specific way, then we can drop the “least important” variables while still retaining the most valuable parts of all of the variables. The first PC includes the most variance of the data.

To clarify the process of PCA, please have a look to this [StatQuest Video on PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ).

```python
from sklearn.decomposition import PCA
pca = PCA(n_components)
pca_data = pca.fit_transform(normalized_X)
pca_data = pandas.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

# Get the ratio of variance for each PC
ratios = pca.explained_variance_ratio_
```




