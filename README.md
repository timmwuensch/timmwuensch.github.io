This is my knowledge glossar about topics on Machine Learning and Data Science. I use it for myself to recapture different topics, dicscriptions, solutions and technologies. 

#### Table of contents
1. [Machine Learning](#machine-learning)
2. [Data Science](#data-science)

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

![Decision Tree Impurity Calculation](https://github.com/timmwuensch/timmwuensch.github.io/blob/master/images/decision_tree_impurity.JPG "Decision Tree Impurity Calculation")

The dataset has three feature and one label column. To build a Decision Tree, we have to find a feature with the lowest Gini Impurity for our data record. The first feature (*male*) is of Boolean Type and easy to handle. You just have to answer the question: How much is obesity dependent from gender? In the figure above, you can see a simple Gini Impurity calculation.

![Impurity Calculation for numeric values](https://github.com/timmwuensch/timmwuensch.github.io/blob/master/images/decision_tree_impurity_weight.JPG "Decision Tree Impurity Calculation for numeric values")

To calculate the Gini Impurity of a non boolean feature like *weight* we have to do a couple of more steps. Firstly, sort the patients by ascending weight. Secondly, calculate the average weight for every adjacent patients. Finally, calculate the Gini Impurity for every average weight and take the lowest to define the Gini Impurity of the feature.

When it comes to ranked or classified data, you have to calculate the Gini Impurity for each combination of classes or rank intervals. 

## Random Forest
## Support Vector Machines
## k-Nearest Neighbor
## k-Means Clustering
## Deep Learning
## Convolutional Neural Networks
## Reinforcement Learning
## Natural Language Processing



# Data Science
The challenge behind Data Science is to turn data into information. These information could be used to support business processes with advanced analytics, predictions and decisions. The tasks of a Data Scientist pyrtly overlaps with Machine Learning, but there are still differences between both field of research. 

The work of a Data Scientist could be divided into three segments. 

IMAGE

The first segment refers to the variety of **Data Sources and Technologies**. Here you have to identify valid data sources (databases or datastreams) and convert data into a usable structure for further analytics (Data Ware House). In this context Data Scientists make use of ETL-Pipelines (Extract-Transform-Load) to transform and convert data.

The second segment refers to so-called **Data Frameworks** (e.g. Pandas or Apache Spark). These frameworks or libraries provide data structures and methods to handle the data. The selection of an adequate Data Framework is often based on the amount, size and type of data. 

The third segment refers to variety of **analytical, statistical and mathemetical methods** which are used by Data Scientists. These methods can range from statistical basics, over regression models to Machine Learning Models. 






