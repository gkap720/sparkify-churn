# Sparkify Churn Analysis
In this project, I analyzed how likely users were to leave the platform "Sparkify", an imaginary music streaming platform. The data is a collection of user events performed on the platform (navigating to certain pages, playing songs and so on). The other challenge is that the dataset is quite large, necessitating the use of a distributed training platform. In this case, I chose Apache Spark.
## Project Structure
The project is organized into three Jupyter notebooks:
1. A preliminary data analysis using a small subset of the data only using pandas - [EDA.ipynb](EDA.ipynb)
2. A conversion of the code from the first notebook into PySpark code, but still running locally - (Spark_local.ipynb)[Spark_local.ipynb]
3. The final notebook used to perform the distributed training on an EMR cluster on AWS - (Spark_final.ipynb)[Spark_final.ipynb]

Follow through each notebook to get an idea of the development process or just skip to the third notebook to see the final implementation.

## Model Choice
For this project, I chose to train a Gradient Boosted Trees model. GBT models are well-suited for sparse data with a relatively large number of features. In this dataset, some users only visited certain pages once so often the columns are only 1's and 0's. After testing this approach on a small subset of the data and seeing the increase in performance over a vanilla LogisticRegression, I was pretty confident that it would perform well. For this task, I would also consider SVM models and K neighbors as things to try in the future, but the overhead and the training time on such a large dataset (as well as the cost) limited my ability to explore other options.
## Results
After training a Gradient Boosted Trees model on the full dataset, I ended up with a model with a precision score of 0.87 and recall of 0.61. Because this is a highly imbalanced dataset and I'm more interested in predicting the target class as much as possible, I then try to optimize for recall. After tweaking the threshold, I ended up with a recall score of 0.8 while sacrificing some precision. This allows me to increase the number of true positives while also inreasing the number of false positives. In this case, the trade-off is worth it to try to retain as many customers as possible, but the strategy for retaining customers must be carefully thought out in order to ensure that we aren't losing money on offers or spamming our users with emails.
