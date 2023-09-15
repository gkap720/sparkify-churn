# Sparkify Churn Analysis
In this project, I analyzed how likely users were to leave the platform "Sparkify", an imaginary music streaming platform. The data is a collection of user events performed on the platform (navigating to certain pages, playing songs and so on). The other challenge is that the dataset is quite large, necessitating the use of a distributed training platform. In this case, I chose Apache Spark. 

To get the full analysis, you can read my blog post [here](https://medium.com/@gkap720/how-to-predict-churn-on-your-platform-a-case-study-c394f3cc0b58).
## Project Structure
The project is organized into three Jupyter notebooks:
1. A preliminary data analysis using a small subset of the data only using pandas - [EDA.ipynb](EDA.ipynb)
2. A conversion of the code from the first notebook into PySpark code, but still running locally - [Spark_local.ipynb](Spark_local.ipynb)
3. The final notebook used to perform the distributed training on an EMR cluster on AWS - [spark_aws_final.ipynb](spark_aws_final.ipynb)

Follow through each notebook to get an idea of the development process or just skip to the third notebook to see the final implementation.

## Model Choice
For this project, I chose to train a Gradient Boosted Trees model. GBT models are well-suited for sparse data with a relatively large number of features. In this dataset, some users only visited certain pages once so often the columns are only 1's and 0's. After testing this approach on a small subset of the data and seeing the increase in performance over a vanilla LogisticRegression, I was pretty confident that it would perform well. For this task, I also tried an SVM model, but though the results were the best when using the smaller dataset, the score went down quite a lot when using the full dataset. A KNeighborsClassifier also performed quite badly in comparison to the other model architectures I tested.
## Results
After training a Gradient Boosted Trees model on the full dataset, I ended up with a model with a precision score of 0.87 and recall of 0.67. Because this is a highly imbalanced dataset and I'm more interested in predicting the target class as much as possible, I then try to optimize for recall. After tweaking the threshold, I ended up with a recall score of 0.8 while sacrificing some precision. This allows me to increase the number of true positives while also inreasing the number of false positives. In this case, the trade-off is worth it to try to retain as many customers as possible, but the strategy for retaining customers must be carefully thought out in order to ensure that we aren't losing money on offers or spamming our users with emails.
## Libraries Used
- pyspark==3.4.1
- pandas==1.5.3
- numpy==1.22.4
- matplotlib==3.5.1
- seaborn==0.11.2
- scikit-learn==1.0.2
- jupyter

You can download all the necessary requirements with this command:

`pip install -r requirements.txt`
## Acknowledgements
Data source (S3 buckets): 
- mini => s3a://udacity-dsnd/sparkify/mini_sparkify_event_data.json
- full => s3a://udacity-dsnd/sparkify/sparkify_event_data.json

Dataset provided by [Udacity](https://udacity.com)

[PySpark Documentation](https://spark.apache.org/docs/latest/api/python/index.html)

[AWS Documentation](https://docs.aws.amazon.com/)
