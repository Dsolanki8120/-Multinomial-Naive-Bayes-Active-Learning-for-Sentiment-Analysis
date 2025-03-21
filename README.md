This project implementing a Multinomial Naive Bayes classifier for sentiment classification on tweets and incorporating an Active Learning Strategy (ALS) to improve annotation efficiency. The dataset used is the Sentiment140 dataset, where tweets are labeled as positive (1) or negative (0).
The primary tasks include:
Implementing a Count Vectorizer from scratch.
Implementing Multinomial Naive Bayes without using external ML libraries.
Implementing Active Learning, selecting the most informative samples for labeling.
Optimizing the training process for efficiency.
Visualizing results and performance comparisons.

Setup & Installation
Step 1: Clone the Repository
git clone https://github.com/Dsolanki8120/-Multinomial-Naive-Bayes-Active-Learning-for-Sentiment-Analysis.git

step2 Download and prepare data:
Download the dataset from: http://10.192.30.174:8000/download ( SR number=24796).

step 3:
Run the Multinomial Naive Bayes Model:
python src/prob1.py --srno 24796
Expected Accuracy: >75% on validation and test set.

step 4:
Run Active Learning Strategy (ALS):
without use of incrmental update of data: python src/prob2.py --srno 24796 --run_id 1 
with incremental update of data : python src/prob2.py --srno 24796 --run_id 1 --is_active
both run it 5 times

step5:

 Generate Accuracy & Training Time Plots:
 python src/plot.py --srno 24796 --supervised_accuracy 0.75

 

Count Vectorizer (utils.py):
Converts sentences into sparse numerical vectors.
Efficient storage using scipy.sparse.
Top 10,000 words are used to build the vocabulary.


Multinomial Naive Bayes (model.py & prob1.py):
Computes class conditional probabilities.
Uses a Bernoulli distribution for inference.
Achieves >75% accuracy on validation & test set.


 Active Learning Strategy (prob2.py):

Selects data points with highest uncertainty (entropy-based or Gini impurity).
Updates Naive Bayes model without retraining from scratch.
Reduces labeled dataset while maintaining accuracy.

Performance Comparison (plot.py):
Plots comparison between random selection vs ALS.
Measures training time efficiency between retraining & model updating.





















