# Sentiment-Analysis-using-various-classifiers
Experiment sentiment analysis using regular classifiers and deep neural networks then compare performance




We just saw how the task of sentiment analysis can be solved via a traditional machine learning approach: BoW + a nonlinear classifier. We now switch gears and use Recurrent Neural Networks, and in particular LSTMs, to perform sentiment analysis in Keras. 


The weakness of BOW is that fact that it ignores the order in which the words appear. So for instance, if you have a positive word and a negation of that positive word, the BOW might not discover that it is in fact a negative sentiment. Here is an example:
- This is a great movie, it is not bad at all
- This is a bad movie, it is not great at all

The first one is a positive sentiment, the second is negative. But notice that both sentences include the same words exactly but in different order. In a one-hot encoding, or in a bag of words model, their resulting feature vectors accross the vocabulary will be identical. So one would imagine that the output is very similar too.

One way to fix this is to consider the order of the words. This is where RNNs and LSTMs come to the rescue. With an RNN (LSTM) architecture, we'd be feeding the one-hot-encoded word vectors one by one. At each point, the model takes as input, the previous output, joined with the new word, in order to produce an output. The final output is an encoding of the sentence.

Once we have the encoding of this sentence, we run that through one or more dense layers, which will then get trained to predict the sentiment of the review.
