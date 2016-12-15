from __future__ import division
import dirichlet
import random
import utils
import math
from collections import Counter


class PhraseLDA(object):

    """ 
    Topic Modeling using lda with collapsed gibbs sampling on "bag-of-phrases".
    :param partitioned_docs:
        List of documents, where each document is partitioned into a list of 
        phrases. 
    :param index_vocab:
        Mapping of integer index to string word.
    :param num_topics:
        Number of requested topics that need to be extracted from the inputted
        documents.
    :param alpha:
        Scalar hyperparameter representing the symmetric Dirichlet prior over 
        document-topic (lambda) distributions. Initially, each topic will have 
        the same prior weight for each document.
    :param beta:
        Scalar hyperparameter representing the symmetric Dirichlet prior over 
        topic-word (phi) distributions. Initially, each word will have the same 
        prior weight for each topic.
    :param iterations:
        The total number of Gibbs sampling iterations over the entire corpus.
    :param optimization_iterations:
        Sets hyperparameter optimization after each ``optimization_iterations``
        of the Gibbs sampling. Set to ``None```` for no optimization.
    :param optimization_burnin:
        Number of gibbs sampling iterations before hyperparameter optimization
        starts.
    """

    def __init__(self, partitioned_docs, index_vocab, 
                 num_topics=5, alpha=4, beta=0.01, iterations=1000,
                 optimization_iterations=100, optimization_burnin=50):
        # initialize corpus
        self.documents = partitioned_docs
        self.num_documents = len(partitioned_docs)
        self.index_vocab = index_vocab
        self.num_words = len(index_vocab)
        self.num_topics = num_topics

        # initialize hyperparameters
        self.alpha = [alpha] * self.num_topics
        self.alpha_sum = alpha * num_topics
        self.beta = beta
        self.beta_sum = self.beta * self.num_words
        
        # gibbs sampling parameters
        self.iterations = iterations
        self.optimization_iterations = optimization_iterations
        self.optimization_burnin = optimization_burnin

    def _initialize(self):
        self._init_documents()

        # Array stores per topic counts
        self.n_t = [0] * self.num_topics

        # 2d array that stores document/topic counts by phrase, and word respectively
        self.n_d_t_phrases = [[0] * self.num_topics for __ in range(self.num_documents)]
        self.n_d_t_words = [[0] * self.num_topics for __ in range(self.num_documents)]

        # 2d array that stores topic/word counts
        self.n_t_w = [[0] * self.num_words for __ in range(self.num_topics)]

        self._init_documents_topics()

        self._init_histogram()

    def _init_documents(self):
        self.documents_words = []

        self.max_documents_phrases_count = 0
        self.max_documents_words_count = 0

        for document in self.documents:
            document_words = []

            document_words_count = 0

            for phrase in document:
                for word in phrase:
                    document_words.append(word)

                    document_words_count += 1

            self.documents_words.append(document_words)

            self.max_documents_phrases_count = max(self.max_documents_phrases_count, len(document))
            self.max_documents_words_count = max(self.max_documents_words_count, document_words_count)

    def _init_documents_topics(self):
        # we assign a random topic to each phrase in the document
        self.documents_phrases_topic = []

        for document_index, document in enumerate(self.documents):
            document_phrases_topic = []
            for phrase_index, phrase in enumerate(document):
                document_phrase_topic = random.randint(0,self.num_topics-1)
                document_phrases_topic.append(document_phrase_topic)
                
                # Increase counts
                self.n_t[document_phrase_topic] += len(phrase)
                self.n_d_t_phrases[document_index][document_phrase_topic] += 1
                self.n_d_t_words[document_index][document_phrase_topic] += len(phrase)
                for word_index in phrase:
                    self.n_t_w[document_phrase_topic][word_index] += 1
            self.documents_phrases_topic.append(document_phrases_topic)


    def _init_histogram(self):
        self.document_length_histogram = [0] * (self.max_documents_words_count + 1)
        for document in self.documents_words:
            self.document_length_histogram[len(document)] += 1
        self._init_topic_document_histogram()

    def _init_topic_document_histogram(self):
        self.topic_document_histogram = [[int()] * (self.max_documents_words_count + 1)
                                         for __ in range(self.num_topics)]

    def _sample_topic(self, sampling_probabilities):
        threshold = random.uniform(0.0,1.0) * sum(sampling_probabilities)
        cumulative_sum = 0
        for topic in range(self.num_topics):
            cumulative_sum += sampling_probabilities[topic]
            if cumulative_sum > threshold:
                break
        return topic

    def _calculate_topic_probabilities(self, document_index, phrase_index):
        topic_probabilities = []
        for topic_index in range(self.num_topics):
            left = self.alpha[topic_index] + self.n_d_t_phrases[document_index][topic_index]
            right = 1.0
            for word_index in self.documents[document_index][phrase_index]:
                right *= (self.beta + self.n_t_w[topic_index][word_index]) / (self.beta_sum + (self.n_t[topic_index]))
            topic_probability = left * right
            topic_probabilities.append(topic_probability)
        return topic_probabilities

    def _should_optimize(self, iterations):
        if self.optimization_iterations is None:
            return false
        iterations_condition = ((iterations+1) % self.optimization_iterations) == 0
        burnin_condition = ((iterations+1) > self.optimization_burnin)
        return iterations_condition and burnin_condition


    def run(self):
        self._initialize()        
        for iteration in range(self.iterations):
            if iteration % 100 == 0:
                print "iteration", iteration

            for document_index, document in enumerate(self.documents):
                for phrase_index, phrase in enumerate(document):
                    document_phrase_topic = self.documents_phrases_topic[document_index][phrase_index]

                    # reduce counts for sampling
                    self.n_t[document_phrase_topic] -= len(phrase)
                    self.n_d_t_phrases[document_index][document_phrase_topic] -= 1
                    self.n_d_t_words[document_index][document_phrase_topic] -= len(phrase)
                    for word_index in phrase:
                        self.n_t_w[document_phrase_topic][word_index] -= 1

                    sampling_probabilities = self._calculate_topic_probabilities(document_index, phrase_index)
                    document_phrase_topic = self._sample_topic(sampling_probabilities)

                    self.documents_phrases_topic[document_index][phrase_index] = document_phrase_topic
                    
                    self.n_t[document_phrase_topic] += len(phrase)
                    self.n_d_t_phrases[document_index][document_phrase_topic] += 1
                    self.n_d_t_words[document_index][document_phrase_topic] += len(phrase)
                    for word_index in phrase:
                        self.n_t_w[document_phrase_topic][word_index] += 1

            if self._should_optimize(iteration):
                self._optimize_hyperparameters()
        
        topics = self._getTopics()
        return self.documents_phrases_topic, self._getMostFrequentPhrasalTopics(topics)

    def _optimize_hyperparameters(self):
        self._init_topic_document_histogram()
        for topic_index in range(self.num_topics):
            for document_index in range(len(self.documents)):
                self.topic_document_histogram[topic_index][self.n_d_t_words[document_index][topic_index]] += 1

        self.alpha_sum = dirichlet.learn_parameters(
            self.alpha, self.topic_document_histogram, self.document_length_histogram)
        max_topic_size = 0

        for topic_index in range (self.num_topics):
            if self.n_t[topic_index] > max_topic_size:
                max_topic_size = self.n_t[topic_index]

        topic_size_histogram = [0] * (max_topic_size + 1)
        count_histogram = [0] * (max_topic_size + 1)

        topic_index = 0
        for topic_index in range(self.num_topics):
            topic_size_histogram[self.n_t[topic_index]] += 1
            for word_index in range(self.num_words):
                count_histogram[
                    self.n_t_w[topic_index][word_index]] += 1

        self.beta_sum = dirichlet.learn_symmetric_concentration(
            count_histogram, topic_size_histogram, self.num_words, self.beta_sum)
        self.beta = self.beta_sum / self.num_words

    def store_phrase_topics(self, path):
        f = open(path, 'w')
        for document in self.documents_phrases_topic:
            f.write(",".join(str(phrase) for phrase in document))
            f.write("\n")

    def _getTopics(self):
        """
        Returns the set of phrases modelling each document.
        """
        topics = []
        for i in range(self.num_topics):
            topics.append(Counter())
        for document_index, document in enumerate(self.documents_phrases_topic):
            for phrase_index, phrase_topic in enumerate(document):
                phrase = " ".join(str(word) for word in self.documents[document_index][phrase_index])
                topics[phrase_topic][phrase] += 1
        return topics

    def _getMostFrequentPhrasalTopics(self, topics):
        output = []
        topic_index = 0
        for topic in topics:
            output_for_topic = []
            print "topic", topic_index
            for phrase, count in topic.most_common():
                if len(phrase.split(" ")) > 1:
                    val = utils._get_string_phrase(phrase, self.index_vocab), count
                    output_for_topic.append(val)
                    print val
            output.append(output_for_topic)
            topic_index += 1
        return output
