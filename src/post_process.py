from collections import Counter
import sys

def _getTopics(num_topics, partitioned_docs):
	"""
	Returns the set of phrases modelling each document.
	"""
	topics = []
	for i in range(num_topics):
		topics.append(Counter())
	f = open('intermediate_output/phrase_topics.txt', 'r')
	doc_count = 0
	for line in f:
		phrase_count = 0
		for phrase in line.split(","): 
			while len(partitioned_docs[doc_count]) == 0:
				doc_count += 1
			topics[int(phrase)][partitioned_docs[doc_count][phrase_count]] += 1
			phrase_count += 1
		doc_count += 1
	return topics

def _restore_vocab():
	"""
	Reads and stores the vocabulary file.
	"""
	vocab = {}
	index_vocab = []
	f = open('intermediate_output/vocab.txt', 'r')
	idx = 0
	for word in f:
		word = word.replace("\n", "")
		vocab[word] = idx
		index_vocab.append(word)
		idx += 1
	return vocab, index_vocab

def _restore_freq_phrases():
	"""
	Reads and stores the frequent phrases.
	"""
	freq_phrase = Counter()
	f = open('intermediate_output/phrases.txt', 'r')
	for phrase in f:
		key,value = phrase.split(", ")
		freq_phrase[key] = int(value)
	return freq_phrase

def _restore_partitioned_docs():
	"""
	Reads and stores the partitioned documents.
	"""

	partitioned_docs = []
	is_first = True
	f = open('intermediate_output/partitioneddocs.txt', 'r')
	for doc in f:
		phrases = doc.split(" ,")
		new_doc = []

		if len(phrases) == 1 and phrases[0]=="\n":
			partitioned_docs.append(new_doc)
			continue

		for phrase in phrases:
			new_doc.append(phrase.replace("\n", ""))
		partitioned_docs.append(new_doc)
	return partitioned_docs

def _get_string_phrase(phrase, index_vocab):
	"""
	Returns the string representation of the phrase.
	"""
	res = ""
	for vocab_id in phrase.split():
		if res == "":
			res += index_vocab[int(vocab_id)]
		else:
			res += " " + index_vocab[int(vocab_id)]
	return res

def _store_top_phrases(freq_phrases, index_vocab):
	"""
	Stores the top phrases.
	"""
	f = open('output/top_phrases.txt', 'w')
	for tup in freq_phrases.most_common():
		f.write(_get_string_phrase(tup[0], index_vocab)+" "+str(tup[1])+"\n")
	f.close()

def _store_topics(partitioned_docs, index_vocab):
	"""
	Stores the phrases modelling each topic in a different file.
	"""
	topics = _getTopics(4, partitioned_docs)
	idx = 1
	for topic in topics:
		file_name = 'output/topic' + str(idx)+'.txt'
		f = open(file_name, 'w')
		for phrase in topic:
			if len(phrase.split(" ")) > 1:
				f.write(_get_string_phrase(phrase, index_vocab)+" "+ str(freq_phrases[phrase])+"\n")
   		f.close()
   		idx += 1

if __name__ == "__main__":
	arguments = sys.argv
	num_topics = int(arguments[1])

	#restore the intermediate files
	vocab, index_vocab = _restore_vocab()
	freq_phrases = _restore_freq_phrases()
	partitioned_docs = _restore_partitioned_docs()

	#get all phrases modelling each topic
	topics = _getTopics(num_topics, partitioned_docs)

	#store the final result
	_store_top_phrases(freq_phrases, index_vocab)
	_store_topics(partitioned_docs, index_vocab)
	for topic in topics:
	    print "------NEW---------"
	    for phrase, count in topic.most_common():
	        if len(phrase.split(" ")) > 1:
	        	print _get_string_phrase(phrase, index_vocab), count