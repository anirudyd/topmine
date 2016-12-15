import phrase_lda
import sys

arguments = sys.argv

num_topics = int(arguments[1])
iteration = 1100
optimization_burnin = 100
alpha = 4
optimization_iterations = 50
beta = 0.01

def importDocuments(path):
    f = open(path, 'r')
    documents = []
    document_index = 0
    for line in f:
        line = line.strip()
        if len(line) < 1:
            continue
        groups = line.split(" ,")
        document = []
        for group in groups:
            group_of_words = map(int,group.split(" "))
            document.append(group_of_words)
        documents.append(document)
    return documents

def importVocab(path):
    f = open(path, 'r')
    index_vocab = []
    index = 0
    for line in f:
        index_vocab.append(line.replace("\n", ""))
    return index_vocab

print 'Running PhraseLDA...'

# input
partitioneddocs_file = "intermediate_output/partitioneddocs.txt"

phrase_topics_file = "intermediate_output/phrase_topics.txt"

vocab_file = "intermediate_output/vocab.txt"


partitioned_docs = importDocuments(partitioneddocs_file)
vocab_file = importVocab(vocab_file)

plda = phrase_lda.PhraseLDA( partitioned_docs, vocab_file, num_topics , alpha, beta, iteration, optimization_iterations, optimization_burnin);

document_phrase_topics = plda.run()
plda.store_phrase_topics(phrase_topics_file)