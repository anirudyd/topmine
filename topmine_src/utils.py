def store_partitioned_docs(partitioned_docs, path="intermediate_output/partitioneddocs.txt"):
    f = open(path, 'w')
    for document in partitioned_docs:
        f.write(", ".join(" ".join(str(word) for word in phrase) for phrase in document))
        f.write("\n")

def load_partitioned_docs(path="intermediate_output/partitioneddocs.txt"):
    f = open(path, 'r')
    partitioned_docs = []
    document_index = 0
    for line in f:
        line = line.strip()
        if len(line) < 1:
            continue
        phrases = line.split(", ")
        partitioned_doc = []
        for phrase in phrases:
            phrase_of_words = map(int,phrase.split(" "))
            partitioned_doc.append(phrase_of_words)
        partitioned_docs.append(partitioned_doc)
    return partitioned_docs

def store_vocab(index_vocab, path="intermediate_output/vocab.txt"):
    """
    Stores vocabulary into a file.
    """
    f = open('intermediate_output/vocab.txt', 'w')
    for word in index_vocab:
        f.write(word+"\n")
    f.close()

def load_vocab(path="intermediate_output/vocab.txt"):
    """
    Loads vocabulary from a file.
    """
    f = open(path, 'r')
    index_vocab = []
    index = 0
    for line in f:
        index_vocab.append(line.replace("\n", ""))
    return index_vocab

def store_frequent_phrases(frequent_phrases, path='output/frequent_phrases.txt'):
    f = open(path, 'w')
    for phrase, val in enumerate(frequent_phrases):
        f.write(str.format("{0} {1}\n",phrase, val))
    f.close()

def store_phrase_topics(document_phrase_topics, path="intermediate_output/phrase_topics.txt"):
    """
    Stores topic for each phrase in the document.
    """
    f = open(path, 'w')
    for document in document_phrase_topics:
        f.write(",".join(str(phrase) for phrase in document))
        f.write("\n")

def store_most_frequent_topics(most_frequent_topics, prefix_path="output/topic"):
    for topic_index, topic in enumerate(most_frequent_topics):
        file_name = str.format("{0}{1}.txt", prefix_path, topic_index)
        f = open(file_name, 'w')
        for phrase, val in topic:
            f.write(str.format("{0} {1}\n",phrase, val))
        f.close()

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
