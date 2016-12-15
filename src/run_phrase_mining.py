import phrase_mining
import sys

arguments = sys.argv
print 'Running Phrase Mining...'

file_name = arguments[1]

# represents the minimum number of occurences you want each phrase to have.
min_support=10 

# represents the threshold for merging two words into a phrase. A lower value
# alpha leads to higher recall and lower precision,
alpha=4

# length of the maximum phrase size
max_phrase_size=10

phrase_miner = phrase_mining.PhraseMining(file_name, min_support, max_phrase_size, alpha);
phrase_miner.mine()