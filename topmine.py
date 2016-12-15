""" Implements the algorithm provided in the following research paper:

El-Kishky, Ahmed, et al. "Scalable topical phrase mining from text corpora." Proceedings of the VLDB Endowment 8.3 (2014): 305-316.
"""
import subprocess, shlex
def get_output_of(command):
	args = shlex.split(command)
	return subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]

file_name="input/dblp_5k.txt"
num_topics=4

phrase_mining_cmd = "pypy src/run_phrase_mining.py {0}".format(file_name)
print get_output_of(phrase_mining_cmd)

phrase_lda_cmd = "pypy src/run_phrase_lda.py {0}".format(num_topics)
print get_output_of(phrase_lda_cmd)