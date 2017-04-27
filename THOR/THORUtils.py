""" Utility functions
"""
import cPickle as pickle

# fast serialization with marshal

def dump(obj, fd):
	pickle.dump(obj, fd, protocal = pickle.HIGHEST_PROTOCOL)

def load(fd):
	return pickle.load(fd)

