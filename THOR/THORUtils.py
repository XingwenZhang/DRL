""" Utility functions
"""
import cPickle as pickle

# fast serialization with marshal

def dump(obj, fd):
	pickle.dump(obj, fd)

def load(fd):
	return pickle.load(fd)

