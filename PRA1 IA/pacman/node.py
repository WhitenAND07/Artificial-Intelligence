#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Node:

	def __init__(self, state, parent=None, action=None, cost=0):
		self.state = state
		self.parent = parent
		self.action = action
		self.cost = cost

	def __str__(self):
		return "-- Node {0} --\n  Parent: {1}\n  Action: {2}\n  Cost: {3}" \
		.format(self.state, self.parent, self.action, self.cost)


	def __eq__(self, other):
		return self.state == other.state 


	def path(self):
		sol = []
		while (self.action != None):
			sol.append(self.action)
			self = self.parent

		sol.reverse()
		return sol

if __name__ == '__main__':
	n = Node('example state')
	n1 = Node((0,0))
	n2 = Node((5,0), n1, "emptyA", 0)
	n3 = Node((5,0), n2, "emptyB", 0)
	print n3.path()
	print n

