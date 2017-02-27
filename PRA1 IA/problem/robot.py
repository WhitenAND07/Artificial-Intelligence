# -*- coding: utf-8 -*-

from edulog.search.containers import HashableList
from edulog.search.problem import DRange, Problem, action

class BlockWorldProblem():

	def __init__(self,**Kwargs):
		self.numBlocks = k
		self.maxPlaces = m
		self.maxHeight = c
		self.startState = startState
		self.goalState = goalState


	def get_start_states(self):
		return self.startState
		

	def is_goal_state(self. state):
		return self.goalState == state


	def is_valid_state(self, state):
		return True

	@action(1, DRange(3), DRange(3))
	def move(self, origen, destino, state):
		
		if len(state[origen]) == 0 or len(state[destino] >= self.maxHeight):
			return None

		new_state = copy.deepcopy(state)
		new_state[destino].append(new_state[origen].pop())

		return new_state


if __name__ == "__main__":

