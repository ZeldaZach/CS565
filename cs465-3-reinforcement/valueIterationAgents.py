# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import copy
import sys
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        Some useful mdp methods you will use:
        mdp.getStates()
        mdp.getPossibleActions(state)
        mdp.getTransitionStatesAndProbs(state, action)
        mdp.getReward(state, action, nextState)
        mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        # Create a save state of our mapping
        self.values = util.Counter()

        # Iterate the same function iter times
        for _ in range(0, self.iterations):
            # Each iteration will create its own value map
            # and we will normalize over the iterations
            temp_value_map = util.Counter()

            # Calculate services for each state in the markov process
            for state in self.mdp.getStates():
                # If we're at a terminal state, there are no more states
                if self.mdp.isTerminal(state):
                    continue

                # Calculate q-values for each possible state, saving the best choice
                best_value = -1 * sys.maxsize
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > best_value:
                        best_value = q_value
                    temp_value_map[state] = best_value
            # After iterating all states, copy back our values
            self.values = copy.copy(temp_value_map)

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        new_q_value = 0
        # Q(state, action) = SUM(new_states) (probability(state, new_state) * (Reward + discount * V(new_state))
        for next_state, probability in self.mdp.getTransitionStatesAndProbs(
            state, action
        ):
            new_q_value += probability * (
                self.mdp.getReward(state, action, next_state)
                + self.discount * self.values[next_state]
            )

        return new_q_value

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        best_option = -1 * sys.maxsize
        decision = None

        # See which q-value is our best option, and take it
        for action in self.mdp.getPossibleActions(state):
            best_q_value = self.computeQValueFromValues(state, action)
            if best_option < best_q_value:
                best_option = best_q_value
                decision = action

        return decision if decision else None

    def getPolicy(self, state):
        """
        Get the best action to take from a state
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        """
        Get the Q Value of a state
        """
        return self.computeQValueFromValues(state, action)
