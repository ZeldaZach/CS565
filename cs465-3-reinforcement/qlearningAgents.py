# qlearningAgents.py
# ------------------
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
import random
import sys
from learningAgents import ReinforcementAgent
from featureExtractors import *


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
    - computeValueFromQValues
    - computeActionFromQValues
    - getQValue
    - getAction
    - update

    Instance variables you have access to
    - self.epsilon (exploration prob)
    - self.alpha (learning rate)
    - self.discount (discount rate)

    Functions you should use
    - self.getLegalActions(state)
      which returns legal actions for a state
    """

    def __init__(self, **args):
        """
        You can initialize Q-values here...
        """
        ReinforcementAgent.__init__(self, **args)

        # Dict of tuples w/ state, action and the value provided
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        legal_actions = self.getLegalActions(state)

        if not legal_actions:
            return 0.0

        best_value = -1 * sys.maxsize
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            if q_value >= best_value:
                best_value = q_value

        return best_value

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        best_value = -1 * sys.maxsize
        best_actions = []

        for action in self.getLegalActions(state):
            q_value = self.getQValue(state, action)
            if q_value > best_value:
                # We have a new best value, clear lesser options
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                # This is just as good, so add to pickable list
                best_actions.append(action)

        if best_actions:
            return random.choice(best_actions)
        return None

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)

        if not legal_actions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legal_actions)

        return self.getPolicy(state)

    def update(self, state, action, next_state, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        # (1-a) * max(Q) + a * (reward + dis * V'(s'))
        new_value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
            reward + self.discount * self.getValue(next_state)
        )

        self.q_values[(state, action)] = new_value

    def getPolicy(self, state):
        """
        Get the policy from Q values
        """
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        """
        Get the value from Q values
        """
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as QLearningAgent, but with different default parameters
    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)

        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        """
        Get weights of agent
        """
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        return self.getWeights() * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, next_state, reward):
        """
        Should update your weights based on transition
        """
        feature_vector = self.featExtractor.getFeatures(state, action)
        max_q_value_next_state = self.computeValueFromQValues(next_state)
        best_q_value = self.getQValue(state, action)

        # diff = (rew + dis * max(Q(s', a')) - Q(s,a)
        difference = reward + self.discount * max_q_value_next_state - best_q_value

        # wi = wi + alpha * diff * fi(s,a)
        for feature in feature_vector:
            self.weights[feature] += self.alpha * difference * feature_vector[feature]

    def final(self, state):
        """
        Called at the end of each game.
        """
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            pass
