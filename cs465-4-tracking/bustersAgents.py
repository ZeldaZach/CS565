# bustersAgents.py
# ----------------
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


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters


class NullGraphics:
    """Placeholder for graphics"""

    def __init__(self):
        pass

    def initialize(self, state, is_blue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def __init__(self):
        self.beliefs = None
        self.legal_positions = None

    def initializeUniformly(self, game_state):
        """Begin with a uniform distribution over ghost positions."""
        self.beliefs = util.Counter()
        for p in self.legal_positions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, game_state):
        noisy_distance = observation
        emission_model = busters.getObservationDistribution(noisy_distance)
        pacman_position = game_state.getPacmanPosition()
        all_possible = util.Counter()

        for p in self.legal_positions:
            true_distance = util.manhattanDistance(p, pacman_position)
            if emission_model[true_distance] > 0:
                all_possible[p] = 1.0
        all_possible.normalize()
        self.beliefs = all_possible

    def elapseTime(self, game_state):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    """An agent that tracks and displays its beliefs about ghost positions."""

    def __init__(
        self,
        index=0,
        inference="ExactInference",
        ghost_agents=None,
        observe_enable=True,
        elapse_time_enable=True,
    ):
        inference_type = util.lookup(inference, globals())
        self.inferenceModules = [inference_type(a) for a in ghost_agents]
        self.observeEnable = observe_enable
        self.elapseTimeEnable = elapse_time_enable
        self.display = None
        self.firstMove = None

    def registerInitialState(self, game_state):
        """Initializes beliefs and inference modules"""
        import __main__

        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(game_state)
        self.ghostBeliefs = [
            inf.getBeliefDistribution() for inf in self.inferenceModules
        ]
        self.firstMove = True

    def observationFunction(self, game_state):
        """Removes the ghost states from the game_state"""
        agents = game_state.data.agentStates
        game_state.data.agentStates = [agents[0]] + [
            None for i in range(1, len(agents))
        ]
        return game_state

    def getAction(self, game_state):
        """Updates beliefs, then chooses an action based on updated beliefs."""
        for index, inf in enumerate(self.inferenceModules):
            if not self.firstMove and self.elapseTimeEnable:
                inf.elapseTime(game_state)
            self.firstMove = False
            if self.observeEnable:
                inf.observeState(game_state)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
        self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(game_state)

    def chooseAction(self, game_state):
        """By default, a BustersAgent just stops.  This should be overridden."""
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    """An agent controlled by the keyboard that displays beliefs about ghost positions."""

    def __init__(self, index=0, inference="KeyboardInference", ghost_agents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghost_agents)

    def getAction(self, game_state):
        return BustersAgent.getAction(self, game_state)

    def chooseAction(self, game_state):
        return KeyboardAgent.getAction(self, game_state)


from distanceCalculator import Distancer
from game import Actions
from game import Directions


class GreedyBustersAgent(BustersAgent):
    """An agent that charges the closest ghost."""

    def __init__(self):
        self.distancer = None

    def registerInitialState(self, game_state):
        """Pre-computes the distance between every two points."""
        BustersAgent.registerInitialState(self, game_state)
        self.distancer = Distancer(game_state.data.layout, False)

    def chooseAction(self, game_state):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) game_state.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             game_state.getLivingGhosts() list.
        """
        pacman_position = game_state.getPacmanPosition()
        legal = [a for a in game_state.getLegalPacmanActions()]
        living_ghosts = game_state.getLivingGhosts()
        living_ghost_position_distributions = [
            beliefs
            for i, beliefs in enumerate(self.ghostBeliefs)
            if living_ghosts[i + 1]
        ]
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
