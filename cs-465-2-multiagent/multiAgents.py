# multiAgents.py
# --------------
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
import util

from game import Agent, Directions


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [
            index for index in range(len(scores)) if scores[index] == best_score
        ]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legal_moves[chosen_index]

    def evaluationFunction(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        # new_ghost_states = successor_game_state.getGhostStates()
        # new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]

        # What is the closest ghost
        min_ghost_score = sys.maxint
        for ghost in successor_game_state.getGhostPositions():
            dist = util.manhattanDistance(ghost, new_pos)
            if dist < min_ghost_score:
                min_ghost_score = dist

        # TOO CLOSE FOR COMFORT -- who cares about food!
        if min_ghost_score == 1:
            return -1 * sys.maxint

        # What is the closest food
        min_food_score = sys.maxint
        for remaining_food in new_food.asList():
            dist = util.manhattanDistance(remaining_food, new_pos)
            if dist < min_food_score:
                min_food_score = float(dist)

        # Send back the old score, with a large gain
        return successor_game_state.getScore() + 1 / min_food_score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        Agent.__init__(self)
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, game_state):
        """
        Get the next best action to take
        :param game_state: Current game state
        :return: Best action to take, based on the state
        """
        # Start us with pacman in the current state
        return self.max_value(game_state, 0, 0)["direction"]

    def max_value(self, game_state, state_depth, agent_number):
        """
        With Pacman, you'll want the best value possible from the tree.
        This will calculate it and give you back the dictionary.
        :param game_state: Current game state
        :param state_depth: How far in the tree you are
        :param agent_number: Which agent is this? (Better be Pacman = 0)
        :return: Best action with cost, as dictionary
        """

        # If we're at an end point, return out
        if game_state.isWin() or game_state.isLose() or state_depth == self.depth:
            return {
                "cost": self.evaluationFunction(game_state),
                "direction": Directions.STOP,
            }

        # Holders to get the largest options while iterating
        best_cost = -1 * sys.maxint
        best_action = Directions.STOP

        legal_actions = game_state.getLegalActions(agent_number)
        for action in legal_actions:
            # Determine the worst of the ghosts and see if greater than current pacman
            next_state = game_state.generateSuccessor(agent_number, action)
            next_result = self.min_value(next_state, state_depth, agent_number + 1)

            if next_result["cost"] > best_cost:
                best_cost = next_result["cost"]
                best_action = action

        # We didn't find a better match
        if best_cost == -1 * sys.maxint:
            best_cost = self.evaluationFunction(game_state)
            best_action = Directions.STOP

        return {"cost": best_cost, "direction": best_action}

    def min_value(self, game_state, state_depth, agent_number):
        """
        With the ghosts, you'll want the worst value possible from the tree.
        This will calculate it and give back a dictionary.
        :param game_state: Current game state
        :param state_depth: How far in the tree
        :param agent_number: Which ghost (better not be 0)
        :return: Worst action with cost, as dictionary
        """
        # If we're at an end point, return out
        if game_state.isWin() or game_state.isLose() or state_depth >= self.depth:
            return {
                "cost": self.evaluationFunction(game_state),
                "direction": Directions.STOP,
            }

        # Holders to get the smallest options while iterating
        best_cost = sys.maxint
        best_action = Directions.STOP

        legal_actions = game_state.getLegalActions(agent_number)
        for action in legal_actions:
            next_state = game_state.generateSuccessor(agent_number, action)

            # Determine what our next action should be
            # Do we depth or breadth
            next_agent = (agent_number + 1) % game_state.getNumAgents()
            plus_state = 1 if not next_agent else 0
            max_pac_min_ghost = self.max_value if not next_agent else self.min_value

            next_action = max_pac_min_ghost(
                next_state, state_depth + plus_state, next_agent
            )

            if next_action["cost"] < best_cost:
                best_cost = next_action["cost"]
                best_action = action

        # No better matches found
        if best_cost == sys.maxint:
            best_cost = self.evaluationFunction(game_state)
            best_action = Directions.STOP

        return {"cost": best_cost, "direction": best_action}


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Get the next best action to take
        :param game_state: Current game state
        :return: Best action to take, based on the state
        """
        # Start us with pacman in the current state
        return self.max_value(game_state, -1 * sys.maxint, sys.maxint, 0, 0)[
            "direction"
        ]

    def max_value(self, game_state, alpha, beta, state_depth, agent_number):
        """
        With Pacman, you'll want the best value possible from the tree.
        This will calculate it and give you back the dictionary.
        :param game_state: Current game state
        :param alpha: Help break ties on the max
        :param beta: Help break ties on the min
        :param state_depth: How far in the tree you are
        :param agent_number: Which agent is this? (Better be Pacman = 0)
        :return: Best action with cost, as dictionary
        """

        # If we're at an end point, return out
        if game_state.isWin() or game_state.isLose() or state_depth == self.depth:
            return {
                "cost": self.evaluationFunction(game_state),
                "direction": Directions.STOP,
            }

        # Holders to get the largest options while iterating
        best_cost = -1 * sys.maxint
        best_action = Directions.STOP

        legal_actions = game_state.getLegalActions(agent_number)
        for action in legal_actions:
            # Determine the worst of the ghosts and see if greater than current pacman
            next_state = game_state.generateSuccessor(agent_number, action)
            next_result = self.min_value(
                next_state, alpha, beta, state_depth, agent_number + 1
            )

            if next_result["cost"] > best_cost:
                best_cost = next_result["cost"]
                best_action = action

            # If we've beat our beta, lets escape
            if best_cost > beta:
                break

            alpha = max(alpha, best_cost)

        # We didn't find a better match
        if best_cost == -1 * sys.maxint:
            best_cost = self.evaluationFunction(game_state)
            best_action = Directions.STOP

        return {"cost": best_cost, "direction": best_action}

    def min_value(self, game_state, alpha, beta, state_depth, agent_number):
        """
        With the ghosts, you'll want the worst value possible from the tree.
        This will calculate it and give back a dictionary.
        :param game_state: Current game state
        :param alpha: Help break ties on the max
        :param beta: Help break ties on the min
        :param state_depth: How far in the tree
        :param agent_number: Which ghost (better not be 0)
        :return: Worst action with cost, as dictionary
        """
        # If we're at an end point, return out
        if game_state.isWin() or game_state.isLose() or state_depth >= self.depth:
            return {
                "cost": self.evaluationFunction(game_state),
                "direction": Directions.STOP,
            }

        # Holders to get the smallest options while iterating
        best_cost = sys.maxint
        best_action = Directions.STOP

        legal_actions = game_state.getLegalActions(agent_number)
        for action in legal_actions:
            next_state = game_state.generateSuccessor(agent_number, action)

            # Determine what our next action should be
            # Do we depth or breadth
            next_agent = (agent_number + 1) % game_state.getNumAgents()
            plus_state = 1 if not next_agent else 0
            max_pac_min_ghost = self.max_value if not next_agent else self.min_value

            next_action = max_pac_min_ghost(
                next_state, alpha, beta, state_depth + plus_state, next_agent
            )

            if next_action["cost"] < best_cost:
                best_cost = next_action["cost"]
                best_action = action

            # If we lost to our alpha, lets escape
            if best_cost < alpha:
                break

            beta = min(beta, best_cost)

        # No better matches found
        if best_cost == sys.maxint:
            best_cost = self.evaluationFunction(game_state)
            best_action = Directions.STOP

        return {"cost": best_cost, "direction": best_action}


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
