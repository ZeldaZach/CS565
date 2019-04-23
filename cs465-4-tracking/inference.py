# inference.py
# ------------
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


import itertools
import util
import random
import busters
import game


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghost_agent):
        """Sets the ghost agent for later access"""
        self.ghostAgent = ghost_agent
        self.index = ghost_agent.index
        self.obs = []  # most recent observation position
        self.legal_positions = []

    def getJailPosition(self):
        return 2 * self.ghostAgent.index - 1, 1

    def getPositionDistribution(self, game_state):
        """
        Returns a distribution over successor positions of the ghost from the
        given game_state.

        You must first place the ghost in the game_state, using setGhostPosition
        below.
        """
        ghost_position = game_state.getGhostPosition(self.index)  # The position you set
        action_dist = self.ghostAgent.getDistribution(game_state)
        dist = util.Counter()
        for action, prob in action_dist.items():
            successor_position = game.Actions.getSuccessor(ghost_position, action)
            dist[successor_position] = prob
        return dist

    def setGhostPosition(self, game_state, ghost_position):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied game_state.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghost_position, game.Directions.STOP)
        game_state.data.agentStates[self.index] = game.AgentState(conf, False)
        return game_state

    def observeState(self, game_state):
        """Collects the relevant noisy distance observation and pass it along."""
        distances = game_state.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, game_state)

    def initialize(self, game_state):
        """Initializes beliefs to a uniform distribution over all positions."""
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legal_positions = [
            p for p in game_state.getWalls().asList(False) if p[1] > 1
        ]
        self.initializeUniformly(game_state)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, game_state):
        """Sets the belief state to a uniform prior belief over all positions."""
        pass

    def observe(self, observation, game_state):
        """Updates beliefs based on the given distance observation and game_state."""
        pass

    def elapseTime(self, game_state):
        """Updates beliefs for a time step elapsing from a game_state."""
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm updates to
    compute the exact belief function at each time step.
    """

    def __init__(self, ghost_agent):
        InferenceModule.__init__(self, ghost_agent)
        self.beliefs = util.Counter()

    def initializeUniformly(self, game_state):
        """Begin with a uniform distribution over ghost positions."""
        for p in self.legal_positions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, game_state):
        """
        Updates beliefs based on the distance observation and Pacman's position.

        The noisyDistance is the estimated Manhattan distance to the ghost you
        are tracking.

        The emissionModel below stores the probability of the noisyDistance for
        any true distance you supply. That is, it stores P(noisyDistance |
        TrueDistance).

        self.legal_positions is a list of the possible ghost positions (you
        should only consider positions that are in self.legal_positions).

        A correct implementation will handle the following special case:
          *  When a ghost is captured by Pacman, all beliefs should be updated
             so that the ghost appears in its prison cell, position
             self.getJailPosition()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).
        """
        noisy_distance = observation
        emission_model = busters.getObservationDistribution(noisy_distance)
        pacman_position = game_state.getPacmanPosition()

        all_possible = util.Counter()
        if noisy_distance is None:
            # Reset the beliefs
            for i in self.beliefs.keys():
                all_possible[i] = 0

            # Ghost found, mark it as certain
            all_possible[self.getJailPosition()] = 1
        else:
            # Noisy is set to a value
            for p in self.legal_positions:
                true_distance = util.manhattanDistance(p, pacman_position)
                if emission_model[true_distance] > 0:
                    all_possible[p] = self.beliefs[p] * emission_model[true_distance]

        all_possible.normalize()
        self.beliefs = all_possible

    def elapseTime(self, game_state):
        """
        Update self.beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position (e.g., for DirectionalGhost).  However, this
        is not a problem, as Pacman's current position is known.

        In order to obtain the distribution over new positions for the ghost,
        given its previous position (oldPos) as well as Pacman's current
        position, use this line of code:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(game_state, oldPos))

        Note that you may need to replace "oldPos" with the correct name of the
        variable that you have used to refer to the previous ghost position for
        which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        newPosDist is a util.Counter object, where for each position p in
        self.legal_positions,

        newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

        (and also given Pacman's current position).  You may also find it useful
        to loop over key, value pairs in newPosDist, like:

          for newPos, prob in newPosDist.items():
            ...

        *** GORY DETAIL AHEAD ***

        As an implementation detail (with which you need not concern yourself),
        the line of code at the top of this comment block for obtaining
        newPosDist makes use of two helper methods provided in InferenceModule
        above:

          1) self.setGhostPosition(game_state, ghostPosition)
              This method alters the game_state by placing the ghost we're
              tracking in a particular position.  This altered game_state can be
              used to query what the ghost would do in this position.

          2) self.getPositionDistribution(game_state)
              This method uses the ghost agent to determine what positions the
              ghost will move to from the provided game_state.  The ghost must be
              placed in the game_state with a call to self.setGhostPosition
              above.

        It is worthwhile, however, to understand why these two helper methods
        are used and how they combine to give us a belief distribution over new
        positions after a time update from a particular position.
        """
        all_possible = util.Counter()
        for position in self.legal_positions:
            # Get the new distance from the ghost
            next_position = self.getPositionDistribution(
                self.setGhostPosition(game_state, position)
            )

            # Update our probabilities
            for next_position, probability in next_position.items():
                all_possible[next_position] += self.beliefs[position] * probability

        all_possible.normalize()
        self.beliefs = all_possible

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, ghost_agent, num_particles=300):
        InferenceModule.__init__(self, ghost_agent)
        self.num_particles = 0
        self.particles = None
        self.setNumParticles(num_particles)

    def setNumParticles(self, num_particles):
        self.num_particles = num_particles

    def initializeUniformly(self, game_state):
        """
        Initializes a list of particles. Use self.num_particles for the number of
        particles. Use self.legal_positions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        # N=4 -> distribution "0 1 2 3 0 1 2 3 ..." to keep it even
        indexes = [
            index % len(self.legal_positions) for index in range(self.num_particles)
        ]

        self.particles = [self.legal_positions[index] for index in indexes]

    def observe(self, observation, game_state):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legal_positions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.getJailPosition()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeUniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattanDistance to calculate the
        distance between a particle and Pacman's position.
        """
        noisy_distance = observation
        emission_model = busters.getObservationDistribution(noisy_distance)
        pacman_position = game_state.getPacmanPosition()

        # Set the jailer since no distance
        if noisy_distance is None:
            self.particles = [self.getJailPosition() for _ in range(self.num_particles)]
            return

        # Generate the new particle distances
        new_particles = util.Counter()
        for particle in self.particles:
            new_particles[particle] += emission_model[
                util.manhattanDistance(pacman_position, particle)
            ]
        new_particles.normalize()

        # Save the new particles
        if new_particles.totalCount():
            self.particles = [
                util.sample(new_particles) for _ in range(self.num_particles)
            ]
        else:
            self.initializeUniformly(game_state)

    def elapseTime(self, game_state):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(game_state, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        # Set the particles with new distributions and samplings
        self.particles = [
            util.sample(
                self.getPositionDistribution(
                    self.setGhostPosition(game_state, particle)
                )
            )
            for particle in self.particles
        ]

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        belief_distribution = util.Counter()
        for particle in self.particles:
            belief_distribution[particle] += 1.0
        belief_distribution.normalize()

        return belief_distribution


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initializeUniformly(self, game_state):
        """Set the belief state to an initial, prior value."""
        if self.index == 1:
            jointInference.initialize(game_state, self.legal_positions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, game_state):
        """Update beliefs based on the given distance observation and game_state."""
        if self.index == 1:
            jointInference.observeState(game_state)

    def elapseTime(self, game_state):
        """Update beliefs for a time step elapsing from a game_state."""
        if self.index == 1:
            jointInference.elapseTime(game_state)

    def getBeliefDistribution(self):
        """Returns the marginal belief over a particular ghost by summing out the others."""
        joint_distribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in joint_distribution.items():
            dist[t[self.index - 1]] += prob
        return dist


class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, num_particles=600):
        self.numGhosts = 0
        self.ghostAgents = []
        self.legal_positions = None
        self.num_particles = 0
        self.particles = 0
        self.setNumParticles(num_particles)

    def setNumParticles(self, num_particles):
        self.num_particles = num_particles

    def initialize(self, game_state, legal_positions):
        """Stores information about the game, then initializes particles."""
        self.numGhosts = game_state.getNumAgents() - 1
        self.ghostAgents = []
        self.legal_positions = legal_positions
        self.initializeParticles()

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.

        Each particle is a tuple of ghost positions. Use self.num_particles for
        the number of particles. You may find the `itertools` package helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space. Look at the `itertools.product` function to get an
        implementation of the Cartesian product.

        Note: If you use itertools, keep in mind that permutations are not
        returned in a random order; you must shuffle the list of permutations in
        order to ensure even placement of particles across the board. Use
        self.legal_positions to obtain a list of positions a ghost may occupy.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        "*** YOUR CODE HERE ***"

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return 2 * i + 1, 1

    def observeState(self, game_state):
        """
        Resamples the set of particles using the likelihood of the noisy
        observations.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell, position
             self.getJailPosition(i) where `i` is the index of the ghost.

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeParticles. After all
             particles are generated randomly, any ghosts that are eaten (have
             noisyDistance of None) must be changed to the jail Position. This
             will involve changing each particle if a ghost has been eaten.

        self.getParticleWithGhostInJail is a helper method to edit a specific
        particle. Since we store particles as tuples, they must be converted to
        a list, edited, and then converted back to a tuple. This is a common
        operation when placing a ghost in jail.
        """
        pacman_position = game_state.getPacmanPosition()
        noisy_distances = game_state.getNoisyGhostDistances()
        if len(noisy_distances) < self.numGhosts:
            return
        emission_models = [
            busters.getObservationDistribution(dist) for dist in noisy_distances
        ]

        "*** YOUR CODE HERE ***"

    def getParticleWithGhostInJail(self, particle, ghost_index):
        """
        Takes a particle (as a tuple of ghost positions) and returns a particle
        with the ghost_index'th ghost in jail.
        """
        particle = list(particle)
        particle[ghost_index] = self.getJailPosition(ghost_index)
        return tuple(particle)

    def elapseTime(self, game_state):
        """
        Samples each particle's next state based on its current state and the
        game_state.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        Then, assuming that `i` refers to the index of the ghost, to obtain the
        distributions over new positions for that single ghost, given the list
        (prevGhostPositions) of previous positions of ALL of the ghosts, use
        this line of code:

          newPosDist = getPositionDistributionForGhost(
             setGhostPositions(game_state, prevGhostPositions), i, self.ghostAgents[i]
          )

        Note that you may need to replace `prevGhostPositions` with the correct
        name of the variable that you have used to refer to the list of the
        previous positions of all of the ghosts, and you may need to replace `i`
        with the variable you have used to refer to the index of the ghost for
        which you are computing the new position distribution.

        As an implementation detail (with which you need not concern yourself),
        the line of code above for obtaining newPosDist makes use of two helper
        functions defined below in this file:

          1) setGhostPositions(game_state, ghostPositions)
              This method alters the game_state by placing the ghosts in the
              supplied positions.

          2) getPositionDistributionForGhost(game_state, ghost_index, agent)
              This method uses the supplied ghost agent to determine what
              positions a ghost (ghost_index) controlled by a particular agent
              (ghostAgent) will move to in the supplied game_state.  All ghosts
              must first be placed in the game_state using setGhostPositions
              above.

              The ghost agent you are meant to supply is
              self.ghostAgents[ghost_index-1], but in this project all ghost
              agents are always the same.
        """
        new_particles = []
        for oldParticle in self.particles:
            new_particle = list(oldParticle)  # A list of ghost positions
            # now loop through and update each entry in new_particle...

            "*** YOUR CODE HERE ***"

            "*** END YOUR CODE HERE ***"
            new_particles.append(tuple(new_particle))
        self.particles = new_particles

    def getBeliefDistribution(self):
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


def getPositionDistributionForGhost(game_state, ghost_index, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    game_state.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghost_position = game_state.getGhostPosition(ghost_index + 1)
    action_dist = agent.getDistribution(game_state)
    dist = util.Counter()
    for action, prob in action_dist.items():
        successor_position = game.Actions.getSuccessor(ghost_position, action)
        dist[successor_position] = prob
    return dist


def setGhostPositions(game_state, ghost_positions):
    """Sets the position of all ghosts to the values in ghostPositionTuple."""
    for index, pos in enumerate(ghost_positions):
        conf = game.Configuration(pos, game.Directions.STOP)
        game_state.data.agentStates[index + 1] = game.AgentState(conf, False)
    return game_state
