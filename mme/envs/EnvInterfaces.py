from __future__ import print_function

from cached_property import cached_property
import math
import numpy as np

from rllab import spaces
from rllab.misc import logger
from mme.envs import GridMap

BIG = 1e6


def get_state_block(state):
    x = state[0].item()
    y = state[1].item()

    x_int = np.floor(x)
    y_int = np.floor(y)
    return x_int * 1000 + y_int


def get_two_random_indices(r, c):
    """Return a 2x2 NumPy array, containing two different index pair."""

    res = np.zeros((2, 2), dtype=np.int)

    while (res[0, 0] == res[1, 0] and \
           res[0, 1] == res[1, 1]):
        res[0, 0] = np.random.randint(0, r)
        res[0, 1] = np.random.randint(0, c)
        res[1, 0] = np.random.randint(0, r)
        res[1, 1] = np.random.randint(0, c)

    return res


class GME_NP_pure(GridMap.GridMapEnv):
    def __init__(self, name="", gridMap=None, workingDir="./"):
        super(GME_NP_pure, self).__init__(name, gridMap, workingDir)

        # # Create map.
        # self.map = GridMap.GridMap2D( 10, 20 )

        gm2d = GridMap.GridMap2D(100, 100, outOfBoundValue=0)  # -200
        gm2d.valueStartingBlock = 0  # -0.1
        gm2d.valueEndingBlock = 0
        gm2d.valueNormalBlock = 0  # -0.1
        gm2d.valueObstacleBlock = 0  # -10
        gm2d.initialize()

        # Create a starting block and an ending block.
        startingBlock = GridMap.StartingBlock()
        # endingBlock = GridMap.EndingBlock()

        # Create an obstacle block.
        obstacle = GridMap.ObstacleBlock()

        # Overwrite blocks.
        gm2d.set_starting_block((0, 0))
        # gm2d.set_ending_block((49, 49), endPoint=(49.1, 49.1))
        for i in range(5):
            for j in range(100):
                if not ((j < 25 and j >= 15) or (j < 85 and j >= 75)):
                    gm2d.add_obstacle((48 + i, j))
                    gm2d.add_obstacle((j, 48 + i))
                # gm2d.add_obstacle((i, 23+j))
        #     gm2d.add_obstacle((5, 10))
        #     gm2d.add_obstacle((6, 10))
        # gm2d.add_obstacle((30, 10))

        # indexEndingBlock = gm2d.get_index_ending_block()
        # ebGm2d = gm2d.get_block(indexEndingBlock)

        # print("ebGm2d.is_in_range(19.2, 9.2, 1) = {}".format(ebGm2d.is_in_range(19.2, 9.2, 1)))

        self.map = gm2d

        self.maxStuckCount = 0
        self.stuckPenaltyFactor = 0  # -10
        self.stuckCount = 0
        self.stuckState = None

        self.timestep = 0

        # Member variables for compatibility.
        # self.observation_space = np.array([0, 0]) # self.observation_spac.shape should be a tuple showing the shape of the state variable.

    @cached_property
    def observation_space(self):
        shp = (2,)
        # shp = (3,)
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @cached_property
    def action_space(self):
        shp = (2,)
        ub = 1.0 * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def enable_stuck_check(self, maxStuckCount, penaltyFactor):
        if (maxStuckCount < 0):
            raise GridMap.GridMapException("Max stuck count must be non-negative number.")

        self.maxStuckCount = maxStuckCount
        self.stuckPenaltyFactor = penaltyFactor

    def disable_stuck_check(self):
        self.maxStuckCount = 0
        self.stuckPenaltyFactor = 1.0

    def step(self, action):
        """
        Override super class.
        """

        act = GridMap.BlockCoorDelta(action[0], action[1])

        coor, val, flagTerm, dummy = super(GME_NP_pure, self).step(act)

        self.timestep += 1
        state = np.array( [coor.x, coor.y], dtype=np.float32)
        # state = np.array([coor.x, coor.y, self.timestep], dtype=np.float32)

        # Check stuck states.
        if (0 != self.maxStuckCount):
            if (self.stuckState is None):
                self.stuckState = state
            else:
                if (state[0] == self.stuckState[0] and \
                        state[1] == self.stuckState[1]):
                    self.stuckCount += 1
                else:
                    self.stuckCount = 0
                    self.stuckState = None

                if (self.maxStuckCount == self.stuckCount):
                    val = self.stuckPenaltyFactor * math.fabs(val)
                    flagTerm = True
        if flagTerm:
            self.timestep = 0
        return state, val, flagTerm, dummy

    def reset(self):
        res = super(GME_NP_pure, self).reset()

        # Clear the stuck states.
        self.stuckCount = 0
        self.stuckState = None
        self.timestep = 0

        return np.array([res.x, res.y])
        # return np.array([res.x, res.y, self.timestep])

    def set_trajectory(self, t):
        """
        t is a numpy ndarray of shape (x, 2). t stores the position (state) history of the agent.
        This function substitutes the self.agentLocs member variable with t. Converts numpy ndarray
        into BlockCoor objects.
        """

        n = t.shape[0]

        temp = []

        for i in range(n):
            temp.append(self.make_a_coor(t[i, 0], t[i, 1]))

        self.agentLocs = temp
        self.nSteps = n

    def random_map(self):
        # There must be a map.
        if (self.map is None):
            raise GridMap.GridMapException("Map must not be None for randomizing.")

        # Get the randomized indices of the staring and ending blocks.
        indices = get_two_random_indices(self.map.rows, self.map.cols)

        # Reset the staring block.
        self.map.set_starting_block(GridMap.BlockIndex(indices[0, 0], indices[0, 1]))

        # Reset the ending block.
        self.map.set_ending_block(GridMap.BlockIndex(indices[1, 0], indices[1, 1]))

    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"]
                for path in paths
            ]
            # logger.record_tabular('AverageForwardProgress', np.mean(progs))
            # logger.record_tabular('MaxForwardProgress', np.max(progs))
            # logger.record_tabular('MinForwardProgress', np.min(progs))
            # logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            0
            # logger.record_tabular('AverageForwardProgress', np.nan)
            # logger.record_tabular('MaxForwardProgress', np.nan)
            # logger.record_tabular('MinForwardProgress', np.nan)
            # logger.record_tabular('StdForwardProgress', np.nan)

    def terminate(self):
        print("end")


class GME_NP_rew(GridMap.GridMapEnv):
    def __init__(self, name="", gridMap=None, workingDir="./"):
        super(GME_NP_rew, self).__init__(name, gridMap, workingDir)

        # # Create map.
        # self.map = GridMap.GridMap2D( 10, 20 )
        # 50 x 50 maze
        gm2d = GridMap.GridMap2D(50, 50, outOfBoundValue=0)  # -200
        gm2d.valueStartingBlock = 0  # -0.1
        gm2d.valueEndingBlock = 1000
        gm2d.valueNormalBlock = 0  # -0.1
        gm2d.valueObstacleBlock = 0  # -10
        gm2d.initialize()

        # Create a starting block and an ending block.
        startingBlock = GridMap.StartingBlock()
        endingBlock = GridMap.EndingBlock()

        # Create an obstacle block.
        obstacle = GridMap.ObstacleBlock()

        # Overwrite blocks.
        gm2d.set_starting_block((0, 0))
        gm2d.set_ending_block((49, 49), endPoint=(49.1, 49.1))
        for i in range(5):
            for j in range(50):
                if not ((j < 15 and j >= 5) or (j < 45 and j >= 35)):
                    gm2d.add_obstacle((23 + i, j))
                    gm2d.add_obstacle((j, 23 + i))

        # # 40 x 40 maze
        # gm2d = GridMap.GridMap2D(40, 40, outOfBoundValue=0)  # -200
        # gm2d.valueStartingBlock = 0  # -0.1
        # gm2d.valueEndingBlock = 1000
        # gm2d.valueNormalBlock = 0  # -0.1
        # gm2d.valueObstacleBlock = 0  # -10
        # gm2d.initialize()
        #
        # # Create a starting block and an ending block.
        # startingBlock = GridMap.StartingBlock()
        # endingBlock = GridMap.EndingBlock()
        #
        # # Create an obstacle block.
        # obstacle = GridMap.ObstacleBlock()
        #
        # # Overwrite blocks.
        # gm2d.set_starting_block((0, 0))
        # gm2d.set_ending_block((39, 39), endPoint=(39.1, 39.1))
        # for i in range(4):
        #     for j in range(40):
        #         if not ((j < 12 and j >= 5) or (j < 35 and j >= 28)):
        #             gm2d.add_obstacle((18 + i, j))
        #             gm2d.add_obstacle((j, 18 + i))

        indexEndingBlock = gm2d.get_index_ending_block()
        ebGm2d = gm2d.get_block(indexEndingBlock)

        print("ebGm2d.is_in_range(19.2, 9.2, 1) = {}".format(ebGm2d.is_in_range(19.2, 9.2, 1)))

        self.map = gm2d

        self.maxStuckCount = 0
        self.stuckPenaltyFactor = 0  # -10
        self.stuckCount = 0
        self.stuckState = None

        # Member variables for compatibility.
        # self.observation_space = np.array([0, 0]) # self.observation_spac.shape should be a tuple showing the shape of the state variable.

    @cached_property
    def observation_space(self):
        shp = (2,)
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @cached_property
    def action_space(self):
        shp = (2,)
        ub = 1.0 * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    def enable_stuck_check(self, maxStuckCount, penaltyFactor):
        if (maxStuckCount < 0):
            raise GridMap.GridMapException("Max stuck count must be non-negative number.")

        self.maxStuckCount = maxStuckCount
        self.stuckPenaltyFactor = penaltyFactor

    def disable_stuck_check(self):
        self.maxStuckCount = 0
        self.stuckPenaltyFactor = 1.0

    def step(self, action):
        """
        Override super class.
        """

        act = GridMap.BlockCoorDelta(action[0], action[1])

        coor, val, flagTerm, dummy = super(GME_NP_rew, self).step(act)

        state = np.array([coor.x, coor.y], dtype=np.float32)

        # Check stuck states.
        if (0 != self.maxStuckCount):
            if (self.stuckState is None):
                self.stuckState = state
            else:
                if (state[0] == self.stuckState[0] and \
                        state[1] == self.stuckState[1]):
                    self.stuckCount += 1
                else:
                    self.stuckCount = 0
                    self.stuckState = None

                if (self.maxStuckCount == self.stuckCount):
                    val = self.stuckPenaltyFactor * math.fabs(val)
                    flagTerm = True

        return state, val, flagTerm, dummy

    def reset(self):
        res = super(GME_NP_rew, self).reset()

        # Clear the stuck states.
        self.stuckCount = 0
        self.stuckState = None

        return np.array([res.x, res.y])

    def set_trajectory(self, t):
        """
        t is a numpy ndarray of shape (x, 2). t stores the position (state) history of the agent.
        This function substitutes the self.agentLocs member variable with t. Converts numpy ndarray
        into BlockCoor objects.
        """

        n = t.shape[0]

        temp = []

        for i in range(n):
            temp.append(self.make_a_coor(t[i, 0], t[i, 1]))

        self.agentLocs = temp
        self.nSteps = n

    def random_map(self):
        # There must be a map.
        if (self.map is None):
            raise GridMap.GridMapException("Map must not be None for randomizing.")

        # Get the randomized indices of the staring and ending blocks.
        indices = get_two_random_indices(self.map.rows, self.map.cols)

        # Reset the staring block.
        self.map.set_starting_block(GridMap.BlockIndex(indices[0, 0], indices[0, 1]))

        # Reset the ending block.
        self.map.set_ending_block(GridMap.BlockIndex(indices[1, 0], indices[1, 1]))

    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"]
                for path in paths
            ]
            # logger.record_tabular('AverageForwardProgress', np.mean(progs))
            # logger.record_tabular('MaxForwardProgress', np.max(progs))
            # logger.record_tabular('MinForwardProgress', np.min(progs))
            # logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            0
            # logger.record_tabular('AverageForwardProgress', np.nan)
            # logger.record_tabular('MaxForwardProgress', np.nan)
            # logger.record_tabular('MinForwardProgress', np.nan)
            # logger.record_tabular('StdForwardProgress', np.nan)

    def terminate(self):
        print("end")