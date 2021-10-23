import numpy as np
from gym import utils
from mme.envs.mujoco_env import MujocoEnv


class SparseWalker2dEnv(MujocoEnv, utils.EzPickle):

    def __init__(self):
        MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]

        # --------- Dense Reward ---------
        # alive_bonus = 1.0
        # reward = ((posafter - posbefore) / self.dt)
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()

        # --------- Sparse Reward ---------
        # a reward +1 is given for every time the agent moves forward over a specific number of units.
        if posafter - self.init_qpos[0] > 1.:
            reward=1.
        else:
            reward=0.

        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20