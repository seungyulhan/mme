import numpy as np
from gym import utils
from mme.envs.mujoco_env import MujocoEnv


class SparseAntEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        # xposbefore = self.get_body_com("torso")[0]
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        # xposafter = self.get_body_com("torso")[0]
        xposafter = self.model.data.qpos[0, 0]

        # --------- Dense Reward ---------
        # forward_reward = (xposafter - xposbefore)/self.dt
        # ctrl_cost = .5 * np.square(a).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # --------- Sparse Reward ---------
        # a reward +1 is given for every time the agent moves forward over a specific number of units.
        if xposafter - self.init_qpos[0] > 1.:
            reward=1.
        else:
            reward=0.

        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5