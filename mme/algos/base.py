import abc
import gtimer as gt

import numpy as np

from rllab.misc import logger
from rllab.algos.base import Algorithm

from mme.core.serializable import deep_clone
from mme.misc import tf_utils
from mme.misc.sampler import rollouts


class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            n_train_repeat=1,
            n_initial_exploration_steps=10000,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render=False,
            control_interval=1
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            n_initial_exploration_steps: Number of steps in the beginning to 
                take using actions drawn from a separate exploration policy.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler

        self._n_epochs = int(n_epochs)
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._n_initial_exploration_steps = n_initial_exploration_steps
        self._control_interval = control_interval

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render

        self._sess = tf_utils.get_default_session()

        self._env = None
        self._policy = None
        self._pool = None

    def _train(self, env, eval_env, policy, initial_exploration_policy, pool):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._init_training(env, eval_env, policy, pool)
        if initial_exploration_policy is None:
            self.sampler.initialize(env, policy, pool)
            initial_exploration_done = True
        else:
            self.sampler.initialize(env, initial_exploration_policy, pool)
            initial_exploration_done = False 

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    # TODO.codeconsolidation: Add control interval to sampler
                    if not initial_exploration_done:
                        if self._epoch_length * epoch >= self._n_initial_exploration_steps:
                            self.sampler.set_policy(policy)
                            initial_exploration_done = True
                    self.sampler.sample()
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')
                curr_len = int(epoch*self._epoch_length) % self.sampler.pool._max_buffer_size
                self._policy.log_diagnostics_curr(self.sampler.pool._observations[curr_len:curr_len+self._epoch_length])
                self._evaluate(epoch)

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                # logger.record_tabular('time-train', times_itrs['train'][-1])
                # logger.record_tabular('time-eval', eval_time)
                # logger.record_tabular('time-sample', times_itrs['sample'][-1])
                # logger.record_tabular('time-total', total_time)
                # logger.record_tabular('epoch', epoch)

                self.sampler.log_diagnostics()

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            self.sampler.terminate()

    def _evaluate(self, epoch):
        """Perform evaluation for the current policy.

        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self._policy,
                             self.sampler._max_path_length, self._eval_n_episodes,
                            )

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))

        if self.sampler._domain == '2Dmaze-cont':
            coverage = len(self.sampler._visited_blocks)
            logger.record_tabular('coverage', coverage)
            logger.record_tabular('max-norm', self.sampler._max_norm)

        self._eval_env.log_diagnostics(paths)
        if self._eval_render:
            self._eval_env.render(paths)

        iteration = epoch*self._epoch_length
        batch = self.sampler.random_batch()
        self.log_diagnostics(iteration, batch)

    @abc.abstractmethod
    def log_diagnostics(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self, env, eval_env, policy, pool):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy: Policy instance.
        :return: None
        """

        self._env = env
        if eval_env is None:
            if self._eval_n_episodes > 0:
                # TODO: This is horrible. Don't do this. Get rid of this.
                import tensorflow as tf
                with tf.variable_scope("low_level_policy", reuse=True):
                    self._eval_env = deep_clone(env)
        self._policy = policy
        self._pool = pool

    @property
    def policy(self):
        return self._policy

    @property
    def env(self):
        return self._env

    @property
    def pool(self):
        return self._pool