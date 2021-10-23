from numbers import Number

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from .base import RLAlgorithm

import gtimer as gt
EPS = 1e-8

class DE_MME(RLAlgorithm, Serializable):

    """
    Disentangled Max-Min Entropy RL (DE-MME)
    """

    def __init__(
            self,
            base_kwargs,

            env,
            eval_env,
            policy,
            policy_E,
            initial_exploration_policy,
            rrqf1,
            rrqf2,
            reqf1,
            reqf2,
            rrvf,
            revf,
            pool,
            task='default',
            plotter=None,
            lr=3e-3,
            alpha_Q=1.0,
            scale_reward=1,
            discount=0.99,
            tau=0.01,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.

            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.

            rrqf1 (`valuefunction`): First R,R Q-function approximator.
            rrqf2 (`valuefunction`): Second R,R Q-function approximator. Usage of two
                Q-functions improves performance by reducing overestimation
                bias.
            rrvf (`ValueFunction`): R,R value function approximator.

            reqf1 (`valuefunction`): First R,E Q-function approximator.
            reqf2 (`valuefunction`): Second R,E Q-function approximator. Usage of two
                Q-functions improves performance by reducing overestimation
                bias.
            revf (`ValueFunction`): R,E value function approximator.

            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.

            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.

            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        Serializable.quick_init(self, locals())
        super(DE_MME, self).__init__(**base_kwargs)

        self._env = env
        self._eval_env = eval_env
        self._task = task
        self._policy = policy
        self._policy_E = policy_E
        self._initial_exploration_policy = initial_exploration_policy
        self._rrqf1 = rrqf1
        self._rrqf2 = rrqf2
        self._reqf1 = reqf1
        self._reqf2 = reqf2
        self._rrvf = rrvf
        self._revf = revf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._rqf_lr = lr
        self._rvf_lr = lr
        self._scale_reward = scale_reward
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior
        self._p_min = 0.0
        self._p_max = 0.0
        self._alpha_Q = alpha_Q

        # Reparameterize parameter must match between the algorithm and the
        # policy actions are sampled from.
        assert reparameterize == self._policy._reparameterize
        self._reparameterize = reparameterize

        self._save_full_state = save_full_state

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_target_ops()

        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))


    @overrides
    def train(self):
        """Initiate training of the DE_MME instance."""

        self._train(self._env, self._eval_env, self._policy, self._initial_exploration_policy, self._pool)

    def _init_placeholders(self):
        """Create input placeholders for the DE_MME algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_pl = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Da),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    @property
    def scale_reward(self):
        if callable(self._scale_reward):
            return self._scale_reward(self._iteration_pl)
        elif isinstance(self._scale_reward, Number):
            return self._scale_reward

        raise ValueError(
            'scale_reward must be either callable or scalar')

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """

        self._rrqf1_t = self._rrqf1.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N
        self._rrqf2_t = self._rrqf2.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N
        self._reqf1_t = self._reqf1.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N
        self._reqf2_t = self._reqf2.get_output_for(
            self._observations_ph, self._actions_ph, reuse=True)  # N

        with tf.variable_scope('target'):
            rrvf_next_target_t = self._rrvf.get_output_for(self._next_observations_ph)  # N
            revf_next_target_t = self._revf.get_output_for(self._next_observations_ph)  # N
            self._rrvf_target_params = self._rrvf.get_params_internal()
            self._revf_target_params = self._revf.get_params_internal()

        ys = tf.stop_gradient(
            self.scale_reward * self._rewards_ph +
            (1 - self._terminals_ph) * self._discount * rrvf_next_target_t
        )  # N

        ys2 = tf.stop_gradient(
            (1 - self._terminals_ph) * self._discount * revf_next_target_t
        )  # N

        self._td_loss1_t = 0.5 * tf.reduce_mean((ys - self._rrqf1_t)**2)
        self._td_loss2_t = 0.5 * tf.reduce_mean((ys - self._rrqf2_t)**2)
        self._td_loss3_t = 0.5 * tf.reduce_mean((ys2 - self._reqf1_t)**2)
        self._td_loss4_t = 0.5 * tf.reduce_mean((ys2 - self._reqf2_t)**2)

        rrqf1_train_op = tf.train.AdamOptimizer(self._rqf_lr).minimize(
            loss=self._td_loss1_t,
            var_list=self._rrqf1.get_params_internal()
        )
        rrqf2_train_op = tf.train.AdamOptimizer(self._rqf_lr).minimize(
            loss=self._td_loss2_t,
            var_list=self._rrqf2.get_params_internal()
        )

        reqf1_train_op = tf.train.AdamOptimizer(self._rqf_lr).minimize(
            loss=self._td_loss3_t,
            var_list=self._reqf1.get_params_internal()
        )
        reqf2_train_op = tf.train.AdamOptimizer(self._rqf_lr).minimize(
            loss=self._td_loss4_t,
            var_list=self._reqf2.get_params_internal()
        )

        self._training_ops.append(rrqf1_train_op)
        self._training_ops.append(rrqf2_train_op)
        self._training_ops.append(reqf1_train_op)
        self._training_ops.append(reqf2_train_op)

    def _init_actor_update(self):
        """Create minimization operations for policy and state value functions.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and value functions with gradient descent, and appends them to
        `self._training_ops` attribute.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        See Equations (8, 13) in [1], for further information
        of the value function and policy function update rules.
        """

        actions, log_pi = self._policy.actions_for(observations=self._observations_ph,
                                                   with_log_pis=True)
        actions_E, log_pi_E = self._policy_E.actions_for(observations=self._observations_ph,
                                                   with_log_pis=True)

        self._rrvf_t = self._rrvf.get_output_for(self._observations_ph, reuse=True)  # N
        self._rrvf_params = self._rrvf.get_params_internal()
        self._revf_t = self._revf.get_output_for(self._observations_ph, reuse=True)  # N
        self._revf_params = self._revf.get_params_internal()

        if self._action_prior == 'normal':
            D_s = actions.shape.as_list()[-1]
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(D_s), scale_diag=tf.ones(D_s))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        log_target_rr1 = self._rrqf1.get_output_for(
            self._observations_ph, actions, reuse=True)  # N
        log_target_rr2 = self._rrqf2.get_output_for(
            self._observations_ph, actions, reuse=True)  # N
        log_target_re_T = self._reqf1.get_output_for(
            self._observations_ph, actions, reuse=True)  # N

        log_target_re1 = self._reqf1.get_output_for(
            self._observations_ph, actions_E, reuse=True)  # N
        log_target_re2 = self._reqf2.get_output_for(
            self._observations_ph, actions_E, reuse=True)  # N
        min_log_target_T = tf.minimum(log_target_rr1, log_target_rr2)
        min_log_target_E = tf.minimum(log_target_re1, log_target_re2)

        policy_kl_loss = tf.reduce_mean(log_pi - log_target_rr1 - log_target_re_T)
        policy_E_kl_loss = tf.reduce_mean(log_pi_E - log_target_re1)

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy.name)
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)

        policy_loss = (policy_kl_loss
                       + policy_regularization_loss)

        policy_E_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy_E.name)
        policy_E_regularization_loss = tf.reduce_sum(
            policy_E_regularization_losses)

        policy_E_loss = (policy_E_kl_loss
                       + policy_E_regularization_loss)

        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.

        self._p = log_pi_E

        self._rrvf_loss_t = 0.5 * tf.reduce_mean((
          self._rrvf_t
          - tf.stop_gradient(min_log_target_T)
        )**2)
        self._revf_loss_t = 0.5 * tf.reduce_mean((
          self._revf_t
          - tf.stop_gradient(min_log_target_E + self._alpha_Q * (self._p - tf.minimum(0.0, tf.reduce_min(self._p))) + policy_prior_log_probs)
        )**2)


        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=policy_loss,
            var_list=self._policy.get_params_internal()
        )

        policy_E_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=policy_E_loss,
            var_list=self._policy_E.get_params_internal()
        )

        rrvf_train_op = tf.train.AdamOptimizer(self._rvf_lr).minimize(
            loss=self._rrvf_loss_t,
            var_list=self._rrvf_params
        )
        revf_train_op = tf.train.AdamOptimizer(self._rvf_lr).minimize(
            loss=self._revf_loss_t,
            var_list=self._revf_params
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(policy_E_train_op)
        self._training_ops.append(rrvf_train_op)
        self._training_ops.append(revf_train_op)

    def _init_target_ops(self):
        """Create tensorflow operations for updating target value function."""

        source_params = self._rrvf_params + self._revf_params
        target_params = self._rrvf_target_params + self._revf_target_params

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, eval_env, policy, pool):
        super(DE_MME, self)._init_training(env, eval_env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        p = self._sess.run([self._p, self._training_ops], feed_dict)[0]

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if iteration is not None:
            feed_dict[self._iteration_pl] = iteration

        return feed_dict

    @overrides
    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        rrqf1, rrqf2, rrvf, reqf1, reqf2, revf, td_loss1, td_loss2, td_loss3, td_loss4 = self._sess.run(
            (self._rrqf1_t, self._rrqf2_t, self._rrvf_t, self._reqf1_t, self._reqf2_t, self._revf_t, self._td_loss1_t, self._td_loss2_t, self._td_loss3_t, self._td_loss4_t), feed_dict)
        logger.record_tabular('rrqf1-avg', np.mean(rrqf1))
        logger.record_tabular('rrqf1-std', np.std(rrqf1))
        logger.record_tabular('rrqf2-avg', np.mean(rrqf2))
        logger.record_tabular('rrqf2-std', np.std(rrqf2))
        logger.record_tabular('reqf1-avg', np.mean(reqf1))
        logger.record_tabular('reqf1-std', np.std(reqf1))
        logger.record_tabular('reqf2-avg', np.mean(reqf2))
        logger.record_tabular('reqf2-std', np.std(reqf2))
        logger.record_tabular('mean-qf-diff', np.mean(np.abs(rrqf1-rrqf2)))
        logger.record_tabular('mean-qf-diff2', np.mean(np.abs(reqf1-reqf2)))
        logger.record_tabular('rrvf-avg', np.mean(rrvf))
        logger.record_tabular('rrvf-std', np.std(rrvf))
        logger.record_tabular('revf-avg', np.mean(revf))
        logger.record_tabular('revf-std', np.std(revf))
        logger.record_tabular('mean-sq-bellman-error1', td_loss1)
        logger.record_tabular('mean-sq-bellman-error2', td_loss2)
        logger.record_tabular('mean-sq-bellman-error3', td_loss3)
        logger.record_tabular('mean-sq-bellman-error4', td_loss4)

        self._policy.log_diagnostics(iteration, batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the DE_MME algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        DE_MME instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                'epoch': epoch,
                'policy': self._policy,
                'rrqf1': self._rrqf1,
                'rrqf2': self._rrqf2,
                'rrvf': self._rrvf,
                'policy_E': self._policy_E,
                'reqf1': self._reqf1,
                'reqf2': self._reqf2,
                'revf': self._revf,
                'env': self._env,
            }

        return snapshot

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'rrqf1-params': self._rrqf1.get_param_values(),
            'rrqf2-params': self._rrqf2.get_param_values(),
            'rrvf-params': self._rrvf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'reqf1-params': self._reqf1.get_param_values(),
            'reqf2-params': self._reqf2.get_param_values(),
            'revf-params': self._revf.get_param_values(),
            'policy_E-params': self._policy_E.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._rrqf1.set_param_values(d['rrqf1-params'])
        self._rrqf2.set_param_values(d['rrqf2-params'])
        self._rrvf.set_param_values(d['rrvf-params'])
        self._policy.set_param_values(d['policy-params'])
        self._reqf1.set_param_values(d['reqf1-params'])
        self._reqf2.set_param_values(d['reqf2-params'])
        self._revf.set_param_values(d['revf-params'])
        self._policy_E.set_param_values(d['policy_E-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
