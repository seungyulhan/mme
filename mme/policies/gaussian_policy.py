""" Gaussian mixture policy. """

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from mme.distributions import Normal 
from mme.policies import NNPolicy
from mme.misc import tf_utils

EPS = 1e-6

class GaussianPolicy(NNPolicy, Serializable):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, reparameterize=True, name='gaussian_policy'):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the Gaussian parameters.
            squash (`bool`): If True, squash the Gaussian the gmm action samples
               between -1 and 1 with tanh.
            reparameterize ('bool'): If True, gradients will flow directly through
                the action samples.
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._is_deterministic = False
        self._fixed_h = None
        self._squash = squash
        self._reparameterize = reparameterize
        self._reg = reg

        self.name = name
        self.build()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")

        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations, multi_actions = False, num_actions = 1, latents=None,
                    name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, regularize=False):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            self._distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations,),
                reg=self._reg
            )
        if multi_actions:
            raw_actions = self._distribution.multi_x_t(num_actions)
        else:
            raw_actions = self._distribution.x_t
        actions = tf.tanh(raw_actions) if self._squash else raw_actions

        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            # TODO.code_consolidation: should come from log_pis_for
            if multi_actions:
                log_pis = self._distribution.log_p_action(raw_actions)
            else:
                log_pis = self._distribution.log_p_t
            if self._squash:
                log_pis -= self._squash_correction(raw_actions)
            return actions, log_pis

        return actions
    
    def log_pis_for(self, actions):
        if self._squash:
           raw_actions = tf.atanh(actions - tf.sign(actions) * 1e-3)
           log_pis = self._distribution.log_p_action(raw_actions)
           log_pis -= self._squash_correction(raw_actions)
           return log_pis
        return self._distribution.log_p_action(actions)

    def log_pis_for_raw(self, raw_actions):
        if self._squash:
           log_pis = self._distribution.log_p_action(raw_actions)
           log_pis -= self._squash_correction(raw_actions)
           return log_pis
        return self._distribution.log_p_action(raw_actions)

    def log_pis_for_other_actions(self, observations, actions, latents=None,
                    name=None, reuse=tf.AUTO_REUSE, regularize=False):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            self._distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(observations,),
                reg=self._reg
            )

        if self._squash:
            raw_actions = tf.atanh(actions - tf.sign(actions) * 1e-3)
            log_pis = self._distribution.log_p_action(raw_actions)
            log_pis -= self._squash_correction(raw_actions)
        else:
            log_pis = self._distribution.log_p_action(actions)
        return log_pis

    def build(self):
        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = Normal(
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                reparameterize=self._reparameterize,
                cond_t_lst=(self._observations_ph,),
                reg=self._reg,
            )

        self._ractions = raw_actions = tf.stop_gradient(self.distribution.x_t)
        self._actions = tf.tanh(raw_actions) if self._squash else raw_actions

    @overrides
    def get_actions(self, observations):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns the mean action for the 
        observations. If False, return stochastically sampled action.

        TODO.code_consolidation: This should be somewhat similar with
        `LatentSpacePolicy.get_actions`.
        """
        if self._is_deterministic: # Handle the deterministic case separately.

            feed_dict = {self._observations_ph: observations}

            # TODO.code_consolidation: these shapes should be double checked
            # for case where `observations.shape[0] > 1`
            mu = tf.get_default_session().run(
                self.distribution.mu_t, feed_dict)  # 1 x Da
            if self._squash:
                mu = np.tanh(mu)

            return mu

        return super(GaussianPolicy, self).get_actions(observations)


    def get_action_with_raw(self, observation):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns the mean action for the
        observations. If False, return stochastically sampled action.

        TODO.code_consolidation: This should be somewhat similar with
        `LatentSpacePolicy.get_actions`.
        """
        feed_dict = {self._observations_ph: observation[None]}
        actions, raw_actions = tf.get_default_session().run([self._actions, self._ractions], feed_dict)
        return actions[0], raw_actions[0]

    def _squash_correction(self, actions):
        if not self._squash: return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=-1)

    @contextmanager
    def deterministic(self, set_deterministic=True, latent=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
            latent (`Number`): Value to set the latent variable to over the
                deterministic context.
        """
        was_deterministic = self._is_deterministic

        self._is_deterministic = set_deterministic

        yield

        self._is_deterministic = was_deterministic

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the GMM
        means, component weights, and covariances.
        """

        feeds = {self._observations_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        mu, log_sig, log_pi, raw_action = sess.run(
            (
                self.distribution.mu_t,
                self.distribution.log_sig_t,
                self.distribution.log_p_t,
                self.distribution.x_t,
            ),
            feeds
        )
        ac = np.tanh(raw_action)
        log_pi_ac = log_pi - np.sum(np.log(1 - ac ** 2 + EPS), axis=-1)

        logger.record_tabular('policy-mus-mean', np.mean(mu))
        logger.record_tabular('policy-mus-min', np.min(mu))
        logger.record_tabular('policy-mus-max', np.max(mu))
        logger.record_tabular('policy-mus-std', np.std(mu))
        logger.record_tabular('log-sigs-mean', np.mean(log_sig))
        logger.record_tabular('log-sigs-min', np.min(log_sig))
        logger.record_tabular('log-sigs-max', np.max(log_sig))
        logger.record_tabular('log-sigs-std', np.std(log_sig))
        logger.record_tabular('log-pi-mean', np.mean(log_pi))
        logger.record_tabular('log-pi-max', np.max(log_pi))
        logger.record_tabular('log-pi-min', np.min(log_pi))
        logger.record_tabular('log-pi-std', np.std(log_pi))
        logger.record_tabular('log-pi-ac-mean', np.mean(log_pi_ac))
        logger.record_tabular('log-pi-ac-max', np.max(log_pi_ac))
        logger.record_tabular('log-pi-ac-min', np.min(log_pi_ac))
        logger.record_tabular('log-pi-ac-std', np.std(log_pi_ac))

    def log_diagnostics_curr(self, obs):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the GMM
        means, component weights, and covariances.
        """

        feeds = {self._observations_ph: obs}
        sess = tf_utils.get_default_session()
        mu, log_sig, log_pi, raw_action = sess.run(
            (
                self.distribution.mu_t,
                self.distribution.log_sig_t,
                self.distribution.log_p_t,
                self.distribution.x_t,
            ),
            feeds
        )
        ac = np.tanh(raw_action)
        log_pi_ac = log_pi - np.sum(np.log(1 - ac ** 2 + EPS), axis=-1)

        logger.record_tabular('curr-policy-mus-mean', np.mean(mu))
        logger.record_tabular('curr-policy-mus-min', np.min(mu))
        logger.record_tabular('curr-policy-mus-max', np.max(mu))
        logger.record_tabular('curr-policy-mus-std', np.std(mu))
        logger.record_tabular('curr-log-sigs-mean', np.mean(log_sig))
        logger.record_tabular('curr-log-sigs-min', np.min(log_sig))
        logger.record_tabular('curr-log-sigs-max', np.max(log_sig))
        logger.record_tabular('curr-log-sigs-std', np.std(log_sig))
        logger.record_tabular('curr-log-pi-mean', np.mean(log_pi))
        logger.record_tabular('curr-log-pi-max', np.max(log_pi))
        logger.record_tabular('curr-log-pi-min', np.min(log_pi))
        logger.record_tabular('curr-log-pi-std', np.std(log_pi))
        logger.record_tabular('curr-log-pi-ac-mean', np.mean(log_pi_ac))
        logger.record_tabular('curr-log-pi-ac-max', np.max(log_pi_ac))
        logger.record_tabular('curr-log-pi-ac-min', np.min(log_pi_ac))
        logger.record_tabular('curr-log-pi-ac-std', np.std(log_pi_ac))
