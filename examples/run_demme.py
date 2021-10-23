import argparse

import tensorflow as tf

from rllab.envs.normalized_env import normalize

from mme.algos import DE_MME
from mme.envs import (
    GymEnv,
    GymEnvDelayed,
    GME_NP_pure,
)

from mme.misc.instrument import run_sac_experiment
from mme.misc.utils import timestamp, unflatten
from mme.policies import GaussianPolicy, UniformPolicy
from mme.misc.sampler import SimpleSampler
from mme.replay_buffers import SimpleReplayBuffer
from mme.value_functions import NNQFunction, NNVFunction
from mme.preprocessors import MLPPreprocessor
from examples.variants import parse_domain_and_task, get_variants, get_variants_delayed, get_variants_sparse

DELAY_CONST = 20
ENVIRONMENTS = {
    'ant': {
        'default': lambda: GymEnv('Ant-v1'),
        'delayed': lambda: GymEnvDelayed('Ant-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseAnt-v1'),
    },
    'hopper': {
        'default': lambda: GymEnv('Hopper-v1'),
        'delayed': lambda: GymEnvDelayed('Hopper-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseHopper-v1'),
    },
    'half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v1'),
        'delayed': lambda: GymEnvDelayed('HalfCheetah-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseHalfCheetah-v1'),
    },
    'walker': {
        'default': lambda: GymEnv('Walker2d-v1'),
        'delayed': lambda: GymEnvDelayed('Walker2d-v1', delay = DELAY_CONST),
        'sparse': lambda: GymEnv('SparseWalker2d-v1'),
    },
    'humanoid-gym': {
        'default': lambda: GymEnv('Humanoid-v1'),
    },
    'humanoid-standup-gym': {
        'default': lambda: GymEnv('HumanoidStandup-v1'),
    },
    '2Dmaze-cont': {
        'pure': GME_NP_pure,
    }
}

AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default=None)
    # env : 'pure' (for 2Dmaze-cont), 'default','sparse','delayed' (for Mujoco tasks)
    parser.add_argument('--task',
                        type=str,
                        choices=AVAILABLE_TASKS,
                        default='delayed')
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian', 'gmm', 'lsp'),
                        default='gaussian')
    # env : 'half-cheetah','hopper','ant','walker','humanoid-gym','humanoid-standup-gym','2Dmaze-cont'
    parser.add_argument('--env', type=str, default='half-cheetah')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha_pi', type=float, default=0.2)
    parser.add_argument('--alpha_Q', type=float, default=2.0)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args

def run_experiment(variant):
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_buffer_params = variant['replay_buffer_params']
    sampler_params = variant['sampler_params']

    task = variant['task']
    domain = variant['domain']

    env = normalize(ENVIRONMENTS[domain][task](**env_params))
    if domain == '2Dmaze-cont':
        with tf.variable_scope("low_level_policy", reuse=True):
            eval_env = normalize(ENVIRONMENTS[domain][task](**env_params))
    else:
        eval_env = None

    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)
    sampler = SimpleSampler(domain=domain, task=task, **sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    rrqf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='rrqf1')
    rrqf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='rrqf2')
    reqf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='reqf1')
    reqf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='reqf2')
    rrvf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='rrvf')
    revf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='revf')

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)

    policy_T = GaussianPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(M,M),
        reparameterize=policy_params['reparameterize'],
        reg=1e-3,
        name='gaussian_policy_target',
    )
    policy_E = GaussianPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(M,M),
        reparameterize=policy_params['reparameterize'],
        reg=1e-3,
        name='gaussian_policy_exploration',
    )

    algorithm = DE_MME(
        base_kwargs=base_kwargs,
        env=env,
        eval_env=eval_env,
        task=task,
        policy=policy_T,
        policy_E=policy_E,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        rrqf1=rrqf1,
        rrqf2=rrqf2,
        reqf1=reqf1,
        reqf2=reqf2,
        rrvf=rrvf,
        revf=revf,
        lr=algorithm_params['lr'],
        alpha_Q=algorithm_params['alpha_Q'],
        scale_reward=1.0/algorithm_params['alpha_pi'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        reparameterize=algorithm_params['reparameterize'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=policy_params['action_prior'],
        save_full_state=False,
    )

    algorithm._sess.run(tf.global_variables_initializer())

    algorithm.train()


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    # TODO: Remove unflatten. Our variant generator should support nested params
    variants = [unflatten(variant, separator='.') for variant in variants]

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        run_params = variant['run_params']
        algo_params = variant['algorithm_params']
        variant['algorithm_params']['alpha_pi'] = args.alpha_pi
        variant['algorithm_params']['alpha_Q'] = args.alpha_Q

        experiment_prefix = variant['prefix'] + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=variant['prefix'], exp_name=args.exp_name, i=i)

        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=run_params['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=run_params['snapshot_mode'],
            snapshot_gap=run_params['snapshot_gap'],
            sync_s3_pkl=run_params['sync_pkl'],
        )


def main():
    args = parse_args()

    domain, task = args.env, args.task

    if (not domain) or (not task):
        domain, task = parse_domain_and_task(args.env)
    if args.task == 'delayed':
        variant_generator = get_variants_delayed(domain=domain, task=task, policy=args.policy, seed = args.seed, gamma = args.gamma)
    elif args.task == 'sparse':
        variant_generator = get_variants_sparse(domain=domain, task=task, policy=args.policy, seed = args.seed, gamma = args.gamma)
    else:
        variant_generator = get_variants(domain=domain, task=task, policy=args.policy, seed = args.seed, gamma = args.gamma)
    launch_experiments(variant_generator, args)

if __name__ == '__main__':
    main()
