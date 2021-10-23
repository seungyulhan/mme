import numpy as np

from rllab.misc.instrument import VariantGenerator
from mme.misc.utils import flatten, get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'gaussian',
    'reg': 1e-3,
    'action_prior': 'uniform',
    'reparameterize': REPARAMETERIZE
}

GAUSSIAN_POLICY_PARAMS = {
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid-gym': { # 17 DoF
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    '2Dmaze-cont': { # 17 DoF
    },
}

POLICY_PARAMS = {
    'gaussian': {
        k: dict(GAUSSIAN_POLICY_PARAMS_BASE, **v)
        for k, v in GAUSSIAN_POLICY_PARAMS.items()
    },
}

VALUE_FUNCTION_PARAMS = {
    'layer_size': M,
}

ENV_DOMAIN_PARAMS = {
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid-gym': { # 17 DoF
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    '2Dmaze-cont': { # 17 DoF
    },
}

ENV_PARAMS = {
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'ant': { # 8 DoF
    },
    'humanoid-gym': { # 17 DoF
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    '2Dmaze-cont': { # 17 DoF
    },
}

ALGORITHM_PARAMS_BASE = {
    'lr': 3e-4,
    'discount': 0.99,
    'target_update_interval': 1,
    'tau': 0.005,
    'reparameterize': REPARAMETERIZE,

    'base_kwargs': {
        'epoch_length': 1000,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': 1000,
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
    }
}

ALGORITHM_PARAMS = {
    'hopper': { # 3 DoF
        'base_kwargs': {
            'n_epochs': 3e3,
        }
    },
    'half-cheetah': { # 6 DoF
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'base_kwargs': {
            'n_epochs': 5e3,
        }
    },
    'ant': { # 8 DoF
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'humanoid-gym': { # 17 DoF
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'humanoid-standup-gym': { # 17 DoF
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    '2Dmaze-cont': {  # 17 DoF
        'base_kwargs': {
            'n_epochs': 5e2,
        }
    },
}

ALGORITHM_PARAMS_DELAYED = {
    'hopper': { # 3 DoF
        'base_kwargs': {
            'n_epochs': 3e3,
        }
    },
    'half-cheetah': { # 6 DoF
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'base_kwargs': {
            'n_epochs': 5e3,
        }
    },
    'ant': { # 8 DoF
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
}

ALGORITHM_PARAMS_SPARSE = {
    'hopper': { # 3 DoF
        'base_kwargs': {
            'n_epochs': 1e3,
        }
    },
    'half-cheetah': { # 6 DoF
        'base_kwargs': {
            'n_epochs': 1e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'base_kwargs': {
            'n_epochs': 1e3,
        }
    },
    'ant': { # 8 DoF
        'base_kwargs': {
            'n_epochs': 3e3,
            'n_initial_exploration_steps': 10000,
        }
    },
}

REPLAY_BUFFER_PARAMS = {
    'max_replay_buffer_size': 1e6,
}

SAMPLER_PARAMS = {
    'max_path_length': 1000,
    'min_pool_size': 1000,
    'batch_size': 256,
}


RUN_PARAMS_BASE = {
    'seed': [1],
    'snapshot_mode': 'gap',
    'snapshot_gap': 1000,
    'sync_pkl': True,
}

RUN_PARAMS = {
    'hopper': { # 3 DoF
        'snapshot_gap': 1000
    },
    'half-cheetah': { # 6 DoF
        'snapshot_gap': 1000
    },
    'walker': { # 6 DoF
        'snapshot_gap': 1000
    },
    'ant': { # 8 DoF
        'snapshot_gap': 1000
    },
    'humanoid-gym': { # 17 DoF
        'snapshot_gap': 1000
    },
    'humanoid-standup-gym': {  # 17 DoF
        'snapshot_gap': 1000
    },
    '2Dmaze-cont': {  # 2 DoF
        'snapshot_gap': 50000
    },
}


DOMAINS = [
    'hopper', # 3 DoF
    'half-cheetah', # 6 DoF
    'walker', # 6 DoF
    'ant', # 8 DoF
    'humanoid-gym', # 17 DoF # gym_humanoid
    'humanoid-standup-gym', # 17 DoF # gym_humanoid
]

TASKS = {
    'hopper': [
        'default',
        'delayed',
        'sparse',
    ],
    'half-cheetah': [
        'default',
        'delayed',
        'sparse',
    ],
    'walker': [
        'default',
        'delayed',
        'sparse',
    ],
    'ant': [
        'default',
        'delayed',
        'sparse',

    ],
    'humanoid-gym': [
        'default',
    ],
    'humanoid-standup-gym': [
        'default',
    ],
    '2Dmaze-cont': {
        'pure',
    },
}

def parse_domain_and_task(env_name):
    domain = next(domain for domain in DOMAINS if domain in env_name)
    domain_tasks = TASKS[domain]
    task = next((task for task in domain_tasks if task in env_name), 'default')
    return domain, task

def get_variants(domain, task, policy, seed, gamma):
    RUN_PARAMS_BASE['seed'] = seed
    ALGORITHM_PARAMS_BASE['discount'] = gamma
    if domain == '2Dmaze-cont':
        ALGORITHM_PARAMS_BASE['base_kwargs']['eval_deterministic'] = False
    params = {
        'prefix': '{}/{}'.format(domain, task),
        'domain': domain,
        'task': task,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS[domain].get(task, {}),
        'policy_params': POLICY_PARAMS[policy][domain],
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS[domain]
        ),
        'replay_buffer_params': REPLAY_BUFFER_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domain]),
    }

    # TODO: Remove flatten. Our variant generator should support nested params
    params = flatten(params, separator='.')

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg

def get_variants_delayed(domain, task, policy, seed, gamma):
    RUN_PARAMS_BASE['seed'] = seed
    ALGORITHM_PARAMS_BASE['discount'] = gamma
    params = {
        'prefix': '{}/{}'.format(domain, task),
        'domain': domain,
        'task': task,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS[domain].get(task, {}),
        'policy_params': POLICY_PARAMS[policy][domain],
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_DELAYED[domain]
        ),
        'replay_buffer_params': REPLAY_BUFFER_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domain]),
    }

    # TODO: Remove flatten. Our variant generator should support nested params
    params = flatten(params, separator='.')

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg

def get_variants_sparse(domain, task, policy, seed, gamma):
    RUN_PARAMS_BASE['seed'] = seed
    ALGORITHM_PARAMS_BASE['discount'] = gamma
    params = {
        'prefix': '{}/{}'.format(domain, task),
        'domain': domain,
        'task': task,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS[domain].get(task, {}),
        'policy_params': POLICY_PARAMS[policy][domain],
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_SPARSE[domain]
        ),
        'replay_buffer_params': REPLAY_BUFFER_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domain]),
    }

    # TODO: Remove flatten. Our variant generator should support nested params
    params = flatten(params, separator='.')

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg