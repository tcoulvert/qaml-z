from .anneal_basic import anneal as anneal_basic
from .anneal_qac import anneal as anneal_qac
from .anneal_copy import anneal as anneal_copy

from .prune_functions import abs_smallest_prune, no_prune


def anneal(config, iter, env, mu):
    """Controls the annealing and energy selection process.

    TODO: split into multiple (two?) functions, depending on regular mode or qac mode

    Called from train() multiple times. Each call to anneal performs a training step
    on the D-Wave hardware.

    Parameters:
    - config       Configuration struct (hold many useful config params)
    - iter         Declares the current iteration within the training method. This is needed
                   as some configuration parameters are per-iteration arrays.
    - env          Env object that determines all data-processing hyperparameters.
    - mu           Vector of expected values of qubit spins.
    """
    if config.anneal["prune_method"] == "abs_smallest":
        config.anneal["prune_method"] = abs_smallest_prune
    elif config.anneal["prune_method"] == "no_prune":
        config.anneal["prune_method"] = no_prune
    else:  # User specified function
        pass

    if config.anneal["anneal_method"] == "basic":
        return anneal_basic(config, iter, env, mu)
    elif config.anneal["anneal_method"] == "qac":
        return anneal_qac(config, iter, env, mu)
    elif config.anneal["anneal_method"] == "copy":
        return anneal_copy(config, iter, env, mu)
    else:  # User specified function
        return config.anneal["anneal_method"](config, iter, env, mu)
