import numpy as np

from .anneal_functions import make_h_J, make_bqm, dwave_connect, unfix


def anneal(config, iter, env, mu):
    """Controls the annealing and energy selection process.

    Called from train() multiple times. Each call to anneal performs a training step
    on the D-Wave hardware.

    Parameters:
    - config       Configuration struct (hold many useful config params)
    - iter         Declares the current iteration within the training method. This is needed
                   as some configuration parameters are per-iteration arrays.
    - env          Env object that determines all data-processing hyperparameters.
    - mu           Vector of expected values of qubit spins.
    """
    print(f"iteration {iter}")
    h, J = make_h_J(env.C_i, env.C_ij, mu, pow(config.zoom_factor, iter))
    J = config.anneal["prune_method"](J, config.anneal["prune_params"])
    bqm, bqm_nx, fixed_dict = make_bqm(h, J)

    if len(fixed_dict) == np.size(h):
        spin_samples = np.zeros(len(fixed_dict))
        for k, v in fixed_dict.items():
            spin_samples[k] = v

        return [
            np.vstack((spin_samples, spin_samples))
        ]  # change to tile, maybe keep same nreads shape

    spin_samples, energies = dwave_connect(config, iter, env.sampler, bqm, bqm_nx)
    print("min energy = %d, max energy = %d" % (np.amin(energies), np.amax(energies)))

    spin_samples = unfix(spin_samples, np.size(h), fixed_dict=fixed_dict)

    if np.size(spin_samples, axis=0) > config.max_states[iter]:
        excited_states_arr = [spin_samples[-config.max_states[iter] :]]
    else:
        excited_states_arr = [spin_samples]

    return excited_states_arr
