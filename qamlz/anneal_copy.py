import numpy as np

from .anneal_functions import make_h_J, make_bqm, dwave_connect, unfix


def make_bqm_copy(h, J, C):
    """
    Encodes the logical qubits with the copy method.

    Parameters:
    - h                     Array of qubit-spin weights.
    - J                     Matrix of coupling strengths between qubits.
    - C                     Same as encoding_depth: Defined for NQAC as the number of copies, can be made into
                            something else for other error-correction schemes.
    """
    rows, cols = np.shape(J)
    cp_J = np.zeros((C * rows, C * cols))
    for i in np.arange(C):
        cp_J[i * rows : (i + 1) * rows, i * cols : (i + 1) * cols] = J
    h = np.repeat(h, C)

    return make_bqm(h, cp_J)


def decode_copy(spin_samples, h_len, orig_len, fixed_dict):
    """
    Decodes the qubits encoded with the copy method.

    Parameters:
    - spin_samples          Obtained spins of all reads from D-Wave.
    - enc_h_len             Number of nodes in the graph sent to D-Wave (after copying).
    - orig_len              Number of nodes originally in the graph (before NQAC encoding).
    - fixed_dict            Dictionary of fixed nodes (fixed just prior to sending
                            D-Wave the graph in the dwave_connect function).
    """
    spin_samples = unfix(spin_samples, h_len, fixed_dict)

    multi_spins = np.zeros((np.size(spin_samples, axis=1) // orig_len))
    for i in range(np.size(multi_spins)):
        multi_spins[i] = spin_samples[:, i * orig_len : (i + 1) * orig_len]

    return multi_spins


def anneal(config, iter, env, mu):
    """Controls the annealing and energy selection process.

    TODO: line 151, change to tile?

    Called from train() multiple times. Each call to anneal performs a training step
    on the D-Wave hardware.

    Parameters:
    - config       Configuration struct (hold many useful config params)
    - iter         Declares the current iteration within the training method. This is needed
                   as some configuration parameters are per-iteration arrays.
    - env          Env object that determines all data-processing hyperparameters.
    - mu           Vector of expected values of qubit spins.
    """
    print(f"iteration is {iter}")
    h, J = make_h_J(env.C_i, env.C_ij, mu, pow(config.zoom_factor, iter))
    J = config.anneal["prune_method"](J, config.anneal["prune_params"])

    orig_len = np.size(h)
    bqm, bqm_nx, fixed_dict = make_bqm_copy(
        h, J, config.anneal["anneal_params"]["encoding_depth"]
    )

    if len(fixed_dict) == config.anneal["anneal_params"]["encoding_depth"] * orig_len:
        enc_spin_samples = np.zeros(len(fixed_dict))
        for k, v in fixed_dict.items():
            enc_spin_samples[k] = v

        sub_spin_samples = np.vstack(
            np.split(enc_spin_samples, config.encoding_depth, axis=0)
        )
        spin_samples = np.zeros(orig_len)
        for i in range(orig_len):
            spin_samples[i] = np.sign(np.sum(sub_spin_samples[:, i]))
            if spin_samples[i] == 0:
                spin_samples[i] = np.random.choice([-1, 1])

        return [
            np.vstack((spin_samples, spin_samples))
        ]  # change to tile, maybe keep same nreads shape

    spin_samples, energies = dwave_connect(
        config, iter, env.sampler, bqm, bqm_nx, config.anneal_time
    )

    spin_samples = decode_copy(
        spin_samples,
        config.anneal["anneal_params"]["encoding_depth"] * orig_len,
        orig_len,
        fixed_dict,
    )

    print("min energy = %d, max energy = %d" % (np.amin(energies), np.amax(energies)))

    if np.size(spin_samples, axis=0) > config.max_states[iter]:
        excited_states_arr = [
            np.vstack((spin_samples[0, :], spin_samples[-config.max_states[iter] :, :]))
        ]
    else:
        excited_states_arr = [spin_samples]

    return excited_states_arr
