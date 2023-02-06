import numpy as np

from .anneal_functions import make_h_J, make_bqm, dwave_connect, unfix


def make_bqm_qac(h, J, C, gamma):
    """
    Encodes the logical qubits with the NQAC method.

    Acts like a tiling for the J and con_J matrices -> places J along "main diagonal" of super-matrix
    (matrix of matrices) and con_J in off-diagonal of super-matrix

    TODO:

    Parameters:
    - h                     Array of qubit-spin weights.
    - J                     Matrix of coupling strengths between qubits.
    - fix_vars              Defines whether to fix and remove low-variance qubits, as
                            described in D-Wave's fix_variables documentation.
    - prune_vars            Method used to remove weak couplings.
    - cutoff_percentile     Minimum percentile for couplings to NOT be removed.
    - C                     Same as encoding_depth: Defined for NQAC as the number of copies, can be made into
                            something else for other error-correction schemes.
    - gamma                 Defined for NQAC as the penalty associated with differing
                            spins for encoded-qubits corresponding to the same
                            logical qubit, can be made into something else
                            for other error-correction schemes.
    """
    rows, cols = np.shape(J)
    con_J = (
        J
        + J.T
        + (-gamma) * np.max(np.abs(np.ndarray.flatten(J))) * np.eye(rows, cols, k=0)
    )
    qa_J = np.zeros((C * rows, C * cols))
    for i in np.arange(C):
        for j in np.arange(i, C):
            if i == j:
                qa_J[i * rows : (i + 1) * rows, j * cols : (j + 1) * cols] = J
            else:
                qa_J[i * rows : (i + 1) * rows, j * cols : (j + 1) * cols] = con_J
    qa_h = np.repeat(h, C) * C

    return make_bqm(qa_h, qa_J)


def decode_qac(spin_samples, enc_h_len, orig_len, fixed_dict):
    """
    Decodes the qubits encoded with the NQAC method.

    Parameters:
    - spin_samples          Obtained spins of all reads from D-Wave.
    - enc_h_len             Number of nodes in the graph sent to D-Wave (after NQAC encoding).
    - orig_len              Number of nodes originally in the graph (before NQAC encoding).
    - fix_vars              Boolean of whether to fix any nodes using D-Wave's method.
    - fixed_dict            Dictionary of fixed nodes (fixed just prior to sending
                            D-Wave the graph in the dwave_connect function).
    """
    spin_samples = unfix(spin_samples, enc_h_len, fixed_dict=fixed_dict)
    spin_samples = np.where(spin_samples == 3, 0, spin_samples)
    true_spins = np.zeros((np.size(spin_samples, 0), orig_len))
    for i in range(orig_len):
        qubit_copies = []
        for j in range(np.size(spin_samples, axis=1) // orig_len):
            qubit_copies.append(spin_samples[:, j * orig_len])
        true_spins[:, i] = np.sign(np.sum(np.array(qubit_copies), axis=0))
    true_spins = np.where(true_spins == 0, np.random.choice([-1, 1]), true_spins)

    return true_spins


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
    bqm, bqm_nx, fixed_dict = make_bqm_qac(
        h,
        J,
        config.anneal["anneal_params"]["encoding_depth"],
        config.anneal["anneal_params"]["gamma"],
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

    spin_samples = decode_qac(
        spin_samples,
        config.anneal["anneal_params"]["encoding_depth"] * orig_len,
        orig_len,
        fixed_dict,
    )

    excited_states_arr = []
    if np.size(spin_samples, axis=1) > config.max_states[iter]:
        for i in range(np.size(spin_samples, 0)):
            print(
                "%d: min energy = %d, max energy = %d"
                % (i, np.amin(energies[i]), np.amax(energies[i]))
            )
            excited_states_arr[i] = [
                *spin_samples[i][0, :],
                *spin_samples[i][-config.max_states[iter] :, :],
            ]
    else:
        for i in range(np.size(spin_samples, 0)):
            excited_states_arr[i] = spin_samples[i]

    return excited_states_arr
