import dimod
import dwave as dw
import minorminer
import networkx as nx
import numpy as np


def make_h_J(C_i, C_ij, mu, sigma):
    """
    Creates the node (h) and edge (J) weightings for use in building the problem graph.

    Parameters:
    - C_i           Intermediate data representation needed for D-Wave's BQM format.
    - C_ij          Intermediate data representation needed for D-Wave's BQM format.
    - mu            Spins of qubits for this iteration.
    - sigma         zoom_factor to the power of the iteration.
    """
    h = 2 * sigma * (np.einsum("ij, j", C_ij, mu) - C_i)
    J = 2 * np.triu(C_ij, k=1) * pow(sigma, 2)

    return h, J


def make_bqm(h, J):
    """
    Creates the BQM and associated nx graph. The two differ in that fixed nodes
    are not removed from the nx graph.

    TODO: Clean up code and add in script tags for Graphs

    Parameters:
    - h                     Array of qubit-spin weights.
    - J                     Matrix of coupling strengths between qubits.
    """
    bqm_nx = nx.from_numpy_array(J)

    node_dict, attr_dict = {}, {}
    for node in bqm_nx:
        attr_dict["h_bias"] = h[node]
        node_dict[node] = attr_dict
    nx.set_node_attributes(bqm_nx, node_dict)

    bqm = dimod.from_networkx_graph(
        bqm_nx,
        vartype="SPIN",
        node_attribute_name="h_bias",
        edge_attribute_name="weight",
    )

    lowerE, fixed_dict = dw.preprocessing.lower_bounds.roof_duality(bqm, strict=True)
    if fixed_dict == {} or len(fixed_dict) < (np.size(h) // 2):
        print("loosely fixed")
        lowerE, fixed_dict = dw.preprocessing.lower_bounds.roof_duality(
            bqm, strict=False
        )
    for i in fixed_dict.keys():
        bqm.fix_variable(i, fixed_dict[i])
        bqm_nx.remove_node(i)

    print(f"num nodes = {bqm.num_variables} \n num edges = {bqm.num_interactions}")

    return bqm, bqm_nx, fixed_dict


def scale_weights(th, tJ, strength):
    """
    Scales the weightings dependent upon the iteration. It is unclear
    what purpose this serves.

    Parameters:
    - th            Embedded h array.
    - tJ            Embedded J matrix.
    - strength      Current-iteration-indexed strengths array.
    """
    for k in list(th.keys()):
        th[k] /= strength
    for k in list(tJ.keys()):
        tJ[k] /= strength

    return th, tJ


def dwave_connect(config, iter, sampler, bqm, bqm_nx):
    """
    Handles all the connection to D-Wave, including:
    - Embedding the problem into a D-Wave architecture
    - Sending the problem to be annealed
    - Receiving the results from the annealing
    - Unembedding the problem from a D-Wave architecture
    - Unpacking the results into the obtained spins and their energies

    TODO: Multiply by a before and after embedding

    Parameters:
    - config            Config object that determines all training hyperparameters.
    - iter              Current iteration of the training.
    - sampler
    - bqm
    - bqm_nx
    - anneal_time
    """

    a = np.sign(np.random.rand(bqm.num_variables) - 0.5)

    embedding = minorminer.find_embedding(bqm_nx, sampler.to_networkx_graph())

    h_dict = nx.classes.function.get_node_attributes(bqm_nx, "h_bias")
    J_dict = nx.classes.function.get_edge_attributes(bqm_nx, "weight")
    # Unsure what the commented-out code is supposed to do
    # for k in h_dict:
    #     h_dict[k] *= a[k]
    # for u, v in J_dict:
    #     J_dict[(u, v)] *= a[u] * a[v]

    th, tJ = dw.embedding.embed_ising(
        h_dict,
        J_dict,
        embedding,
        sampler.adjacency,
    )
    th, tJ = scale_weights(th, tJ, config.strengths[iter])

    qaresult = sampler.sample_ising(
        th,
        tJ,
        num_reads=config.num_reads,
        annealing_time=config.anneal_time,
        answer_mode="histogram",
    )
    unembed_qaresult = dw.embedding.unembed_sampleset(qaresult, embedding, bqm)

    a = np.tile(a, (np.size(unembed_qaresult.record.sample, axis=0), 1))
    samples, energies = (
        unembed_qaresult.record.sample * a,
        unembed_qaresult.record.energy,
    )

    return samples, energies


def unfix(spin_samples, h_len, fixed_dict):
    """
    Unpacks the qubits fixed using fix_vars.

    Parameters:
    - spin_samples          Obtained spins of all reads from D-Wave.
    - h_len                 Number of nodes in the graph sent to D-Wave.
    - fixed_dict            Dictionary of fixed nodes (fixed just prior to sending
                            D-Wave the graph in the dwave_connect function).
    """
    spin_samples = np.zeros((np.size(spin_samples, axis=0), h_len))
    j = 0
    for i in range(h_len):
        if i in fixed_dict:
            spin_samples[:, i] = fixed_dict[i]
        else:
            spin_samples[:, i] = spin_samples[:, j]
            j += 1

    return spin_samples
