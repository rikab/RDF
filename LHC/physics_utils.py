# standard numerical library imports
import numpy as np
from pyjet import cluster



def WTA_kT_N_4vec(events, N, R, d_ij_cut = None):

    jets = []

    for event in events:

        # Set up 4-vectors
        four_vectors = []
        for particle in event:

            # Important that this is a list () and not a tuple []
            four_vectors.append((particle[0], particle[1], particle[2], particle[3]))
            
        four_vectors = np.array(four_vectors, dtype=[("E", "f8"), ("px", "f8"), ("py", "f8"), ("pz", "f8")])

        # Cluster with kT (p = 1)

        sequence = cluster(four_vectors, R=R, p=1, recomb_scheme = "WTA_modp_scheme", ep=True)
        if d_ij_cut is None:
            subjets = sequence.exclusive_jets(N)
        else:
            subjets = sequence.exclusive_jets_dcut(d_ij_cut)

        output = np.zeros((N, 4))
        for i, subjet in enumerate(subjets[:N]):
            output[i,0] = subjet.e
            output[i,1] = subjet.px
            output[i,2] = subjet.py
            output[i,3] = subjet.pz


        # Normalize
        output[:,0] = np.nan_to_num(output[:,0] / np.sum(output[:,0]))

        jets.append(output)


    return np.array(jets)


def get_WTA_axis(events):

    return WTA_kT_N_4vec(events, 1, 1)[:,0]


def get_angularity_4vec(events, axis, beta):

    total_energies = np.sum(events[:,:,0], axis = 1)
    angles = angle_between_3vec(events[:,:,1:], axis[:,None,1:])

    # print(angles)

    angularities = np.sum(events[:,:,0] * np.power(angles, beta), axis = -1) / total_energies
    return angularities


def angle_between_3vec(vecs1, vecs2):

    norms1 = np.sqrt( np.sum( np.square(vecs1), axis = -1) )
    norms2 = np.sqrt( np.sum( np.square(vecs2), axis = -1) )

    normed1 = np.nan_to_num(vecs1 / norms1[..., None])
    normed2 = np.nan_to_num(vecs2 / norms2[..., None])

    dot_product = np.sum(normed1 * normed2, axis = -1)

    # return dot_product
    angles = np.arccos(dot_product)
    return angles


# def rotate_phi_theta(events, phis, thetas):

    