import numpy as np
import torch

from PersistencePython import cubePers


def compute_dgm_force(lh_dgm, gt_dgm):
    # get persistence list from both diagrams
    lh_pers = lh_dgm[:, 1] - lh_dgm[:, 0]
    gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]

    # more lh dots than gt dots

    # print(lh_pers.shape)
    # print(lh_pers.size)
    # print(gt_pers.shape)
    assert lh_pers.size > gt_pers.size

    # check to ensure that all gt dots have persistence 1
    tmp = gt_pers > 0.999

    assert tmp.sum() == gt_pers.size

    gt_n_holes = gt_pers.size  # number of holes in gt

    # get "perfect holes" - holes which do not need to be fixed, i.e., find top
    # lh_n_holes_perfect indices
    # check to ensure that at least one dot has persistence 1; it is the hole
    # formed by the padded boundary
    # if no hole is ~1 (ie >.999) then just take all holes with max values
    tmp = lh_pers > 0.999  # old: assert tmp.sum() >= 1
    # print(type(tmp))
    if np.sum(tmp) >= 1:
        # if tmp.sum >= 1:
        # n_holes_to_fix = gt_n_holes - lh_n_holes_perfect
        lh_n_holes_perfect = tmp.sum()
        idx_holes_perfect = np.argpartition(lh_pers, -lh_n_holes_perfect)[
                            -lh_n_holes_perfect:]
    else:
        idx_holes_perfect = np.where(lh_pers == lh_pers.max())[0]

    # find top gt_n_holes indices
    idx_holes_to_fix_or_perfect = np.argpartition(lh_pers, -gt_n_holes)[
                                  -gt_n_holes:]

    # the difference is holes to be fixed to perfect
    idx_holes_to_fix = list(
        set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

    # remaining holes are all to be removed
    idx_holes_to_remove = list(
        set(range(lh_pers.size)) - set(idx_holes_to_fix_or_perfect))

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    # TODO values below this are small dents so dont fix them; tune this value?
    pers_thd = 0.03
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)

    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / np.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / np.sqrt(2.0)

    return force_list, idx_holes_to_fix, idx_holes_to_remove


def get_critical_points(f):
    """
    Computes persistence diagram and critical points in a 2D function (can be N-dim). Generates only 1D homology dots
    and critical points.
    """

    assert len(f.shape) == 2  # f has to be 2D function
    dim = 2

    # pad the function with a few pixels of minimum values
    # this way one can compute the 1D topology as loops
    # remember to transform back to the original coordinates when finished
    pad_width = 2
    pad_value = f.min()
    f_padded = np.pad(f.cpu().detach().numpy(), pad_width, 'constant', constant_values=pad_value.cpu().detach().numpy())

    # call persistence code to compute diagrams
    # loads PersistencePython.so (compiled from C++); should be in current dir
    persistence_result = cubePers(np.reshape(f_padded, f_padded.size).tolist(), list(f_padded.shape), 0.001)

    # only take 1-dim topology, first column of persistence_result is dimension
    persistence_result_filtered = np.array(list(filter(lambda x: x[0] == 1, persistence_result)))

    # persistence diagram (second and third columns are coordinates)
    dgm = persistence_result_filtered[:, 1:3]

    # critical points
    birth_cp_list = persistence_result_filtered[:, 4:4 + dim]
    death_cp_list = persistence_result_filtered[:, 4 + dim:]

    # when mapping back, shift critical points back to the original coordinates
    birth_cp_list = birth_cp_list - pad_width
    death_cp_list = death_cp_list - pad_width

    return dgm, birth_cp_list, death_cp_list


def get_topo_loss(likelihood, gt, topo_size=100):
    """
    Calculate the topology loss of the predicted image and ground truth image
    Warning: To make sure the topology loss is able to back-propagation, likelihood
    tensor requires to clone before detach from GPUs. In the end, you can hook the
    likelihood tensor to GPUs device.

    Args:
        likelihood:   The likelihood pytorch tensor.
        gt        :  The groundtruth of pytorch tensor.
        topo_size :  The size of the patch is used. Default: 100

    Returns:
        loss_topo :   The topology loss value (tensor)

    """

    topo_cp_weight_map = torch.zeros_like(likelihood, dtype=torch.float)
    topo_cp_ref_map = torch.zeros_like(likelihood, dtype=torch.float)

    for y in range(0, likelihood.shape[0], topo_size):
        for x in range(0, likelihood.shape[1], topo_size):

            lh_patch = likelihood[y:min(y + topo_size, likelihood.shape[0]), x:min(x + topo_size, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_size, gt.shape[0]), x:min(x + topo_size, gt.shape[1])]

            if torch.min(lh_patch) == 1 or torch.max(lh_patch) == 0:
                continue
            if torch.min(gt_patch) == 1 or torch.max(gt_patch) == 0:
                continue

            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh = get_critical_points(lh_patch)
            pd_gt, bcp_gt, dcp_gt = get_critical_points(gt_patch)

            pers_thd_gt = 0.03

            if pd_gt.shape[0] > 0:  # number of critical points (n, 2)
                gt_pers = pd_gt[:, 1] - pd_gt[:, 0]
                gt_pers_valid = gt_pers[np.where(gt_pers > pers_thd_gt)]
            else:
                gt_pers_valid = np.array([])

            if pd_lh.shape[0] <= gt_pers_valid.shape[0]:
                continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt)

            if len(idx_holes_to_fix) > 0:
                for hole_index in idx_holes_to_fix:
                    if int(bcp_lh[hole_index][0]) >= 0 and int(bcp_lh[hole_index][0]) < likelihood.shape[0] and int(bcp_lh[hole_index][1]) >= 0 and int(bcp_lh[hole_index][1]) < likelihood.shape[1]:
                        topo_cp_weight_map[y + int(bcp_lh[hole_index][0]), x + int(bcp_lh[hole_index][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_index][0]), x + int(bcp_lh[hole_index][1])] = 0
                    if int(dcp_lh[hole_index][0]) >= 0 and int(dcp_lh[hole_index][0]) < likelihood.shape[0] and int(dcp_lh[hole_index][1]) >= 0 and int(dcp_lh[hole_index][1]) < likelihood.shape[1]:
                        topo_cp_weight_map[y + int(dcp_lh[hole_index][0]), x + int(dcp_lh[hole_index][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_index][0]), x + int(dcp_lh[hole_index][1])] = 1

            if len(idx_holes_to_remove) > 0:
                for hole_index in idx_holes_to_remove:
                    if int(bcp_lh[hole_index][0]) >= 0 and int(bcp_lh[hole_index][0]) < likelihood.shape[0] and int(bcp_lh[hole_index][1]) >= 0 and int(bcp_lh[hole_index][1]) < likelihood.shape[1]:
                        topo_cp_weight_map[y + int(bcp_lh[hole_index][0]), x + int(bcp_lh[hole_index][1])] = 1  # push birth to death  # push to diagonal
                        if int(dcp_lh[hole_index][0]) >= 0 and int(dcp_lh[hole_index][0]) < likelihood.shape[0] and int(dcp_lh[hole_index][1]) >= 0 and int(dcp_lh[hole_index][1]) < likelihood.shape[1]:
                            topo_cp_ref_map[y + int(bcp_lh[hole_index][0]), x + int(bcp_lh[hole_index][1])] = likelihood[int(dcp_lh[hole_index][0]), int(dcp_lh[hole_index][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_index][0]), x + int(bcp_lh[hole_index][1])] = 1
                    if int(dcp_lh[hole_index][0]) >= 0 and int(dcp_lh[hole_index][0]) < likelihood.shape[0] and int(dcp_lh[hole_index][1]) >= 0 and int(dcp_lh[hole_index][1]) < likelihood.shape[1]:
                        topo_cp_weight_map[y + int(dcp_lh[hole_index][0]), x + int(dcp_lh[hole_index][1])] = 1  # push death to birth # push to diagonal
                        if int(bcp_lh[hole_index][0]) >= 0 and int(bcp_lh[hole_index][0]) < likelihood.shape[0] and int(bcp_lh[hole_index][1]) >= 0 and int(bcp_lh[hole_index][1]) < likelihood.shape[1]:
                            topo_cp_ref_map[y + int(dcp_lh[hole_index][0]), x + int(dcp_lh[hole_index][1])] = likelihood[int(bcp_lh[hole_index][0]), int(bcp_lh[hole_index][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_index][0]), x + int(dcp_lh[hole_index][1])] = 0

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((likelihood * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()

    return loss_topo
