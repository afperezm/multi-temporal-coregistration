import torch
import torch_topological.nn as tt


def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    """
    Compute the persistent diagram of the image

    Args:
        lh_dgm: likelihood persistent diagram.
        gt_dgm: ground truth persistent diagram.
        pers_thresh: Persistent threshold, which also called dynamic value, which measure the difference.
        between the local maximum critical point value with its neighouboring minimum critical point value.
        The value smaller than the persistent threshold should be filtered. Default: 0.03
        pers_thresh_perfect: The distance difference between two critical points that can be considered as
        correct match. Default: 0.99
        do_return_perfect: Return the persistent point or not from the matching. Default: False

    Returns:
        force_list: The matching between the likelihood and ground truth persistent diagram
        idx_holes_to_fix: The index of persistent points that requires to fix in the following training process
        idx_holes_to_remove: The index of persistent points that require to remove for the following training
        process

    """
    lh_pers = torch.abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if gt_dgm.shape[0] == 0:
        gt_pers = None
        gt_n_holes = 0
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size(dim=0)  # number of holes in gt

    if gt_pers is None or gt_n_holes == 0:
        idx_holes_to_fix = torch.tensor([]).to(lh_dgm.device)
        idx_holes_to_remove = torch.range(0, lh_pers.size(dim=0) - 1, dtype=torch.int)
        idx_holes_perfect = torch.tensor([]).to(lh_dgm.device)
    else:
        # check to ensure that all gt dots have persistence 1
        # tmp = torch.gt(gt_pers, pers_thresh_perfect)

        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = torch.gt(lh_pers, pers_thresh_perfect)  # old: assert tmp.sum() >= 1
        lh_pers_sorted_indices = torch.flip(torch.argsort(lh_pers), dims=[0])
        if torch.sum(tmp) >= 1:
            idx_holes_perfect = lh_pers_sorted_indices[:torch.sum(tmp)]
        else:
            idx_holes_perfect = torch.tensor([]).to(lh_dgm.device)

        # find top gt_n_holes indices
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes]

        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = idx_holes_to_fix_or_perfect[
            torch.ne(idx_holes_to_fix_or_perfect.view(1, -1), idx_holes_perfect.view(-1, 1)).all(dim=0)]

        # remaining holes are all to be removed
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:]

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    pers_thd = pers_thresh
    idx_valid = torch.where(torch.gt(lh_pers, pers_thd))[0]
    idx_holes_to_remove = idx_holes_to_remove[
        torch.eq(idx_holes_to_remove.view(1, -1), idx_valid.view(-1, 1)).any(dim=0)]

    force_list = torch.zeros_like(lh_dgm)

    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / torch.sqrt(torch.tensor(2.0))
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / torch.sqrt(torch.tensor(2.0))

    if do_return_perfect:
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect

    return force_list, idx_holes_to_fix, idx_holes_to_remove


def get_critical_points(likelihood):
    """
    Compute the critical points of the image (Value range from 0 -> 1)

    Args:
        likelihood: Likelihood image from the output of the neural networks

    Returns:
        pd_lh:  persistence diagram.
        bcp_lh: Birth critical points.
        dcp_lh: Death critical points.
        Bool:   Skip the process if number of matching pairs is zero.

    """
    lh = 1 - likelihood

    # Compute persistence diagram
    cc_mod = tt.CubicalComplex()
    cc = cc_mod(lh)

    # Return empty tensors and false to skip if there are no pairings for dimension 0
    if len(cc[0].pairing) == 0:
        return torch.tensor([[]]).to(likelihood.device), torch.tensor([[]]).to(likelihood.device), torch.tensor([[]]).to(likelihood.device), False

    # Compute persistence diagram
    pd_lh = lh.view(-1)[torch.mm(cc[0].pairing.view(-1, 2), torch.tensor([[lh.shape[0]], [1]])).view(-1, 2)]
    # Compute birth critical points
    bcp_lh = cc[0].pairing.view(-1, 2)[[i for i in range(0, cc[0].pairing.view(-1, 2).size(dim=0), 2)]]
    # Compute death critical points
    dcp_lh = cc[0].pairing.view(-1, 2)[[i for i in range(1, cc[0].pairing.view(-1, 2).size(dim=0), 2)]]

    return pd_lh, bcp_lh, dcp_lh, True


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
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = get_critical_points(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = get_critical_points(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not pairs_lh_pa:
                continue
            if not pairs_lh_gt:
                continue

            force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt, pers_thresh=0.03)

            if idx_holes_to_fix.size(dim=0) > 0:
                for hole_index in idx_holes_to_fix:
                    if int(bcp_lh[hole_index][0]) >= 0 and int(bcp_lh[hole_index][0]) < likelihood.shape[0] and int(bcp_lh[hole_index][1]) >= 0 and int(bcp_lh[hole_index][1]) < likelihood.shape[1]:
                        topo_cp_weight_map[y + int(bcp_lh[hole_index][0]), x + int(bcp_lh[hole_index][1])] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_index][0]), x + int(bcp_lh[hole_index][1])] = 0
                    if int(dcp_lh[hole_index][0]) >= 0 and int(dcp_lh[hole_index][0]) < likelihood.shape[0] and int(dcp_lh[hole_index][1]) >= 0 and int(dcp_lh[hole_index][1]) < likelihood.shape[1]:
                        topo_cp_weight_map[y + int(dcp_lh[hole_index][0]), x + int(dcp_lh[hole_index][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_index][0]), x + int(dcp_lh[hole_index][1])] = 1

            if idx_holes_to_remove.size(dim=0) > 0:
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
