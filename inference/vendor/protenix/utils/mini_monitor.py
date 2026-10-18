# Modified for the CORAL inference distribution (2026); see inference/NOTICE.
"""Score the mini-rollout that training already computes and throws away.

`N_step_mini_rollout: 20` runs a real 20-step sampler every training step under `no_grad`, then
`.detach()`s it; its only consumer is the confidence head, which our data gates off entirely
(`resolution = -1.0` fails the `[0.1, 4.0]` window, so every confidence loss is zeroed). So the one
signal that does track assembly quality is computed for free and discarded.

Measured on four pfLevAll checkpoints, a 20-step rollout reproduces the 200-step verdict 4/4, with the
in-plane protofilament offset separating 41-43 A (healthy) from 1.1-1.3 A (interleaved) -- a 30-fold
gap with no overlap. Loss over the same events moves 0.13%, inside a 6.24% per-step noise floor.

Reports the three orthogonal numbers, because each alone is blind to one failure mode:
  nn      fraction of chains whose nearest neighbour is in the same declared protofilament
          -- catches interleaving, but reads 100% for end-to-end stacking
  inplane component of the PF1->PF2 offset perpendicular to the fibril axis (GT ~44 A)
          -- catches under-separation, which nn cannot see
  axial   component along the axis (GT ~2 A) -- catches end-to-end stacking
  lay     mean same-protofilament layer spacing (GT ~4.8 A) -- catches disintegration
"""
import torch


@torch.no_grad()
def mini_assembly_stats(coordinate_mini, input_feature_dict, crop_tol=0.8, resid_tol=0.8):
    """Return (nn_pct, inplane, axial, layer_spacing) or None if this sample is not a 2-PF fibril."""
    pf = input_feature_dict.get("pf_id", None)
    layer = input_feature_dict.get("layer_id", None)
    a2t = input_feature_dict.get("atom_to_token_idx", None)
    if pf is None or layer is None or a2t is None:
        return None

    x = coordinate_mini
    while x.dim() > 2:
        x = x[0]
    if x.dim() != 2 or x.shape[-1] != 3:
        return None

    pf = pf.flatten().long()
    layer = layer.flatten().long()
    a2t = a2t.flatten().long()
    if a2t.numel() != x.shape[0] or int(a2t.max()) >= pf.numel():
        return None

    apf = pf[a2t]
    alay = layer[a2t]

    # Exclude ligand atoms. On the ligand-bearing arm the rise graph put some ligands into
    # protofilament 2, so "the PF1 vs PF2 chain-centroid distance" was comparing a 588-atom protein
    # chain against a 104-atom ligand, which produced impossible readings (layer spacing 1.30 A,
    # axial 11 A) while the protein-only arm read normally. pf_id alone is not enough; is_ligand
    # has to gate it.
    _lg = input_feature_dict.get("is_ligand", None)
    if _lg is not None:
        _lg = _lg.flatten().bool()
        if _lg.numel() == pf.numel():
            apf = torch.where(_lg[a2t], torch.zeros_like(apf), apf)
        elif _lg.numel() == a2t.numel():
            apf = torch.where(_lg, torch.zeros_like(apf), apf)
    # pf_id is three-state: 1/2 are the two protofilaments, 0 means "not applicable"
    if not ((apf == 1).any() and (apf == 2).any()):
        return None

    cen, grp, sizes, masks = [], [], [], []
    # residue_index is per TOKEN, like pf_id and layer_id; guard the length rather than assume it,
    # because indexing a per-atom array by atom_to_token_idx silently returns the wrong residues.
    ri = input_feature_dict.get("residue_index", None)
    resid = None
    if ri is not None:
        ri = ri.flatten().long().to(x.device)
        if ri.numel() == pf.numel():
            resid = ri[a2t]
    for g in (1, 2):
        for l in torch.unique(alay[apf == g]).tolist():
            m = (apf == g) & (alay == l)
            k = int(m.sum())
            if k < 20:  # a chain the crop kept only a sliver of
                continue
            cen.append(x[m].float().mean(0))
            grp.append(g)
            sizes.append(k)
            masks.append(m)
    if sum(1 for g in grp if g == 1) < 2 or sum(1 for g in grp if g == 2) < 2:
        return None

    # Reject rollouts the crop truncated unevenly.
    #
    # Training runs at crop_size 384 with 100% SpatialInterfaceCropping, while a full assembly is a
    # median 1248 tokens: only 13% fit whole, the mean sample keeps 44%, and each chain is left with
    # ~38 of its ~140 residues -- preferentially the ones near the interface. A group's centroid is
    # then the centroid of a FRAGMENT, and the two protofilaments' fragments are different parts of
    # their chains. Measured on synthetic crops of one identical structure whose true in-plane offset
    # is 44.0 A, the reading moved to 30.5 / 57.5 / 71.0 A depending only on which residues survived.
    #
    # There is no way to correct this from inside a cropped sample: restricting to the residues common
    # to every chain makes the centroids comparable but they are then the centroids of a handful of
    # interface residues, which is not the protofilament separation either. So filter instead of
    # correcting, and let the caller see how many were dropped.
    if min(sizes) < crop_tol * max(sizes):
        return None

    # Equal group sizes are not enough: a crop that leaves each protofilament the same NUMBER of
    # residues but a different PART of the chain still makes the centroids incomparable. Synthetic
    # check on one structure whose true offset is 44.0 A -- keeping PF1's first quarter and PF2's last
    # quarter leaves both groups the same size and reads 71.0 A. So also require the groups to cover
    # the same residue indices, which is the only thing that makes their centroids the same point of
    # the chain.
    if resid is not None:
        sets = [frozenset(resid[m].tolist()) for m in masks]
        common = frozenset.intersection(*sets)
        union = frozenset.union(*sets)
        if not union or len(common) < resid_tol * len(union):
            return None
    C = torch.stack(cen)
    G = torch.tensor(grp, device=C.device)

    X1 = C[G == 1]
    X2 = C[G == 2]
    ax = torch.linalg.svd(X1 - X1.mean(0))[2][0]
    ax = ax / ax.norm()

    D = torch.cdist(C, C)
    D.fill_diagonal_(float("inf"))
    nn = 100.0 * float((G[D.argmin(1)] == G).float().mean())

    v = X2.mean(0) - X1.mean(0)
    axial = float((v @ ax).abs())
    inplane = float((v - (v @ ax) * ax).norm())

    sp = []
    for Xg in (X1, X2):
        if Xg.shape[0] >= 2:
            sp.append(torch.sort(Xg @ ax)[0].diff().abs())
    lay = float(torch.cat(sp).mean()) if sp else float("nan")
    return nn, inplane, axial, lay


_MINI_SKIP = 0


def log_mini_assembly(coordinate_mini, input_feature_dict, step=None):
    """Print one line per step. Never raises -- a monitor must not be able to kill a training run."""
    try:
        r = mini_assembly_stats(coordinate_mini, input_feature_dict)
    except Exception as e:  # noqa: BLE001
        print("[MINIMON] error %s" % type(e).__name__, flush=True)
        return
    if r is None:
        global _MINI_SKIP
        _MINI_SKIP += 1
        if _MINI_SKIP % 25 == 0:
            print("[MINIMON] skipped=%d (not a clean 2-PF sample, or the crop truncated it unevenly)"
                  % _MINI_SKIP, flush=True)
        return
    nn, inplane, axial, lay = r
    print(
        "[MINIMON] step=%s nn=%.0f inplane=%.1f axial=%.1f lay=%.2f"
        % ("?" if step is None else step, nn, inplane, axial, lay),
        flush=True,
    )


_MINI_DUMP_N = 0


@torch.no_grad()
def dump_mini_pdb(coordinate_mini, input_feature_dict, out_dir, every=50):
    """Write the mini-rollout to a CA-only PDB every `every`-th call.

    MINIMON reports numbers, and numbers alone cannot separate "narrow but ordered" from "splayed" --
    checkpoint 4749 scored in-plane 13.8 A and looked, when finally rendered, like two disordered arcs
    rather than two tidy columns. Dumping the structure makes the live run renderable without spending
    a separate inference job on a checkpoint.

    CA-only keeps it a few tens of kB. Never raises.
    """
    global _MINI_DUMP_N
    try:
        _MINI_DUMP_N += 1
        if every <= 0 or (_MINI_DUMP_N - 1) % every:
            return
        pf = input_feature_dict.get("pf_id", None)
        layer = input_feature_dict.get("layer_id", None)
        a2t = input_feature_dict.get("atom_to_token_idx", None)
        ca = input_feature_dict.get("is_ligand", None)
        if pf is None or layer is None or a2t is None:
            return
        x = coordinate_mini
        while x.dim() > 2:
            x = x[0]
        a2t = a2t.flatten().long()
        if a2t.numel() != x.shape[0]:
            return
        apf = pf.flatten().long()[a2t]
        alay = layer.flatten().long()[a2t]

        # Exclude ligand atoms. On the ligand-bearing arm the rise graph put some ligands into
        # protofilament 2, so "the PF1 vs PF2 chain-centroid distance" was comparing a 588-atom protein
        # chain against a 104-atom ligand, which produced impossible readings (layer spacing 1.30 A,
        # axial 11 A) while the protein-only arm read normally. pf_id alone is not enough; is_ligand
        # has to gate it.
        _lg = input_feature_dict.get("is_ligand", None)
        if _lg is not None:
            _lg = _lg.flatten().bool()
            if _lg.numel() == pf.numel():
                apf = torch.where(_lg[a2t], torch.zeros_like(apf), apf)
            elif _lg.numel() == a2t.numel():
                apf = torch.where(_lg, torch.zeros_like(apf), apf)
        if not (bool((apf == 1).any()) and bool((apf == 2).any())):
            return
        import os as _os
        _os.makedirs(out_dir, exist_ok=True)
        path = _os.path.join(out_dir, "mini_%06d.pdb" % _MINI_DUMP_N)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWX"
        n = 0
        with open(path, "w") as fh:
            for g in (1, 2):
                for li, l in enumerate(sorted(set(alay[apf == g].tolist()))):
                    m = (apf == g) & (alay == l)
                    idx = int((g - 1) * 12 + li)
                    if idx >= len(letters):
                        continue
                    c = letters[idx]
                    # one pseudo-CA per chain-member atom keeps the file honest about the geometry
                    for r, p in enumerate(x[m].float().tolist(), start=1):
                        n += 1
                        fh.write("ATOM  %5d  CA  ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                                 % (n, c, r, p[0], p[1], p[2]))
            fh.write("END\n")
    except Exception as e:  # a monitor must never kill a run
        print("[MINIDUMP] error %s" % type(e).__name__, flush=True)
