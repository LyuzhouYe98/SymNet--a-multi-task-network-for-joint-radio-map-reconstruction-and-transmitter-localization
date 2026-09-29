"""Original sparse neighbor selection and patch statistics."""

from pathlib import Path
import numpy as np


class PatchGraphBuilder:
    def __init__(self, data_root, split, patch_topk, los_root=None):
        self.root = (
            Path(los_root)
            if los_root
            else Path(data_root) / "precomputed_patch_los_ps8"
        )
        self.datatype = split
        self.patch_size = 8
        self.patch_topk = int(patch_topk)
        self.conf_sample_cap = 4.0
        self.lambda_los = 1.0
        self.tau = 0.12
        self.patch_source_mode = "all"
        self.patch_target_mode = "unknown_only"
        self.patch_coarse_alpha = 1.0
        self.patch_bucket_bounds = (50.0, 100.0)
        self.patch_bucket_topm = (12, 12, 8)
        self._los_cache = {}

    def load_los(self, building_id):
        if building_id not in self._los_cache:
            path = self.root / self.datatype / f"{building_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}; run the converter with --with-los."
                )
            with np.load(path, allow_pickle=False) as data:
                if int(data["patch_size"]) != self.patch_size:
                    raise ValueError(f"Expected patch size 8 in {path}")
                self._los_cache[building_id] = {
                    name: data[name]
                    for name in (
                        "patch_los_matrix",
                        "patch_rep_xy",
                        "patch_valid_mask",
                        "patch_hw",
                    )
                }
        return self._los_cache[building_id]

    @staticmethod
    def _patch_id(pi, pj, Wp):
        return pi * Wp + pj

    def _build_patch_graph(self, mask_img, building_image, full_norm, los_pack):
        import math
        import numpy as np

        H, W = mask_img.shape
        p = self.patch_size
        Hp, Wp = (H // p, W // p)
        Np = Hp * Wp
        eps = 1e-06
        coarse_alpha = getattr(self, "patch_coarse_alpha", 1.0)
        bucket_bounds = getattr(self, "patch_bucket_bounds", (50.0, 100.0))
        bucket_topm = getattr(self, "patch_bucket_topm", (12, 12, 8))
        valid_map = 1.0 - building_image > 0.5
        sampled_map = mask_img > 0.5
        patch_type = np.zeros((Np,), dtype=np.int64)
        patch_sample_count = np.zeros((Np,), dtype=np.float32)
        patch_valid_ratio = np.zeros((Np,), dtype=np.float32)
        patch_conf = np.zeros((Np,), dtype=np.float32)
        patch_obs_value = np.zeros((Np,), dtype=np.float32)
        patch_rep_xy = np.full((Np, 2), -1, dtype=np.int32)
        patch_center_xy = np.zeros((Np, 2), dtype=np.float32)
        precomputed_rep_xy = los_pack["patch_rep_xy"]
        precomputed_valid_mask = los_pack["patch_valid_mask"]
        precomputed_hw = los_pack["patch_hw"]
        patch_los_matrix = los_pack["patch_los_matrix"]
        if precomputed_hw[0] != Hp or precomputed_hw[1] != Wp:
            raise ValueError(
                f"patch_hw mismatch: file has {tuple(precomputed_hw.tolist())}, current sample expects {(Hp, Wp)}"
            )
        for pi in range(Hp):
            for pj in range(Wp):
                idx = self._patch_id(pi, pj, Wp)
                y0, y1 = (pi * p, (pi + 1) * p)
                x0, x1 = (pj * p, (pj + 1) * p)
                valid_patch = valid_map[y0:y1, x0:x1]
                sampled_patch = sampled_map[y0:y1, x0:x1]
                signal_patch = full_norm[y0:y1, x0:x1]
                valid_count = float(valid_patch.sum())
                sample_count = float((sampled_patch & valid_patch).sum())
                valid_ratio = valid_count / float(p * p)
                patch_sample_count[idx] = sample_count
                patch_valid_ratio[idx] = valid_ratio
                patch_center_xy[idx] = np.array(
                    [x0 + p / 2.0, y0 + p / 2.0], dtype=np.float32
                )
                if valid_count <= 0:
                    patch_type[idx] = 0
                    patch_conf[idx] = 0.0
                    patch_obs_value[idx] = 0.0
                    continue
                if precomputed_valid_mask[idx]:
                    patch_rep_xy[idx] = precomputed_rep_xy[idx]
                if sample_count > 0:
                    patch_type[idx] = 2
                    vals = signal_patch[sampled_patch & valid_patch]
                    patch_obs_value[idx] = float(vals.mean()) if vals.size > 0 else 0.0
                else:
                    patch_type[idx] = 1
                    patch_obs_value[idx] = 0.0
                patch_conf[idx] = min(
                    1.0, sample_count / max(self.conf_sample_cap, eps)
                )
        valid_idx = np.where(patch_type > 0)[0]
        known_idx = np.where(patch_type == 2)[0]
        unknown_valid_idx = np.where(patch_type == 1)[0]
        if self.patch_source_mode == "known_only":
            source_idx = known_idx
        else:
            source_idx = valid_idx
        if self.patch_target_mode == "all_valid":
            target_idx = valid_idx
        else:
            target_idx = unknown_valid_idx
        if len(source_idx) == 0 or len(target_idx) == 0:
            return {
                "patch_type": patch_type.astype(np.int64),
                "patch_sample_count": patch_sample_count.astype(np.float32),
                "patch_valid_ratio": patch_valid_ratio.astype(np.float32),
                "patch_conf": patch_conf.astype(np.float32),
                "patch_obs_value": patch_obs_value.astype(np.float32),
                "patch_rep_xy": patch_rep_xy.astype(np.int32),
                "patch_center_xy": patch_center_xy.astype(np.float32),
                "patch_edge_index": np.zeros((2, 0), dtype=np.int64),
                "patch_edge_attr": np.zeros((0, 5), dtype=np.float32),
                "patch_edge_weight": np.zeros((0,), dtype=np.float32),
                "patch_hw": np.array([Hp, Wp], dtype=np.int64),
            }
        source_valid_mask = patch_rep_xy[source_idx, 0] >= 0
        target_valid_mask = patch_rep_xy[target_idx, 0] >= 0
        source_idx = source_idx[source_valid_mask]
        target_idx = target_idx[target_valid_mask]
        if len(source_idx) == 0 or len(target_idx) == 0:
            return {
                "patch_type": patch_type.astype(np.int64),
                "patch_sample_count": patch_sample_count.astype(np.float32),
                "patch_valid_ratio": patch_valid_ratio.astype(np.float32),
                "patch_conf": patch_conf.astype(np.float32),
                "patch_obs_value": patch_obs_value.astype(np.float32),
                "patch_rep_xy": patch_rep_xy.astype(np.int32),
                "patch_center_xy": patch_center_xy.astype(np.float32),
                "patch_edge_index": np.zeros((2, 0), dtype=np.int64),
                "patch_edge_attr": np.zeros((0, 5), dtype=np.float32),
                "patch_edge_weight": np.zeros((0,), dtype=np.float32),
                "patch_hw": np.array([Hp, Wp], dtype=np.int64),
            }
        src_rep = patch_rep_xy[source_idx].astype(np.float32)
        dst_rep = patch_rep_xy[target_idx].astype(np.float32)
        src_conf = patch_conf[source_idx].astype(np.float32)
        src_valid_ratio = patch_valid_ratio[source_idx].astype(np.float32)
        dst_valid_ratio = patch_valid_ratio[target_idx].astype(np.float32)
        Ns = src_rep.shape[0]
        Nt = dst_rep.shape[0]
        diff = src_rep[:, None, :] - dst_rep[None, :, :]
        dist_pix = np.sqrt(np.sum(diff * diff, axis=-1), dtype=np.float32)
        max_dist = np.float32(np.sqrt(H * H + W * W) + eps)
        dist_norm = dist_pix / max_dist
        src_log_conf = np.log(src_conf + eps).astype(np.float32)[:, None]
        coarse_score = src_log_conf - coarse_alpha * dist_norm
        same_mask = source_idx[:, None] == target_idx[None, :]
        coarse_score[same_mask] = -1000000000.0
        b0, b1 = bucket_bounds
        m0, m1, m2 = bucket_topm
        edge_src, edge_dst = ([], [])
        edge_attr = []
        edge_weight = []
        for t_col in range(Nt):
            j = int(target_idx[t_col])
            rep_j = patch_rep_xy[j]
            dst_vr = float(patch_valid_ratio[j])
            dcol_pix = dist_pix[:, t_col]
            scol = coarse_score[:, t_col]
            mask0 = dcol_pix < b0
            mask1 = (dcol_pix >= b0) & (dcol_pix < b1)
            mask2 = dcol_pix >= b1
            cand_rows = []

            def pick_top_rows(mask, m):
                idxs = np.where(mask)[0]
                if idxs.size == 0 or m <= 0:
                    return np.empty((0,), dtype=np.int64)
                if idxs.size <= m:
                    sub = idxs
                else:
                    sub_scores = scol[idxs]
                    keep_local = np.argpartition(sub_scores, kth=idxs.size - m)[-m:]
                    sub = idxs[keep_local]
                sub = sub[np.argsort(scol[sub])[::-1]]
                return sub.astype(np.int64)

            cand_rows.extend(pick_top_rows(mask0, m0).tolist())
            cand_rows.extend(pick_top_rows(mask1, m1).tolist())
            cand_rows.extend(pick_top_rows(mask2, m2).tolist())
            if len(cand_rows) > 0:
                seen = set()
                ordered = []
                for r in cand_rows:
                    if r not in seen:
                        seen.add(r)
                        ordered.append(r)
                cand_rows = ordered
            cand_final = []
            for r in cand_rows:
                i = int(source_idx[r])
                rep_i = patch_rep_xy[i]
                d_ij_pix = float(dist_pix[r, t_col])
                d_ij_norm = float(dist_norm[r, t_col])
                los_ij = float(patch_los_matrix[i, j])
                d_eff = d_ij_norm * (1.0 + self.lambda_los * (1.0 - float(los_ij)))
                w = float(src_conf[r]) * math.exp(-d_eff / max(self.tau, eps))
                if w <= 0.0:
                    continue
                dx_ij = float((rep_i[0] - rep_j[0]) / max(1, W - 1))
                dy_ij = float((rep_i[1] - rep_j[1]) / max(1, H - 1))
                attr = np.array(
                    [d_ij_norm, float(los_ij), float(src_conf[r]), dx_ij, dy_ij],
                    dtype=np.float32,
                )
                cand_final.append((i, w, attr))
            if len(cand_final) == 0:
                continue
            cand_final.sort(key=lambda x: x[1], reverse=True)
            cand_final = cand_final[: self.patch_topk]
            for i, w, attr in cand_final:
                edge_src.append(i)
                edge_dst.append(j)
                edge_weight.append(w)
                edge_attr.append(attr)
        if len(edge_src) == 0:
            patch_edge_index = np.zeros((2, 0), dtype=np.int64)
            patch_edge_attr = np.zeros((0, 5), dtype=np.float32)
            patch_edge_weight = np.zeros((0,), dtype=np.float32)
        else:
            patch_edge_index = np.stack(
                [
                    np.array(edge_src, dtype=np.int64),
                    np.array(edge_dst, dtype=np.int64),
                ],
                axis=0,
            )
            patch_edge_attr = np.stack(edge_attr, axis=0).astype(np.float32)
            patch_edge_weight = np.array(edge_weight, dtype=np.float32)
        return {
            "patch_type": patch_type.astype(np.int64),
            "patch_sample_count": patch_sample_count.astype(np.float32),
            "patch_valid_ratio": patch_valid_ratio.astype(np.float32),
            "patch_conf": patch_conf.astype(np.float32),
            "patch_obs_value": patch_obs_value.astype(np.float32),
            "patch_rep_xy": patch_rep_xy.astype(np.int32),
            "patch_center_xy": patch_center_xy.astype(np.float32),
            "patch_edge_index": patch_edge_index,
            "patch_edge_attr": patch_edge_attr,
            "patch_edge_weight": patch_edge_weight,
            "patch_hw": np.array([Hp, Wp], dtype=np.int64),
        }
