// Motion category, from the AddBiomechanics trial-name-based `trial_id_mapping`
// used in notebook/latent_space_analysis.ipynb on the lab workstation (broader
// than a simple walking/running split: walk, run, sit_to_stand, stair, static,
// gait_any, other, ...). Kept as a plain string rather than a fixed union since
// the exact category set comes from that mapping and may grow.
export type MotionLabel = string;

export interface LatentPoint {
  x: number;
  y: number;
  label: MotionLabel;
  /** Pose-frame preview shown on hover. */
  thumbnailUrl?: string;
}

// Exported via scripts/sample_full_dofs_thumbnails.py (workstation) — reads
// frames directly (not through AddBiomechanicsDataset's DataLoader path,
// which only exposes the 23-DOF VAE-input subset), capturing the FULL
// 37-DOF Rajagopal2015 pos vector (pelvis through wrist) per sampled frame,
// since the curated 23-DOF subset silently omits pelvis, subtalar, mtp, and
// wrist — fine for VAE training, not for posing a full-body render.
//
// plot/skeleton_frames/_latent_thumbs_prepare_poses.py (local) then converts
// each frame into this project's own model's convention, verified directly
// against both models' .osim SpatialTransform XML rather than assumed:
//   - Pelvis rotation: Rajagopal composes intrinsic Z(tilt)->X(list)->
//     Y(rotation); this project's model composes intrinsic Y(rotation)->
//     X(obliquity)->Z(tilt) — same 3 physical angles, reversed axis order.
//     Fixed via an actual rotation-matrix decompose/recompose (scipy
//     Rotation, round-trip verified exact), not a linear approximation.
//   - Knee flexion: opposite sign convention from Rajagopal (confirmed
//     against src/vaemodel.py's own `scaling[:,[3,10]] = -1` for this same
//     conversion elsewhere in the project) — negated.
//   - Hip/ankle/subtalar/mtp/lumbar/shoulder/elbow/pro_sup: verified to use
//     the same axis, order and sign in both models — copied through as-is.
//   - Pelvis_ty (height) copied directly (plain translation, no rotation-
//     order ambiguity); pelvis_tx/tz (absolute position in the recording's
//     world frame) left at 0 — not meaningful for a single static pose.
//
// plot/skeleton_frames/_latent_thumbs_transforms.py (local, OpenSim FK) +
// _latent_thumbs_render_blender.py (local, Blender) then render each pose
// with this project's real sipp_generic_runmad_smoothsphere.osim bone mesh —
// the same one every other render on this site uses — with a per-pose
// adaptive camera (computed from each pose's actual mesh bounding box, +10%
// margin) since arbitrary dataset poses don't stay within the narrow
// envelope the project's other, fixed-camera gait-cycle renders assume, and
// no floor plane (pelvis height isn't a real "standing on this floor"
// position for an arbitrary static pose).
//
// 1500 points, stratified across trial_id_mapping categories (capped per
// category, rare ones just take all available — "sit to stand"/"stair" have
// only a handful of frames in the whole dataset), t-SNE'd over just these
// 1500 rather than a larger random sample, so the whole displayed embedding
// is thumbnail-covered.
import { asset } from '../lib/asset';

export const LATENT_DATA_URL: string | null = asset('media/latent_embedding.json');
