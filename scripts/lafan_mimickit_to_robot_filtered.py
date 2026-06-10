#!/usr/bin/env python3
"""Bridge a MimicKit G1 motion pkl -> SONIC robot_filtered pkl (WBC tracking screen).

MimicKit motion: {loop_mode, fps, frames=(N, 35)} where each row is
  [root_pos(3), root_rot_expmap(3), dof_pos(29)]   (dof in g1.xml / MuJoCo order)
(see scripts/lafan_g1_npz_to_mimickit.py).

SONIC robot_filtered entry needs only pelvis pose + 29 dofs (convert_sequence
reads body_pos_w[:,0] and body_quat_w[:,0] only); smpl_joints is a dummy.

The stored fps stays the TRUE source fps (e.g. 30); SONIC's fk_batch resamples
to the eval target (50 Hz) at load time -- do NOT relabel fps or the motion warps.

Usage (in isaaclab env, cwd = GR00T-WholeBodyControl):
  python <this> --input dependencies/MimicKit/data/motions/g1/lafan_fight_5s.pkl \
      --key lafan_fight --output data/lafan_fight_robot.pkl --joint_order mj
"""
import argparse
import os
import sys

import joblib
import numpy as np
from scipy.spatial import transform

# Import the canonical SONIC converter helpers (DOF axes, joint remap, builders).
_WBC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_WBC, "dependencies", "GR00T-WholeBodyControl"))
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import convert_sequence  # noqa: E402


def mimickit_to_robot_filtered(input_pkl, motion_key, joint_order):
    m = joblib.load(input_pkl)
    frames = np.asarray(m["frames"], dtype=np.float64)  # (N, 35)
    fps = int(m["fps"])
    assert frames.shape[1] == 35, f"expected (N,35), got {frames.shape}"
    n = frames.shape[0]

    root_pos = frames[:, 0:3]                                   # (N, 3)
    root_expmap = frames[:, 3:6]                                # (N, 3) axis-angle
    dof_pos = frames[:, 6:35].astype(np.float32)               # (N, 29)

    # expmap -> quat xyzw -> wxyz (convert_sequence expects body_quat_w in wxyz)
    root_quat_xyzw = transform.Rotation.from_rotvec(root_expmap).as_quat()  # (N,4) xyzw
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    # Only index 0 (pelvis) is read by convert_sequence; rest is unused scaffolding.
    body_pos_w = np.zeros((n, 14, 3), dtype=np.float32)
    body_quat_w = np.zeros((n, 14, 4), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w[:, 0, :] = root_quat_wxyz

    seq = {
        "joint_pos": dof_pos,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "joint_order": joint_order,  # "mj" = use as-is, "il" = reorder MJ_TO_IL
    }
    entry = convert_sequence(seq, fps=fps)  # stores true fps; loader resamples to 50
    return {motion_key: entry}, n, fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--key", required=True, help="motion_key in the output pkl")
    p.add_argument("--output", required=True)
    p.add_argument("--joint_order", default="mj", choices=["mj", "il"])
    a = p.parse_args()

    out, n, fps = mimickit_to_robot_filtered(a.input, a.key, a.joint_order)
    joblib.dump(out, a.output)
    e = out[a.key]
    print(f"[bridge] {a.input} -> {a.output}")
    print(f"[bridge] key={a.key} frames={n} fps={fps} joint_order={a.joint_order}")
    print(f"[bridge] dof{e['dof'].shape} root_trans{e['root_trans_offset'].shape} "
          f"root_rot{e['root_rot'].shape} pose_aa{e['pose_aa'].shape}")
    print(f"[bridge] root_trans z range: {e['root_trans_offset'][:,2].min():.3f}..{e['root_trans_offset'][:,2].max():.3f}")


if __name__ == "__main__":
    main()
