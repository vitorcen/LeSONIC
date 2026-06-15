"""UNITREE_G1_SONIC — StarVLA data config for the SONIC motion-token datasets.

A/B baseline vs GR00T N1.7 (see doc/sonic_starvla_swap_brainstorm.html §11):
same frozen GEAR-SONIC WBC, same LAFAN flow3 dataset, StarVLA as the swapped
token producer. Deployed into the starVLA repo via symlink
``examples/UNITREE_G1_SONIC/train_files/data_registry`` (auto-discovered by
starVLA/dataloader/gr00t_lerobot/registry.py).

Key decisions (all argued in the brainstorm doc):
  * action  = motion_token(64) + left/right_hand_joints(7+7) = 78 dims,
    horizon 40 — mirrors the GR00T ``unitree_g1_sonic`` embodiment exactly so
    the open-loop MSE convention is comparable.
  * action normalization = IDENTITY (keys NOT in normalization_modes are
    passed through untouched) — the FSQ token grid k/16 in [-0.5625, 0.5]
    must survive verbatim; any stats-derived scaling moves the bin centres.
  * state   = 7 joint groups (43, finger dims are zero-filled) +
    projected_gravity (3) = 46 dims, min_max — REQUIRED, because QwenPI_v3
    discretizes state into 256 bins fixed on [-1,1]; raw radians (knee > 2)
    would clip into the edge bins and destroy the proprio signal.
  * video   = single ego_view (480x640, _pack_sample resizes to 448).
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class UnitreeG1SonicConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.ego_view"]
    # Same key order as the GR00T unitree_g1_sonic embodiment config.
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.projected_gravity",
    ]
    action_keys = [
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(40))  # action_horizon = 40 (SONIC WBC chunk)

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # State: min_max into [-1,1] (zero-range finger dims map to 0 — handled
        # by the transform's min==max mask). Action: tensor conversion ONLY —
        # no normalization entry means identity, keeping the FSQ grid intact.
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={k: "min_max" for k in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_sonic": UnitreeG1SonicConfig(),
}

# embodiment_tag is read from the DataConfig classvar by the registry.
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    # LAFAN flow3: 8 hand-picked windows / 5777 frames / 8 prompts.
    "sonic_lafan_flow3": [
        ("sonic_vla_lerobot_flow3", 1.0, "unitree_g1_sonic"),
    ],
    # Single-window iteration testbed (episode 5 = "run in a circle", 749 frames):
    # used by the QwenPI_CE head experiments — prove a breakthrough on one motion
    # before scaling back to all 8. Keeps the FULL-dataset stats.json so state
    # normalization is identical to the 8-window runs.
    "sonic_lafan_run3": [
        ("sonic_vla_lerobot_run3", 1.0, "unitree_g1_sonic"),
    ],
    # BonesSeed: the 7 distinct LAFAN motions (dance / forward lunge / macarena /
    # kick / squat / jump on one leg / walk and turn around), 7 episodes / 3815
    # frames / fps 50. Same GR00T embodiment schema (modality.json identical to
    # flow3) so the UnitreeG1SonicConfig above applies verbatim. This is the
    # token-generation testbed for StarVLA QwenGR00T_v2, mirroring the GR00T
    # N1.7 BonesSeed finetune and the FlowDP-SONIC head.
    "sonic_bonesseed": [
        ("sonic_vla_lerobot", 1.0, "unitree_g1_sonic"),
    ],
}
