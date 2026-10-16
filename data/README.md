# Preprocessed data

These are the exact files the released checkpoints were trained and evaluated on, so training, trajectory evaluation and
the ResiFuse readouts run directly after cloning.  Coordinates are in arena units (pixels / `arena_px`: 710 for MABe22,
1827.7 for CalMS21), sampled at 7.5 Hz (every 4th frame of the 30-Hz videos); every window has 20 steps = 10 observed +
10 predicted.

## windows/

| file | `{split}_data` shape | windows (train / val / test) | used for |
|---|---|---|---|
| `dataset_social_r1_w20_fs4_seqsplit.npz` | (N, 20, 3 mice, 4 kp, 2): nose, neck, center_back, tail_base | 31,976 / 3,761 / 4,203 | `err_12kp` |
| `dataset_social_r1_w20_fs4_seqsplit_12kpfull.npz` | (N, 20, 3, 12, 2): all 12 MABe keypoints | the same windows | `err_36kp` |
| `dataset_natural_r1_w20_fs4_test.npz`, `..._12kpfull.npz` | (N, 20, 3, 4 or 12, 2) | test only: 8,137 | natural-window test |
| `dataset_calms21_w20_fs4.npz` | (N, 20, 2 mice, 7 kp, 2) | 22,514 / 2,639 / 13,041 | `err_calms14` |

MABe22 files: event-centred windows from the MABe22 trajectory-track training file, split by 60-s sequence (80 / 10 / 10,
seed 42).  Per-window metadata: `{split}_event_type` (bitmask: 1 approach, 2 contact, 4 anogenital, 8 nose-nose,
16 following, 32 chase; 0 = background window), `{split}_focal` / `_partner` (mouse indices), `_event_step`, `_seq_id`,
`_t_start`, `_lights`, `_chase`, `_activity`, `_min_nb_dist`; `meta_seq_keys` and `meta_split_seq_ids_{split}` give the
sequence split.  The natural test files contain every sliding window (stride 5 steps) of the 160 test sequences.

CalMS21 file: sliding windows (stride 5 steps) of the task-1 videos (62 train / 8 val / 19 test videos).  It also stores the
behaviour label of every step (`{split}_labels`: attack 0, investigation 1, mount 2, other 3; never used in training) and
the scale in `meta_arena_px`.

The `data_prep/` scripts rebuild these files byte for byte from the raw datasets (see the main README, "Data").

## resifuse/

* `finetune108_dlcrf.npy`: the 108 clips of the ResiFuse fine-tuning split (54 / 11 / 43) with the ResiFuse DeepLabCut poses
  in MABe format, `{"sequences": {clip_id: {"keypoints": (1800, 3, 12, 2), "annotations": (2, 1800)}}}`.  Mouse-frames whose
  mean keypoint likelihood is below 0.2 are removed (stored as 0, as in ResiFuse); coordinates are rescaled to the 710-px
  body-length space of the MABe22 training file.  Built by `data_prep/resifuse_convert_poses.py` and `resifuse_subset_clips.py`.
* `finetune_scored_frames.npz`: the frames the ResiFuse Mouse Triplet protocol scores (train 1,840 / val 235 / test 1,328)
  with their 4 labels (chases, huddles, oral contact, oral-genital contact), and ResiFuse's 200-frame fine-tuning windows
  (78 / 14 / 65) used by its BiGRU head.  Built by `eval/resifuse/protocol_frames.py` with the ResiFuse code.

## Sources

Derived from MABe22 (CaltechDATA records 8kdn3-95j37 and rdsa8-rde65), CalMS21 (CaltechDATA record s0vdx-0k302) and the
ResiFuse re-estimated poses (Hugging Face `WeihanLikk/ResiFuse-MABe22`, github.com/BRAINML-GT/ResiFuse).  Use them under the
terms of those datasets and cite them.
