# Released ERR checkpoints

All three use the same architecture (D = 64, C = 8, k = 3, two-stream attention, edge GRU, intention head read after the
state update, steering term, mode sampling) and the same training recipe: variety loss over K = 5 samples + observed
one-step loss + body-band hinge + 0.01 x hard-target intention cross-entropy, Adam 1e-3 decayed x0.9 every 2 epochs, effective
batch 64, early stopping after 15 epochs without improvement of the validation minADE20 (center_back), seed 42.
Each file is `{epoch, state_dict, args, best_ade}`; `args` is the full training command line, so
`err.checkpoint.load_checkpoint` rebuilds the model without flags.

| file | nodes | training data | trained with | best epoch (val minADE20 cb) | parameters |
|---|---|---|---|---|---|
| `err_12kp.pth.tar` | 3 mice x 4 kp = 12 | MABe22 training file, event-centred windows | the original research trainer (*) | 38 (12.39 px) | 103,421 |
| `err_36kp.pth.tar` | 3 mice x 12 kp = 36 | same windows, all keypoints | `err/train.py`, batch 32 x accum 2 (L40S GPU) | 27 (12.58 px) | 103,485 |
| `err_calms14.pth.tar` | 2 mice x 7 kp = 14 | CalMS21 task 1 training videos (62 train / 8 val) | `err/train.py`, batch 64, `--arena_px 1827.7` (L40S GPU) | 21 (50.45 CalMS px) | 103,445 |

(*) `err_12kp` predates the consolidated trainer.  Its recipe is the one above (`python err/train.py --node_mode all12`
reproduces it); the only differences are that the original trainer had no loss-spike / divergence guards (they never fire on
12-node runs) and drew different random samples for the validation metric.  The model itself is identical:
this code and the original implementation give bit-identical outputs on this checkpoint.

The 12-node model is the one used for the zero-shot CalMS21 transfer (`eval/calms21/embed.py --nodes 4`), the 36-node model
for the MABe22 representation benchmark and the ResiFuse protocol.  Retraining reproduces these numbers only up to GPU
non-determinism: two trainings of the same configuration differed by about 2 % in test minADE20 and by <= 0.002 in the
CalMS21 readout F1 in our runs.
