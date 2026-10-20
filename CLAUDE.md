# Independent CurriculumAgent WCCI deployment

Read README.md, docs/ADAPTATIONS.md and hpc/protocol.json. User explicitly authorized
this standalone GitHub publication. Two trained seeds0/1, shared model across
A/B/C/D for each seed,800test episodes total. Keep actual seed count visible.
Never alter frozen configs/ or experiments/ or the shared evaluation harness.
No test-based tuning/seed selection or hiding negative/illegal outcomes.
Separate Python3.10Ray learner from existing Python3.13Grid2Op simulator. No
TensorFlow/NNI/Lightning, no SB3 substitute for native Senior. Teacher artifact
is reused, not newly discovered. No claim of original-paper reproduction.
Slurm qigl auto-assigns memory from CPU count: do not add --mem. Use18cores and
one scheduler-assignedGPU per seed, maximum2seed jobs. Never overwrite physical
CUDA_VISIBLE_DEVICES. Only code/metadata edits and static checks are currently
validated; do not claim a runtime test passed without evidence.
