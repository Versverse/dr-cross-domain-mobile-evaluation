# Retained ablation configuration provenance

Five saved configurations are retained; Table 3 displays the four configurations following the augmentation-first sequence. The sampler-only configuration is retained in this provenance table and in the original summary CSV, rather than silently discarded.

Original APTOS_retest cell numbers are one-based. In the curated 02_notebooks/02_论文表3消融.ipynb, markdown headings preserve the original cell numbers: cell 15 trained the augmentation and augmentation-plus-sampler variants and evaluated the retained full checkpoint; cell 16 trained the baseline and sampler-only variants and evaluated existing augmentation-plus-sampler/full checkpoints. Original cell 5 supplied the main full checkpoint. This provenance reconstruction does not rerun training or establish any unrecorded training date. The checkpoint basename is an identity aid, not a public access link.
