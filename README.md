# PKD
Official code for "Prompt-aware Knowledge Distillation for Joint Instance-level and Bag-level Learning in Whole Slide Image Classification"

## Dataset Splits

The dataset split files are stored in the following directories:

| Directory | Dataset |
| --- | --- |
| `task_camelyon_tumor_vs_normal` | CAMELYON16 |
| `task_grading_panda` | PANDA |
| `task_subtyping_brca` | TCGA-BRCA |
| `task_subtyping_nsclc` | TCGA-NSCLC |

Each training script reads `splits_<fold>.csv` from `--split_dir`. The CSV columns contain slide IDs, with empty entries ignored:

- CAMELYON16 uses `train` and `val`; its test slides are loaded separately from the configured testing directories.
- PANDA, TCGA-BRCA, and TCGA-NSCLC use `train`, `val`, and `test`.

Use the slide-ID CSV files, rather than the accompanying `*_bool.csv` or `*_descriptor.csv` files. Keep the training, validation, and test sets disjoint when preparing custom splits. `--k` controls how many existing split files are processed; it does not generate new splits.

## Prerequisites

Run all commands from the repository root so that the relative checkpoint, split, and output paths resolve correctly.

### Python environment and GPU

Install a CUDA-enabled PyTorch build and a compatible torchvision build for your system. The other packages imported by the training scripts, dataset loaders, and bundled CONCH implementation can be installed with:

```bash
python -m pip install numpy pandas scikit-learn tensorboardX tqdm timm transformers huggingface_hub pillow
```

The repository does not provide pinned dependency versions. Use versions compatible with the bundled `conch/` code, and record the environment used for an experiment. In particular, the scripts use `torch.load(..., weights_only=False)`, and the bundled model utilities depend on PyTorch APIs.

Use one NVIDIA GPU per training process. The current feature-based training branches explicitly move modules and attention masks to CUDA, so `--cpu` does not enable CPU-only training. Multi-GPU execution is not implemented correctly for these branches: the legacy `DataParallel` block refers to `model.features`, which is not defined in the default feature-based branch.

Select the GPU with a single ID, such as `--device 0` or `--device 1`. `util.setup_runtime()` sets `CUDA_VISIBLE_DEVICES` from this argument; the selected GPU is then addressed internally as `cuda:0`. The scripts overwrite `--modeldevice` with `--device`, so setting `--modeldevice` separately has no effect.

### Pretrained CONCH weights

Obtain the pretrained `conch_ViT-B-16` weights as described in the [CONCH model card](checkpoints/CONCH/README.md), and place the checkpoint at:

```text
checkpoints/CONCH/pytorch_model.bin
```

All six scripts load this path in `load_conch()`. Keep the bundled model configuration and tokenizer files under `conch/open_clip_custom/` available as well. The NumPy feature files described below must be prepared before training; these scripts do not extract features from raw whole slide images.

## Data Preparation

### Configure local data paths

The dataset loaders contain absolute paths from the development environment. Before running a script, update the paths in the corresponding loader to point to your local data. `--split_dir` changes only the CSV directory; it does not change patch or feature locations.

| Training script(s) | Dataset loader | Paths to configure |
| --- | --- | --- |
| `train_CAMELYON16_PKD.py` | [dataset_CAMELYON16_new.py](Clinical_Dataset/dataset_CAMELYON16_new.py) | Training/validation `root_dir` and `self.feat_dir`; test `root_dir` and the hard-coded testing feature glob in `CAMELYON_16_feat_test`. The test loader appends `testing` to its `root_dir`. |
| All three `train_PANDA_Binary_Gleason*_PKD.py` scripts | [dataset_PANDA_4classes_new.py](Clinical_Dataset/dataset_PANDA_4classes_new.py) | `root_dir` and `self.feat_dir` in both `PANDA_feat` and `PANDA_feat_test`. |
| `train_TCGA_BRCA_PKD.py` | [dataset_TCGA_BRCA_new.py](Clinical_Dataset/dataset_TCGA_BRCA_new.py) | `self.feat_dir` in both the training/validation and test classes. |
| `train_TCGA_NSCLC_PKD.py` | [dataset_TCGA_NSCLC_new.py](Clinical_Dataset/dataset_TCGA_NSCLC_new.py) | `self.feat_dir` in both the training/validation and test classes. |

Keep a trailing `/` on roots that the loaders concatenate directly with slide IDs. Slide directory names must match the IDs in the split files. For CAMELYON16, `normal_001` becomes `normal_001_neg`, and `tumor_001` becomes `tumor_001_pos`. The TCGA training/validation loaders also apply this suffix rule if an ID contains `normal` or `tumor`; their test loaders use IDs directly.

### Feature and label files

Organize the precomputed features in one directory per slide:

```text
<feature_root>/
  <slide_id>/
    patches.npy
    patches_label.npy                       # CAMELYON16 and PANDA
    patches_corresponding_slide_label.npy    # PANDA and TCGA
```

`patches.npy` must contain one 512-dimensional feature vector per patch, with shape `(N, 512)` after any singleton feature axes are squeezed. Feature rows and label rows must refer to the same patches in the same order.

| Dataset | Required labels in each slide directory |
| --- | --- |
| CAMELYON16 | `patches_label.npy`: `(N,)` binary patch labels, with `0` for normal and `1` for tumor. The bag loader derives the slide label from the maximum patch label. |
| PANDA | `patches_label.npy`: `(N, 3)` binary indicators ordered as `[Gleason 3, Gleason 4, Gleason 5]`. `patches_corresponding_slide_label.npy`: slide-level indicators in the same order, broadcastable to `(N, 3)`, for example a `(3,)` vector. A patch negative for all three targets has `[0, 0, 0]`. |
| TCGA-BRCA / TCGA-NSCLC | `patches_corresponding_slide_label.npy`: `(N,)` binary slide labels repeated for every patch in the slide. All patches in a slide must carry the same slide label. Patch-level ground-truth labels are not required by these loaders. |

The TCGA bag prompts are ordered as `[invasive ductal carcinoma, invasive lobular carcinoma]` for BRCA and `[lung adenocarcinoma, lung squamous cell carcinoma]` for NSCLC. Their bag losses and evaluation use score channel `1` as the positive class, so encode these slide labels as `0/1` in the same order.

CAMELYON16 and PANDA also require the corresponding patch-image directories under `root_dir`, containing the `.jpg` patches used to count samples. The number of JPEG patches per slide must match the feature and label row counts. CAMELYON16 additionally uses `_pos` in positive patch filenames for statistics and filtering; positive slides must contain positive patches. For its fixed test set, keep the slides under the testing patch root and the testing feature root consistent, because the test loader scans those locations separately.

The loaders eagerly load features into RAM, and the training scripts create multiple instance/bag dataset copies. Allow enough host memory for these copies. GPU memory also depends on the largest slide, because each bag loader processes a whole slide at once; reducing `--batch_size` only reduces the number of patches in an instance batch.

## Running the Training Scripts

After configuring the paths and preparing the data, run the appropriate command below. Each example trains on split indices `0` through `4` sequentially for 50 epochs per split, then evaluates the selected checkpoints on the test set. The commands use the default feature-based architectures and explicitly specify the split directories.

### CAMELYON16

This script performs binary normal/tumor classification and reports both instance-level and bag-level test metrics. Training and validation come from the split CSV; testing uses the separate fixed test directories.

```bash
python train_CAMELYON16_PKD.py \
  --arch Feat_CAMELYON \
  --split_dir ./task_camelyon_tumor_vs_normal \
  --device 0 --workers 0 \
  --epochs 50 --batch_size 512 --lr 0.0001 --wd -5 \
  --seed 10 --k 5 --k_start 0 --k_end 5 \
  --comment PKD_CAMELYON16
```

### PANDA: Gleason 3, 4, and 5

The three PANDA scripts train separate binary presence/absence tasks for their respective Gleason patterns. They share the same split files and three-column label format; each script selects its target column. Run each task independently. They do not require checkpoints from one another and do not directly output the six-class ISUP grade.

```bash
python train_PANDA_Binary_Gleason3_PKD.py \
  --arch Feat_PANDA \
  --split_dir ./task_grading_panda \
  --device 0 --workers 0 \
  --epochs 50 --batch_size 512 --lr 0.0001 --wd -5 \
  --seed 10 --k 5 --k_start 0 --k_end 5 \
  --comment PKD_PANDA_Gleason3
```

```bash
python train_PANDA_Binary_Gleason4_PKD.py \
  --arch Feat_PANDA \
  --split_dir ./task_grading_panda \
  --device 0 --workers 0 \
  --epochs 50 --batch_size 512 --lr 0.0001 --wd -5 \
  --seed 10 --k 5 --k_start 0 --k_end 5 \
  --comment PKD_PANDA_Gleason4
```

```bash
python train_PANDA_Binary_Gleason5_PKD.py \
  --arch Feat_PANDA \
  --split_dir ./task_grading_panda \
  --device 0 --workers 0 \
  --epochs 50 --batch_size 512 --lr 0.0001 --wd -5 \
  --seed 10 --k 5 --k_start 0 --k_end 5 \
  --comment PKD_PANDA_Gleason5
```

### TCGA-BRCA

This script trains binary breast cancer subtype classification and reports bag-level test metrics.

```bash
python train_TCGA_BRCA_PKD.py \
  --arch Feat_BRCA \
  --split_dir ./task_subtyping_brca \
  --device 0 --workers 0 \
  --epochs 50 --batch_size 512 --lr 0.0001 --wd -5 \
  --seed 10 --k 5 --k_start 0 --k_end 5 \
  --comment PKD_TCGA_BRCA
```

### TCGA-NSCLC

This script trains binary lung cancer subtype classification and reports bag-level test metrics. Pass `--split_dir` explicitly: its current default is an absolute development-machine path, rather than the split directory in this repository. Its default seed is `20`; the other five scripts default to `10`.

```bash
python train_TCGA_NSCLC_PKD.py \
  --arch Feat_NSCLC \
  --split_dir ./task_subtyping_nsclc \
  --device 0 --workers 0 \
  --epochs 50 --batch_size 512 --lr 0.0001 --wd -5 \
  --seed 10 --k 5 --k_start 0 --k_end 5 \
  --comment PKD_TCGA_NSCLC
```

### Common arguments and selecting folds

| Argument | Default | Meaning / usage |
| --- | --- | --- |
| `--epochs` | `50` | Training epochs per split; use a positive value so that validation can save checkpoints. |
| `--batch_size` | `512` | Instance/patch batch size. Bag batch size is fixed at `1`. |
| `--lr` | `0.0001` | Initial learning rate. |
| `--wd` | `-5` | Base-10 exponent of weight decay: the optimizer receives `10 ** wd`, so `--wd -5` means `1e-5`. |
| `--workers` | `0` | DataLoader worker count. Start with `0` when checking data loading and memory usage. |
| `--k` | `5` | Number of split indices to process when `--k_end` is left at `-1`. |
| `--k_start` | `-1` | First split index, inclusive. `-1` resolves to `0`. |
| `--k_end` | `-1` | Last split boundary, exclusive. `-1` resolves to `--k`. |

For a single-split run, replace `--k_start 0 --k_end 5` in any example with `--k_start 0 --k_end 1`. For split `2` only, use `--k_start 2 --k_end 3`. Keep `0 <= k_start < k_end <= k` and ensure every requested `splits_<fold>.csv` exists. Retain the task-specific `Feat_*` architecture shown above; the other legacy architecture choices do not initialize the complete PKD training pipeline.

Some descriptions printed by `--help` contain legacy defaults; the defaults above follow the actual argument definitions. Seeds improve repeatability, but `util.setup_runtime()` enables CuDNN benchmarking and disables deterministic mode, so bitwise reproducibility is not guaranteed.

## Outputs and Important Notes

| Script | TensorBoard log root | Checkpoint directory |
| --- | --- | --- |
| `train_CAMELYON16_PKD.py` | `runs_CAMELYON/` | `results_CAMELYON16_checkpoints/` |
| `train_PANDA_Binary_Gleason3_PKD.py` | `runs_PANDA_gleason3/` | `results_PANDA_checkpoints_gleason3/` |
| `train_PANDA_Binary_Gleason4_PKD.py` | `runs_PANDA_gleason4/` | `results_PANDA_checkpoints_gleason4/` |
| `train_PANDA_Binary_Gleason5_PKD.py` | `runs_PANDA_gleason5/` | `results_PANDA_checkpoints_gleason5/` |
| `train_TCGA_BRCA_PKD.py` | `runs_TCGA_BRCA/` | `results_TCGA_BRCA_checkpoints_new/` |
| `train_TCGA_NSCLC_PKD.py` | `runs_TCGA_NSCLC/` | `results_TCGA_NSCLC_checkpoints_new/` |

- Each script alternates teacher and student training with validation, then automatically evaluates its selected checkpoints. There is no command-line evaluation-only mode. CAMELYON16 and PANDA save separate checkpoint sets selected by validation instance AUC and validation bag AUC; the TCGA scripts save the set selected by validation bag AUC.
- Each checkpoint set contains separate `.pt` state dictionaries for the encoder, teacher, teacher head, distillation head, and student. Keep the matching files together. Filenames include the split index, and PANDA filenames also include the Gleason target.
- A matching `*_threshold_*.json` file stores the threshold selected by maximum validation macro-F1 over `0.01, 0.02, ..., 0.99` at the saved checkpoint's epoch. Test evaluation requires this JSON and reuses its threshold. Keep it with its checkpoint set; older checkpoints without threshold files cannot be evaluated by the current evaluation methods without adaptation. Do not select thresholds on the test set.
- Console output includes ROC-AUC, PR-AUC, accuracy, macro-F1, sensitivity, specificity, and confusion counts, followed by means and standard deviations across the processed splits. Prepare nonempty splits with both positive and negative examples for each evaluated target; otherwise ROC-AUC is undefined and checkpoint selection may fail.
- Checkpoint directories are fixed in the scripts, and filenames do not include the seed or `--comment`. Repeating a task on the same split, including concurrent runs with different seeds, can overwrite checkpoints and threshold files. Archive existing results or edit the `ckpt_dir` argument passed to `Optimizer` before starting a separate experiment.
