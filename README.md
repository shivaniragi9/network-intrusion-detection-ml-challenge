# Network Intrusion Detection – GDG on Campus ML Challenge

4-class classification of network flows (**Benign, Availability, Reconnaissance, Spoofing**), scored by **Macro F1** on Kaggle.

- **Final public leaderboard Macro F1:** 0.84811
- **Libraries:** scikit-learn only (no GPU, runs in about 1 minute)
- **Seed:** 42 (fully reproducible; re-running gives an identical `submission.csv`)

## Data
- `train.csv`: 84,713 flows × 84 columns (82 features, `id`, `Target`)
- `test.csv`: 33,161 flows
- Class counts in train: Availability 80,072 · Spoofing 2,323 · Benign 1,583 · Reconnaissance 735

The data is heavily imbalanced. Availability makes up about 95% of the training rows, and the other three classes together have only 4,641 rows.

## Approach

### 1. Key finding: destination port 21 identifies Availability
In train, 79,698 of the 79,699 rows with `id.resp_p == 21` are Availability, and the one exception is Reconnaissance. Port 21 therefore gives an almost perfect rule for most of the Availability class, so I predict Availability for every row with `id.resp_p == 21`.

### 2. Why not let a model learn Availability from flow features?
There is a train/test distribution shift on reply traffic:

| | share of flows with `bwd_pkts_tot == 0` (no reply) |
|---|---|
| Train Availability flows | 0.1% |
| Test port-21 flows | 88.1% |

A model that learns "Availability = flows that get replies" from train would reject most test port-21 flows and label them Reconnaissance or Benign. The port is a more stable signal than the flow-level reply statistics, so the port rule decides Availability.

### 3. Model for the other three classes
A 3-class model is trained **only on Benign / Reconnaissance / Spoofing rows** (4,641 rows). The 80k Availability rows are left out, so they don't dominate training or the class balance.

Two models are averaged by summing their predicted probabilities:
- `HistGradientBoostingClassifier` (250 iterations, learning rate 0.06, `class_weight="balanced"`, native categorical handling for `proto` and `service`)
- `ExtraTreesClassifier` (300 trees, `class_weight="balanced"`, categories as integer codes)

Why this pair:
- `class_weight="balanced"` matters because Macro F1 gives each class equal weight. Reconnaissance, the smallest class, counts as much as the rest.
- A boosted model and a bagged random-split model make different kinds of errors, so averaging them gives more stable predictions than either alone.
- Both are fast on a few thousand rows and need no heavy tuning.

Categorical encoding: the vocabulary of `proto` and `service` is built from train and test together, so categories that only appear in test are handled consistently.

### 4. Final prediction
1. Take the argmax of the averaged 3-class probabilities.
2. Overwrite every row with `id.resp_p == 21` as **Availability**.

Predicted class shares on test: Benign 31.8%, Availability 31.4%, Reconnaissance 20.8%, Spoofing 16.1%.

## Limitations
- The port-21 rule is a data-driven rule, not something the model learned. It works because of how this dataset was generated, and it would not generalize to real network traffic.
- Availability traffic on other ports would be missed. Train has no such cases, so I couldn't test this.
- I did not run cross-validation for this final version, so the leaderboard score is the only evidence of how well it works.

## How to run
```bash
pip install -r requirements.txt
mkdir data && cp /path/to/{train,test,sample_submission}.csv data/
python solution.py          # writes submission.csv
```
The script finds the data folder itself (it searches `/kaggle/input` and the current directory) and checks the output against `sample_submission.csv` (columns, IDs, order). On Kaggle it writes to `/kaggle/working/submission.csv`.

## Files
- `solution.py`: full pipeline (the final code used for the submission)
- `submission.csv`: predictions submitted to Kaggle
- `requirements.txt`: dependencies
- Data files are not included, so get them from the Kaggle challenge page.
