import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score

# This contains the predictions written by the user submission's Model.predict()
prediction_dir = os.path.join('/app/input/', 'res')

# Output directory where the final score needs to be saved
score_dir = '/app/output/'

# START CUSTOM CODE
## Path to the held-out labels for the eval data (nominal + all severities
## share the same underlying events/labels). Must be shared with the service
## account mlchallenges, and swapped to the final-phase data when the
## competition moves from Development Phase to Testing Phase.
reference_dir = "REPLACE_ME"  # NRP PVC mount path for held-out labels (secret eval)

def parse_severity(path):
    return int(re.search(r"\d+", os.path.basename(path)).group())

def auc_for(pred_path, labels, num_classes):
    probs = np.load(pred_path)
    if num_classes == 2:
        return roc_auc_score(labels, probs[:, 1])
    return roc_auc_score(labels, probs, multi_class="ovr", average="macro")

labels = np.load(os.path.join(reference_dir, "labels.npy"))
num_classes = int(labels.max()) + 1

severity_files = sorted(glob.glob(os.path.join(prediction_dir, "pred_severity_*.npy")), key=parse_severity)

severities = [0]
aucs = [auc_for(os.path.join(prediction_dir, "pred_nominal.npy"), labels, num_classes)]
for path in severity_files:
    severities.append(parse_severity(path))
    aucs.append(auc_for(path, labels, num_classes))

denom = severities[-1] - severities[0]
robustness_auc = np.trapz(aucs, severities) / denom if denom > 0 else aucs[0]
scores = {"auc": robustness_auc}
# END CUSTOM CODE

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
