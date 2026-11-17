# Results

Default = standard threshold (0.5; anomaly: 5 % false alarms on validation ok parts).
High-recall = threshold chosen on validation so that defect recall >= 99%, then applied unchanged to test.

## Split: clean (n_test = 195)

| Method | ROC-AUC | Acc | Recall def | Precision def | F1 def | False alarms | HR recall def | HR false alarms | HR missed | Train s |
|---|---|---|---|---|---|---|---|---|---|---|
| hog_logreg | 0.9934 | 0.9538 | 0.9658 | 0.9576 | 0.9617 | 0.0641 | 1.0000 | 0.2308 | 0 | 0.1 |
| hog_svm_rbf | 0.9997 | 0.9897 | 0.9915 | 0.9915 | 0.9915 | 0.0128 | 0.9915 | 0.0256 | 1 | 1.1 |
| small_cnn | 0.9966 | 0.9744 | 0.9573 | 1.0000 | 0.9782 | 0.0000 | 0.9573 | 0.0000 | 5 | 761.9 |
| resnet18_frozen_logreg | 0.9950 | 0.9692 | 0.9829 | 0.9664 | 0.9746 | 0.0513 | 0.9829 | 0.0513 | 2 | 35.9 |
| resnet18_finetune | 0.9999 | 0.9795 | 0.9658 | 1.0000 | 0.9826 | 0.0000 | 0.9658 | 0.0000 | 4 | 399.7 |
| patchcore_anomaly | 0.9504 | 0.8000 | 0.6752 | 0.9875 | 0.8020 | 0.0128 | 0.9915 | 0.6538 | 1 | 183.4 |
| ensemble_soft_vote | 0.9999 | 0.9949 | 0.9915 | 1.0000 | 0.9957 | 0.0000 | 0.9915 | 0.0000 | 1 | – |
| ensemble_stacking | 0.9991 | 0.9949 | 0.9915 | 1.0000 | 0.9957 | 0.0000 | 0.9915 | 0.0000 | 1 | – |

## Split: official (n_test = 715)

| Method | ROC-AUC | Acc | Recall def | Precision def | F1 def | False alarms | HR recall def | HR false alarms | HR missed | Train s |
|---|---|---|---|---|---|---|---|---|---|---|
| hog_logreg | 0.9927 | 0.9622 | 0.9757 | 0.9651 | 0.9704 | 0.0611 | 0.9845 | 0.1794 | 7 | 1.9 |
| hog_svm_rbf | 0.9999 | 0.9930 | 0.9912 | 0.9978 | 0.9945 | 0.0038 | 0.9912 | 0.0038 | 4 | 38.1 |
| small_cnn | 1.0000 | 0.9524 | 0.9249 | 1.0000 | 0.9610 | 0.0000 | 0.9868 | 0.0000 | 6 | 1595.2 |
| resnet18_frozen_logreg | 0.9985 | 0.9860 | 0.9823 | 0.9955 | 0.9889 | 0.0076 | 0.9823 | 0.0076 | 8 | 299.4 |
| resnet18_finetune | 1.0000 | 0.9972 | 0.9956 | 1.0000 | 0.9978 | 0.0000 | 0.9956 | 0.0000 | 2 | 1448.9 |
| patchcore_anomaly | 0.9608 | 0.8559 | 0.8102 | 0.9557 | 0.8769 | 0.0649 | 0.9912 | 0.4351 | 4 | 1090.8 |
| ensemble_soft_vote | 1.0000 | 0.9958 | 0.9934 | 1.0000 | 0.9967 | 0.0000 | 0.9934 | 0.0000 | 3 | – |
| ensemble_stacking | 1.0000 | 0.9986 | 0.9978 | 1.0000 | 0.9989 | 0.0000 | 0.9978 | 0.0000 | 1 | – |
