import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_split_numeric(root, dataset_path):
    data, _ = arff.loadarff(root / dataset_path)
    df = pd.DataFrame(data)
    y_raw = df['class'].apply(lambda v: int(v.decode() if isinstance(v,(bytes,bytearray)) else v)).to_numpy(np.int64)
    # Preserve the numeric class ordering used in the historical Helena probes.
    # Shift by a constant only so model targets are 0..K-1; this does not change
    # class groups or stratified RNG ordering.
    y = y_raw - int(y_raw.min())
    X = df[[f'V{i}' for i in range(1,28)]].to_numpy(np.float32)
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=.30, random_state=42, stratify=y)
    va, te = train_test_split(tmp, test_size=.50, random_state=42, stratify=y[tmp])
    meta, cal = train_test_split(va, test_size=.50, random_state=42, stratify=y[va])
    sc = StandardScaler().fit(X[tr])
    X = sc.transform(X).astype(np.float32)
    assert (len(tr),len(meta),len(cal),len(te)) == (45637,4889,4890,9780)
    print('SPLIT_NUMERIC',len(tr),len(meta),len(cal),len(te),flush=True)
    return X,y,tr,meta,cal,te
