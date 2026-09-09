import numpy as np
import run_latent_factor_identification as F


def writable_impute_matrix(frame, cols):
    X = frame[cols].to_numpy(dtype=float, copy=True)
    for j in range(X.shape[1]):
        med = np.nanmedian(X[:, j])
        X[~np.isfinite(X[:, j]), j] = med
    return X


F.impute_matrix = writable_impute_matrix

if __name__ == '__main__':
    F.main()
