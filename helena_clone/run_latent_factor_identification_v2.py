import numpy as np
import run_latent_factor_identification as F


def writable_impute_matrix(frame, cols):
    X = frame[cols].to_numpy(dtype=float, copy=True)
    for j in range(X.shape[1]):
        med = np.nanmedian(X[:, j])
        X[~np.isfinite(X[:, j]), j] = med
    return X


# Preserve the F1 v2 behavior while explicitly re-exporting the frozen F1
# definitions needed by F1.5. This is a compatibility-only change; no
# statistical protocol, feature definition, or threshold is altered.
F.impute_matrix = writable_impute_matrix
GROUPS = F.GROUPS
make_features = F.make_features
residual_mats = F.residual_mats
raw_geometry_features = F.raw_geometry_features
svd_factors = F.svd_factors
match_factors = F.match_factors

if __name__ == '__main__':
    F.main()
