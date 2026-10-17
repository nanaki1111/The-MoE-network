import numpy as np

def yc_patch(A, l1, l2, o1, o2):
    n1, n2 = A.shape

    if (n1 - l1) % o1 != 0:
        A = np.concatenate([A, np.zeros((o1 - (n1 - l1) % o1, n2))], axis=0)
    if (n2 - l2) % o2 != 0:
        A = np.concatenate([A, np.zeros((A.shape[0], o2 - (n2 - l2) % o2))], axis=1)

    N1, N2 = A.shape
    X = []

    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            X.append(A[i1:i1+l1, i2:i2+l2].reshape(l1*l2, 1))

    return np.array(X)[:, :, 0]


def yc_patch_inv(X, n1, n2, l1, l2, o1, o2):
    A = np.zeros((n1, n2))
    mask = np.zeros_like(A)

    idx = 0
    for i1 in range(0, n1 - l1 + 1, o1):
        for i2 in range(0, n2 - l2 + 1, o2):
            A[i1:i1+l1, i2:i2+l2] += X[:, idx].reshape(l1, l2)
            mask[i1:i1+l1, i2:i2+l2] += 1
            idx += 1

    return A / (mask + 1e-8)