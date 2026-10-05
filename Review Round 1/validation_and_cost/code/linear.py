"""Column-scaled QR and reorthogonalized block growth; no normal-equation inverse."""
import numpy as np
from scipy.linalg import qr, solve_triangular, svdvals


class LinearSpace:
    def __init__(self, matrix, rhs, rank_tol=1e-10):
        self.rank_tol = rank_tol
        self.scales = np.linalg.norm(matrix, axis=0)
        if np.any(self.scales == 0):
            raise ValueError('Zero column in initial space.')
        scaled = matrix / self.scales
        self.Q, self.R = qr(scaled, mode='economic', check_finite=False)
        singular = svdvals(self.R, check_finite=False)
        if singular[-1] <= rank_tol * singular[0]:
            raise ValueError('Initial or refined space is numerically rank deficient.')
        self.rhs = rhs
        self.residual = rhs - self.Q @ (self.Q.T @ rhs)
        self.initial_condition = float(singular[0] / singular[-1])

    def proposal(self, block):
        scales = np.linalg.norm(block, axis=0)
        if np.any(scales < 1e-15):
            return None
        scaled = block / scales
        cross = self.Q.T @ scaled
        z = scaled - self.Q @ cross
        # A second projection prevents loss of orthogonality for correlated blocks.
        correction = self.Q.T @ z
        cross += correction
        z -= self.Q @ correction
        q, r = qr(z, mode='economic', check_finite=False)
        singular = svdvals(r, check_finite=False)
        if singular[-1] < self.rank_tol:
            return None
        gain = float(np.sum((q.T @ self.residual) ** 2))
        return dict(Q=q, R=r, cross=cross, scales=scales, gain=gain,
                    smallest_independent_singular=float(singular[-1]))

    def append(self, proposal):
        n, k = self.R.shape[0], proposal['R'].shape[0]
        expanded = np.zeros((n + k, n + k))
        expanded[:n, :n] = self.R
        expanded[:n, n:] = proposal['cross']
        expanded[n:, n:] = proposal['R']
        self.R = expanded
        self.Q = np.column_stack([self.Q, proposal['Q']])
        self.scales = np.concatenate([self.scales, proposal['scales']])
        self.residual -= proposal['Q'] @ (proposal['Q'].T @ self.residual)

    def refresh(self, proposal, added_q):
        """Update a cached candidate using only the newly accepted block.

        The old candidate was already orthogonal to the old selected space.
        Retain the exact same full-pool greedy criterion after each acceptance.
        """
        z = proposal['Q'] @ proposal['R']
        cross = added_q.T @ z
        z -= added_q @ cross
        correction = added_q.T @ z
        cross += correction
        z -= added_q @ correction
        q, r = qr(z, mode='economic', check_finite=False)
        singular = svdvals(r, check_finite=False)
        if singular[-1] < self.rank_tol:
            return None
        return dict(Q=q, R=r, cross=np.row_stack([proposal['cross'], cross]),
                    scales=proposal['scales'], gain=float(np.sum((q.T @ self.residual) ** 2)),
                    smallest_independent_singular=float(singular[-1]))

    def coefficients(self):
        scaled = solve_triangular(self.R, self.Q.T @ self.rhs, check_finite=False)
        return scaled / self.scales[:, None]

    def orthogonality_error(self):
        return float(np.max(np.abs(self.Q.T @ self.Q - np.eye(self.Q.shape[1]))))
