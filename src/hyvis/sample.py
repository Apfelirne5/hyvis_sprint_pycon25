# Add necessary imports to the top of your file where this class will live
import warnings

import numpy as np
from typing import Callable, Optional, Tuple, Union, List
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA # Keep for comparison or alternative methods if needed

# Assuming dr_tools and basic_scans are in the same package or accessible
# Use relative imports if they are in the same package structure:
from .dr_tools import AffineSubspace, numeric_hessian, Hessian
from .basic_scans import landscape_scan_linear, LinearScan

# Or absolute imports if hyvis is installed or in the python path
# from hyvis.dr_tools import AffineSubspace, numeric_hessian, Hessian
# from hyvis.basic_scans import landscape_scan_linear, LinearScan


DEFAULT_HESSIAN_EPSILON = 1e-4 # Epsilon for Hessian calculation
DEFAULT_SQUISH_FACTOR = 0.01 # Factor to reduce variance in non-principal directions


class LossSensitiveExplorer:
    """
    Performs analysis and exploration of a loss landscape guided by sensitivity.

    This class helps identify directions in a parameter space where a given loss
    function changes most significantly, typically by analyzing the Hessian matrix.
    It can then define and scan subspaces spanned by these sensitive directions.
    It also supports adapting a sampling distribution (like a Gaussian) based
    on this sensitivity analysis.

    Attributes:
        loss_func (Callable): The loss function mapping R^n -> R.
        center (np.ndarray): The central point for analysis (e.g., model params).
                             Shape: (1, ambient_dim).
        ambient_dim (int): The dimensionality of the full parameter space.
        hessian (Optional[Hessian]): The computed Hessian of the loss function
                                     at the center point.
        loss_eigenvalues (Optional[np.ndarray]): Eigenvalues of the Hessian,
                                                 sorted by descending absolute value.
        loss_eigenvectors (Optional[np.ndarray]): Eigenvectors of the Hessian,
                                                  ordered corresponding to loss_eigenvalues.
                                                  Stored as columns (ambient_dim x ambient_dim).
        current_covariance (Optional[np.ndarray]): If tracking a distribution,
                                                   its current covariance matrix.
                                                   Shape: (ambient_dim, ambient_dim).
    """

    def __init__(
        self,
        loss_func: Callable[[np.ndarray], float],
        center: np.ndarray,
        initial_covariance: Optional[np.ndarray] = None,
    ):
        """
        Initializes the LossSensitiveExplorer.

        Args:
            loss_func: The loss function to analyze. Takes a 1D or 2D (1xN) numpy
                       array representing a point in the parameter space and
                       returns a float.
            center: The central point in the parameter space for analysis.
                    Can be 1D (ambient_dim,) or 2D (1, ambient_dim).
            initial_covariance: Optional initial covariance matrix for a Gaussian
                               distribution centered at `center`. Used for sampling
                               and adaptation workflows. Shape (ambient_dim, ambient_dim).
                               If None, adaptation workflows are not directly supported
                               unless a covariance is provided later.
        """
        self.loss_func = loss_func
        self.ambient_dim = center.shape[-1] # Get dim from last axis

        # Ensure center is 2D (1 x ambient_dim)
        if center.ndim == 1:
            self.center = center.reshape(1, -1)
        elif center.ndim == 2 and center.shape[0] == 1:
            self.center = center
        else:
             raise ValueError(f"Center has incompatible shape {center.shape}. "
                              f"Expected ({self.ambient_dim},) or (1, {self.ambient_dim}).")

        if initial_covariance is not None:
            if initial_covariance.shape != (self.ambient_dim, self.ambient_dim):
                raise ValueError(f"Initial covariance shape mismatch. Expected "
                                 f"({self.ambient_dim}, {self.ambient_dim}), got "
                                 f"{initial_covariance.shape}")
            # Optional: Check if initial_covariance is positive semi-definite
        self.current_covariance = initial_covariance

        self.hessian: Optional[Hessian] = None
        self.loss_eigenvalues: Optional[np.ndarray] = None
        self.loss_eigenvectors: Optional[np.ndarray] = None # Stored as columns

    def compute_loss_sensitivity(
        self,
        epsilon: float = DEFAULT_HESSIAN_EPSILON,
        recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Hessian of the loss function at the center point and finds
        its eigenvectors and eigenvalues, ordered by sensitivity (absolute eigenvalue).

        Args:
            epsilon: Step size for numerical Hessian calculation.
            recompute: If True, forces recomputation even if already computed.

        Returns:
            Tuple (sorted_eigenvalues, sorted_eigenvectors):
                - sorted_eigenvalues: Eigenvalues sorted by descending absolute value.
                - sorted_eigenvectors: Corresponding eigenvectors as columns.
        """
        if self.hessian is None or recompute:
            print(f"Computing Hessian at center: {self.center.flatten()}...")
            # Define a temporary subspace covering the full ambient space centered correctly
            ambient_subspace = AffineSubspace(
                directions=np.eye(self.ambient_dim),
                center=self.center,
                orthonormalize=False, # Identity is already orthonormal
                skip_checks=True
            )
            self.hessian = numeric_hessian(
                func=self.loss_func,
                subspace=ambient_subspace,
                epsilon=epsilon
            )
            self.hessian.calc_evs() # Calculates and sorts eigenvalues ascending

            # Re-sort by descending absolute eigenvalue magnitude
            abs_eigenvalues = np.abs(self.hessian.eigenvalues)
            sort_indices = np.argsort(abs_eigenvalues)[::-1] # Descending order

            self.loss_eigenvalues = self.hessian.eigenvalues[sort_indices]
            self.loss_eigenvectors = self.hessian.eigenvectors[:, sort_indices]
            print("Hessian computation complete.")

        if self.loss_eigenvalues is None or self.loss_eigenvectors is None:
             raise RuntimeError("Hessian calculation failed unexpectedly.") # Should not happen

        return self.loss_eigenvalues, self.loss_eigenvectors

    def get_principal_subspace(
        self,
        num_dims: int = 2,
        use_sensitivity: bool = True,
    ) -> AffineSubspace:
        """
        Returns an AffineSubspace spanned by the most important directions.

        Args:
            num_dims: The desired dimension of the subspace (e.g., 2 for plotting).
            use_sensitivity: If True (default), uses the top `num_dims` directions
                             based on loss sensitivity (Hessian eigenvectors sorted
                             by abs(eigenvalue)). If False, requires `current_covariance`
                             and performs standard PCA on it to find directions of
                             highest *data* variance.

        Returns:
            An AffineSubspace object.
        """
        if num_dims > self.ambient_dim or num_dims < 1:
            raise ValueError("num_dims must be between 1 and ambient_dim.")

        if use_sensitivity:
            if self.loss_eigenvectors is None:
                print("Loss sensitivity not computed yet. Computing now...")
                self.compute_loss_sensitivity()
                if self.loss_eigenvectors is None: # Check again after computation attempt
                     raise RuntimeError("Failed to compute loss sensitivity.")

            # Select top num_dims eigenvectors (which are stored as columns)
            principal_directions = self.loss_eigenvectors[:, :num_dims].T # Transpose to get (num_dims, ambient_dim)
        else:
            # Use standard PCA based on current_covariance
            if self.current_covariance is None:
                raise ValueError("current_covariance must be set to use use_sensitivity=False.")
            print("Performing standard PCA on current_covariance...")
            pca = PCA(n_components=num_dims)
            # PCA expects data, but we can give it the covariance matrix directly
            # by finding its eigenvectors.
            # Note: PCA on data finds directions explaining variance. Eigenvectors of Cov matrix do the same.
            cov_eigenvalues, cov_eigenvectors = np.linalg.eigh(self.current_covariance)
            # Sort by eigenvalue descending (highest variance first)
            sort_indices = np.argsort(cov_eigenvalues)[::-1]
            principal_directions = cov_eigenvectors[:, sort_indices[:num_dims]].T

        # The subspace is centered at the analysis point
        # The directions from Hessian/PCA are already orthonormal
        return AffineSubspace(
            directions=principal_directions,
            center=self.center,
            orthonormalize=False, # Should already be orthonormal
            skip_checks=True      # Trust numpy's eigh/PCA components
        )

    def adapt_covariance(
        self,
        num_principal_dims: int,
        squish_factor: float = DEFAULT_SQUISH_FACTOR,
        base_covariance: Optional[np.ndarray] = None
    ) -> None:
        """
        Adapts the `current_covariance` based on loss sensitivity (Hessian eigenvectors).

        Keeps variance related to the original covariance along the top
        `num_principal_dims` most loss-sensitive directions and reduces variance
        along the other directions by `squish_factor`.

        Requires `compute_loss_sensitivity` to have been called.

        Args:
            num_principal_dims: The number of principal directions (most loss-sensitive)
                                to preserve variance along.
            squish_factor: Multiplicative factor applied to variance in non-principal
                           directions (e.g., 0.01 reduces variance to 1%).
            base_covariance: The covariance matrix to use as the base for adaptation.
                             If None, uses `self.current_covariance`. Must be provided
                             if `self.current_covariance` is None.
        """
        if self.loss_eigenvectors is None or self.loss_eigenvalues is None:
             raise RuntimeError("Loss sensitivity must be computed before adapting covariance.")

        if base_covariance is None:
            if self.current_covariance is None:
                 raise ValueError("Either initial_covariance must be set or base_covariance provided.")
            base_covariance = self.current_covariance
        elif base_covariance.shape != (self.ambient_dim, self.ambient_dim):
             raise ValueError("Provided base_covariance has incorrect shape.")

        if not (0 <= squish_factor <= 1):
            warnings.warn("squish_factor should ideally be between 0 and 1.")

        V = self.loss_eigenvectors # Basis of loss sensitivity (columns are eigenvectors)

        # Project the base covariance onto the loss sensitivity basis
        # C_H = V.T @ C @ V
        # However, modifying variances *along* these axes is more direct.
        # Variance along direction v_i is v_i.T @ C @ v_i

        new_variance_diag = np.zeros(self.ambient_dim)
        for i in range(self.ambient_dim):
             v_i = V[:, i]
             variance_along_vi = v_i.T @ base_covariance @ v_i
             if i < num_principal_dims:
                 # Keep original variance along this principal direction
                 new_variance_diag[i] = variance_along_vi
             else:
                 # Squish variance along this non-principal direction
                 new_variance_diag[i] = variance_along_vi * squish_factor

        # Ensure non-negative variances (numerical issues might cause small negatives)
        new_variance_diag[new_variance_diag < 0] = 0

        # Construct the new covariance matrix in the standard basis
        # C_new = V @ diag(new_variance_diag) @ V.T
        D_squished = np.diag(new_variance_diag)
        self.current_covariance = V @ D_squished @ V.T

        print(f"Covariance adapted. Kept {num_principal_dims} principal dims, "
              f"squished others by factor {squish_factor}.")

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generates samples from a multivariate Gaussian distribution defined by
        `self.center` and `self.current_covariance`.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            np.ndarray of shape (num_samples, ambient_dim).
        """
        if self.current_covariance is None:
             raise RuntimeError("Covariance matrix is not set. Cannot sample.")

        # Ensure covariance is numerically positive semi-definite for sampling
        # Add small identity jitter if needed (optional, np.random.multivariate_normal handles it somewhat)
        # try:
        #     samples = np.random.multivariate_normal(
        #         self.center.flatten(), self.current_covariance, size=num_samples
        #     )
        # except np.linalg.LinAlgError:
        #     warnings.warn("Covariance matrix possibly not positive semi-definite. Adding jitter.")
        #     jitter = np.eye(self.ambient_dim) * 1e-10
                     # einheitsmatrix 2x2 * 10^-10
        #     samples = np.random.multivariate_normal(
        #         self.center.flatten(), self.current_covariance + jitter, size=num_samples
        #     )

        samples = np.random.multivariate_normal(
            mean=self.center.flatten(), cov=self.current_covariance, size=num_samples
        )# puts all arrays into 1 dim,

        return samples

    def scan_principal_subspace(
        self,
        num_dims: int = 2,
        scope: Union[float, np.ndarray] = 5.0,
        resolution: Union[int, np.ndarray] = 20,
        use_sensitivity: bool = True,
        **scan_kwargs
    ) -> LinearScan:
        """
        Performs a linear scan within the principal subspace.

        Args:
            num_dims: Dimension of the subspace to scan (e.g., 2).
            scope: Scope of the scan along the principal directions.
            resolution: Resolution of the scan along the principal directions.
            use_sensitivity: Passed to get_principal_subspace to determine
                             if subspace is based on loss sensitivity or data PCA.
            **scan_kwargs: Additional keyword arguments passed to
                           `landscape_scan_linear`.

        Returns:
            A LinearScan object containing the results.
        """
        subspace = self.get_principal_subspace(num_dims=num_dims, use_sensitivity=use_sensitivity)

        print(f"Scanning {num_dims}D subspace defined by {'loss sensitivity' if use_sensitivity else 'data PCA'}...")
        scan_result = landscape_scan_linear(
            func=self.loss_func,
            subspace=subspace,
            scope=scope,
            resolution=resolution,
            **scan_kwargs
        )
        print("Scan complete.")
        return scan_result

    def run_gaussian_workflow(
        self,
        num_dims_scan: int = 2,
        num_dims_adapt: Optional[int] = None, # Dims to keep variance for
        squish_factor: float = DEFAULT_SQUISH_FACTOR,
        hessian_epsilon: float = DEFAULT_HESSIAN_EPSILON,
        scan_scope: Union[float, np.ndarray] = 5.0,
        scan_resolution: Union[int, np.ndarray] = 20,
        **scan_kwargs
    ) -> LinearScan:
        """
        Runs the example workflow for a Gaussian distribution:
        1. Compute loss sensitivity (Hessian) at the center.
        2. Optionally adapt the covariance matrix based on sensitivity.
        3. Identify the principal subspace (based on loss sensitivity).
        4. Scan the principal subspace.

        Args:
            num_dims_scan: Dimension of the final subspace to scan (e.g., 2).
            num_dims_adapt: Number of principal directions to *keep* variance for
                            during adaptation. If None, adaptation is skipped.
                            Must be >= num_dims_scan if adaptation is performed.
            squish_factor: Factor to reduce variance in non-principal directions
                           during adaptation.
            hessian_epsilon: Epsilon for Hessian calculation.
            scan_scope: Scope for the final linear scan.
            scan_resolution: Resolution for the final linear scan.
            **scan_kwargs: Additional arguments for landscape_scan_linear.

        Returns:
            LinearScan object of the principal subspace.
        """
        # 1. Compute Loss Sensitivity
        self.compute_loss_sensitivity(epsilon=hessian_epsilon)

        # 2. Adapt Covariance (Optional)
        if num_dims_adapt is not None:
            if self.current_covariance is None:
                 warnings.warn("Adaptation requested but no initial_covariance provided. Skipping.")
            elif num_dims_adapt < num_dims_scan:
                 raise ValueError("num_dims_adapt must be >= num_dims_scan if adapting.")
            else:
                 self.adapt_covariance(num_principal_dims=num_dims_adapt,
                                       squish_factor=squish_factor)

        # 3 & 4. Get Principal Subspace and Scan
        scan_result = self.scan_principal_subspace(
            num_dims=num_dims_scan,
            scope=scan_scope,
            resolution=scan_resolution,
            use_sensitivity=True, # Use loss sensitivity for the scan subspace
            **scan_kwargs
        )
        return scan_result

    # --- Methods for Randomized Search Workflow would go here ---
    # This is more involved, likely requiring definitions for parameter search spaces,
    # sampling strategies for distributions, and evaluation metrics.
    # Example sketch:
    # def run_randomized_search_workflow(self, search_space_def, num_candidates, ...):
    #     best_params = self.find_best_distribution_params(search_space_def, num_candidates, ...)
    #     self.center = best_params['mean']
    #     self.current_covariance = best_params['covariance']
    #     # Now run the Gaussian workflow from this new state
    #     return self.run_gaussian_workflow(...)