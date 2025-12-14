"""Configuration settings for the project.

This module centralizes all paths and default parameters used throughout
the codebase, making it easy to modify settings in one place.
"""

from pathlib import Path


# =============================================================================
# Directory Paths
# =============================================================================

# Project root directory (parent of src/)
ROOT_DIR = Path(__file__).parent.parent

# Data directory (source data)
DATA_DIR = ROOT_DIR / "data"

# Cache directory (computed results)
CACHE_DIR = ROOT_DIR / "cache"

# Plots output directory
PLOTS_DIR = ROOT_DIR / "plots"

# Documentation directory
DOCS_DIR = ROOT_DIR / "docs"


# =============================================================================
# Data Files
# =============================================================================

# Welch-Goyal predictor data
PREDICTOR_DATA_FILE = DATA_DIR / "PredictorData2021 - Monthly.csv"

# NBER recession dates
NBER_DATA_FILE = DATA_DIR / "NBER_20210719_cycle_dates_pasted.csv"

# Cached simulation results
METRICS_CACHE_FILE = CACHE_DIR / "metrics.parquet"


# =============================================================================
# Model Default Parameters
# =============================================================================

# Backtest defaults
DEFAULT_TRAIN_WINDOW = 12  # Training window size (T) in months
DEFAULT_RIDGE_LAMBDA = 1000  # Ridge regularization parameter (z)
ANNUALIZATION_FACTOR = 12  # Monthly to annual conversion

# Random Fourier Features defaults
DEFAULT_RFF_GAMMA = 2.0  # Bandwidth parameter
DEFAULT_RFF_N_FEATURES = 6000  # Number of features (P = 2 * n_features)

# Standardization
STANDARDIZATION_BURN_IN = 36  # Months required for initial standardization
VOLATILITY_WINDOW = 12  # Rolling window for return volatility


# =============================================================================
# Simulation Parameters
# =============================================================================

# Default gamma values to test
GAMMA_VALUES = [0.25, 0.5, 1, 2, 4]

# Default ridge lambda values (log-spaced)
Z_VALUES_RANGE = (-3, 6, 10)  # (start, stop, num) for np.logspace

# Number of iterations for RFF variance reduction
DEFAULT_ITERATIONS = 500


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_dirs_exist() -> None:
    """Create output directories if they don't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
