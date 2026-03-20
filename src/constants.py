"""Bridge Classification — shared constants and lightweight utilities.

This module has NO heavy dependencies (no torch, spconv, pdal, numpy) so it
can be safely imported anywhere, including environments without a GPU.
"""

# --- Model class schema ---
NUM_CLASSES = 4

CLASS_NAMES = {
    0: "Background",
    1: "Ground/Water",
    2: "Bridge Deck",
    3: "Obstacles",
}

CLASS_COLORS = {
    0: "black",
    1: "orange",
    2: "blue",
    3: "yellow",
}

# --- ASPRS <-> Model class codes ---
BRIDGE_DECK_MODEL_CLASS = 2
BRIDGE_DECK_ASPRS_CODE = 17
OBSTACLES_MODEL_CLASS = 3
OBSTACLES_ASPRS_CODE = 18

# Model class -> ASPRS LAS code (used at inference time)
MODEL_TO_LAS_MAP = {
    0: 1,                                                    # Background  -> Unclassified
    1: 2,                                                    # Ground/Water -> Ground
    BRIDGE_DECK_MODEL_CLASS: BRIDGE_DECK_ASPRS_CODE,         # Bridge Deck -> Bridge Deck
    OBSTACLES_MODEL_CLASS: OBSTACLES_ASPRS_CODE,             # Obstacles   -> High Noise
}

# ASPRS LAS code -> Model class (used at preprocessing/training time)
# Maps standard ASPRS/LAS codes to Model Training Labels (0-3)
# Note: multiple LAS codes map to a single model class (2,9 -> 1)
# 0: Background/Unclassified (included merged piers/pylons)
# 1: Ground/ Water (Non-Bridge Surface)
# 2: Bridge Deck (Primary Target)
# 3: Obstacles (Cars, Poles, High Noise)
LAS_TO_MODEL_MAP = {
    2: 1,    # Ground -> 1
    9: 1,    # Water -> 1
    17: 2,   # Bridge Deck -> 2
    18: 3,   # High Noise -> 3
}

# --- Inference defaults ---
VOXEL_SIZE = 0.1
# Extra voxel buffer added to spconv spatial shape to avoid
# out-of-bounds during strided/transposed conv operations
SPATIAL_SHAPE_PADDING = 10
# Files with fewer points are skipped
MIN_POINT_COUNT = 100
BRIDGE_TIMEOUT = 150

# --- AWS ---
AWS_MAX_RETRIES = 3


# --- Timeout machinery (lightweight — no spconv/torch dependency) ---
class BridgeTimeout(BaseException):
    """Raised when a single bridge exceeds the per-bridge wall-clock timeout."""
    pass


def _timeout_handler(signum, frame):
    raise BridgeTimeout()
