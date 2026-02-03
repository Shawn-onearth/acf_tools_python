### `get_acf`

Use IRASA to perform ACF on EEG data, and optionally return a 1/f–removed time‑domain signal based on the aperiodic elements returned by IRASA.

Requires: **YASA**

#### **Function Signature**
```python
get_acf(
    x: np.ndarray,
    fs: float,
    rm_ap: bool = True,
    band_low: float = 0.1,
    band_high: float = 30.0,
    normalize_x: bool = False,
    force_x_positive: bool = False,
    normalize_acf_to_1: bool = True,
    normalize_acf_z: bool = False,
    verbose: int = 0,
    acf_half: bool = True
)

input shape is (1, time) or (channel, time).
