use IRASA to perform acf, AND return a 1/f removed tie-domain signal based on the aperiodic elements returned by IRASA. 
Need to have YASA installed.

Usage:
get_acf(x: np.ndarray, 
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
