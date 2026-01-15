"""
自相关函数(ACF)计算与1/f降噪

支持输入格式:
- (1, time): 单通道时序
- (Channel, time): 多通道脑电信号

核心功能:
1. FFT计算与幅度谱获取
2. 1/f拟合(使用YASA库的IRASA方法)
3. 1/f降噪
4. ACF计算与归一化
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict
import warnings

try:
    from yasa import irasa
    HAS_IRASA = True
except ImportError:
    HAS_IRASA = False
    warnings.warn(
        "未安装yasa库。请运行: pip install yasa"
    )


def get_acf(x: np.ndarray, 
            fs: float,
            rm_ap: bool = False,
            fit_ap: bool = False,
            irasa_h: Tuple[int, int] = (1, 32),
            ap_fit_flims: Optional[Tuple[float, float]] = None,
            normalize_x: bool = False,
            force_x_positive: bool = False,
            normalize_acf_to_1: bool = True,
            normalize_acf_z: bool = False,
            verbose: int = 0,
            get_x_norm: bool = False) -> Dict:
    """
    计算自相关函数(ACF)，可选地移除1/f频谱分量
    
    Parameters
    ----------
    x : np.ndarray
        输入时间序列，shape为(1, time)或(channel, time)
        其中time为最后一维
    fs : float
        采样率(Hz)
    rm_ap : bool, default=False
        是否拟合并移除1/f分量
    fit_ap : bool, default=False
        是否拟合1/f(即使不移除)
    irasa_h : tuple, default=(1, 32)
        IRASA: resampling因子范围 (min_h, max_h)
    ap_fit_flims : tuple, optional
        1/f拟合的频率范围 [fmin, fmax]，默认[1, fs/2]
    normalize_x : bool, default=False
        是否标准化时间域输入到zscore
    force_x_positive : bool, default=False
        是否强制x为正值
    normalize_acf_to_1 : bool, default=True
        是否将ACF归一化到[-1, 1]
    normalize_acf_z : bool, default=False
        是否将ACF进行zscore归一化
    verbose : int, default=0
        输出详细程度 {0, 1, 2}
    get_x_norm : bool, default=False
        是否返回1/f去除后的时间域信号
    
    Returns
    -------
    dict with keys:
        'acf' : np.ndarray, shape=(channel, lags)
            自相关函数
        'lags' : np.ndarray
            lag时间(秒)
        'freq' : np.ndarray
            频率数组(Hz)
        'mX' : np.ndarray, shape=(channel, freq)
            幅度谱
        'ap_linear' : np.ndarray, shape=(channel, freq) or None
            1/f分量(线性幅度)
        'x_norm' : np.ndarray, shape=(channel, time) or None
            1/f去除后的时间序列
        'X_norm' : np.ndarray, shape=(channel, N) or None
            1/f去除后的完整复数频谱
        'irasa_ap' : np.ndarray or None
            IRASA估计的1/f幅度谱 (channel, freq)
    """
    
    if not HAS_IRASA and (fit_ap or rm_ap):
        raise ImportError(
            "需要yasa库进行IRASA 1/f拟合。请安装: pip install yasa"
        )
    
    # 参数验证与初始化
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]  # 转为 (1, time)
    
    if x.ndim != 2:
        raise ValueError(f"输入x应为2D数组(channel, time)，得到{x.ndim}D")
    
    n_channels, n_time = x.shape
    N = n_time
    
    if ap_fit_flims is None:
        ap_fit_flims = (1.0, fs / 2)
    
    # 确保band不超过Nyquist频率
    band_high = min(ap_fit_flims[1], fs / 2 - 0.1)
    band_low = max(ap_fit_flims[0], 0.5)
    
    # 时间域预处理
    if normalize_x:
        x = stats.zscore(x, axis=-1, keepdims=True)
    
    if force_x_positive and np.any(x < 0):
        warnings.warn("强制x为正值")
        x = x - x.min(axis=-1, keepdims=True)
    
    # FFT计算
    nyq = fs / 2
    hN = N // 2 + 1  # 单边谱的长度
    
    # 频率数组
    freq = np.arange(hN) / N * fs
    
    # FFT: (channel, N) -> (channel, N)
    X = np.fft.fft(x, axis=-1) / N * 2
    
    # 幅度谱 (channel, hN)
    mX = np.abs(X[:, :hN])
    
    # Lag数组 (秒)
    lags = np.arange(hN) / fs
    
    # 初始化输出
    ap_linear = None
    irasa_ap = None
    X_norm = None
    x_norm = None
    
    # ========== 1/f拟合 (使用YASA的IRASA) ==========
    if fit_ap:
        if verbose:
            print(f"拟合1/f分量 (使用YASA IRASA)")
        
        ap_linear = np.zeros((n_channels, hN))
        irasa_ap = np.zeros((n_channels, hN))
        
        for ch in range(n_channels):
            if verbose:
                print(f"  通道 {ch+1}/{n_channels}")
            
            try:
                # YASA的IRASA接收时域信号
                # 返回 (freq_array, ap_psd, osc_psd) 当 return_fit=False
                h_range = list(np.arange(irasa_h[0], irasa_h[1] + 0.01, 0.05))
                # 限制band上限，避免超过Nyquist频率的警告
                band_high_safe = min(band_high, fs / 2 - 1)
                freqs, ap_psd, osc_psd = irasa(x[ch], sf=fs, 
                                               band=(band_low, band_high_safe),
                                               hset=h_range, 
                                               return_fit=False,
                                               verbose=False)
                
                # 插值到我们的频率分辨率
                ap_psd_interp = np.interp(freq, freqs, ap_psd, 
                                          left=ap_psd[0], right=ap_psd[-1])
                
                # 转换为幅度
                ap_linear[ch] = np.sqrt(np.abs(ap_psd_interp))
                irasa_ap[ch] = np.sqrt(np.abs(ap_psd_interp))
                
                if verbose > 1:
                    print(f"    IRASA 1/f提取完成 (freq范围: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz)")
                
            except Exception as e:
                warnings.warn(f"通道 {ch} IRASA拟合失败: {e}")
                ap_linear[ch] = np.ones(hN)  # 回退到平坦谱
                irasa_ap[ch] = np.ones(hN)
    
    # ========== 1/f降噪 ==========
    if rm_ap and ap_linear is not None:
        if verbose:
            print("执行1/f降噪...")
        
        # 处理DC分量(频率0 Hz处的无穷大)
        ap_for_norm = ap_linear.copy()
        ap_for_norm[:, 0] = 1  # 保持DC不变
        
        # 镜像1/f分量以获得完整谱(包含负频率)
        if N % 2 == 0:
            # 偶数长度: [0, 1,..., N/2, ..., N-1]
            ap_whole_spect = np.concatenate([
                ap_for_norm,
                ap_for_norm[:, -2:0:-1]  # 反向，从N/2-1到1
            ], axis=-1)
        else:
            # 奇数长度: [0, 1,..., (N-1)/2, ..., N-1]
            ap_whole_spect = np.concatenate([
                ap_for_norm,
                ap_for_norm[:, -1:0:-1]  # 反向，从hN-1到1
            ], axis=-1)
        
        # 降噪: 从X中减去估计的1/f
        mX_whole_spect = np.abs(X)
        mX_whole_spect[mX_whole_spect == 0] = 1  # 避免除以0
        
        # X的单位向量，因为除以了
        X_norm_vecs = X / mX_whole_spect
        
        # 1/f向量
        X_ap_vecs = X_norm_vecs * ap_whole_spect
        
        # 初始化降噪后的X
        X_norm = X.copy()
        
        # 掩码: 信号幅度低于噪声幅度的地方
        mask_dont_touch = np.abs(X) - np.abs(X_ap_vecs) < 0
        X_norm[mask_dont_touch] = 0
        
        # 减去1/f分量
        X_norm[~mask_dont_touch] = X[~mask_dont_touch] - X_ap_vecs[~mask_dont_touch]
    else:
        X_norm = X.copy()
    
    # ========== 计算ACF ==========
    if verbose:
        print("计算ACF...")
    
    # Wiener-Khintchin定理: ACF = IFFT(|X|²)
    # (channel, N) -> (channel, N)
    acf_full = np.fft.ifft(X_norm * np.conj(X_norm), axis=-1).real
    
    # 截取到hN (只需lag 0到N/2)
    acf = acf_full[:, :hN]
    
    # 归一化ACF到[-1, 1]
    if normalize_acf_to_1:
        acf = acf / acf[:, 0:1]  # 除以lag=0的值
    
    # zscore归一化
    if normalize_acf_z:
        acf = stats.zscore(acf, axis=-1, keepdims=True)
    
    # 时间域1/f去除信号(可选)
    if get_x_norm and X_norm is not None:
        x_norm = np.fft.ifft(X_norm, axis=-1).real
    
    return {
        'acf': acf,
        'lags': lags,
        'freq': freq,
        'mX': mX,
        'ap_linear': ap_linear,
        'x_norm': x_norm,
        'X_norm': X_norm,
        'irasa_ap': irasa_ap,
    }


if __name__ == '__main__':
    # 简单测试
    np.random.seed(42)
    
    # 生成测试信号: 1/f + 振荡
    fs = 100
    N = 1000
    t = np.arange(N) / fs
    
    # 1/f噪声 (pink noise)
    f = np.fft.fftfreq(N, 1/fs)[:N//2+1]
    psd = 1 / np.maximum(np.abs(f), 0.1) ** 1.5
    pink_noise_fft = np.sqrt(psd) * (np.random.randn(N//2+1) + 1j * np.random.randn(N//2+1))
    pink_noise = np.fft.irfft(pink_noise_fft, n=N)
    
    # 加入10 Hz的响应
    response = 0.5 * np.sin(2 * np.pi * 10 * t)
    x = pink_noise + response
    
    # 测试单通道
    print("=== 测试单通道 (1, 1000) ===")
    result = get_acf(x[np.newaxis, :], fs, rm_ap=True, verbose=1)
    print(f"ACF shape: {result['acf'].shape}")
    print(f"ACF[0] lag0值: {result['acf'][0, 0]:.4f}")
    
    if result['irasa_ap'] is not None:
        print(f"IRASA提取的1/f: min={result['irasa_ap'][0].min():.3f}, "
              f"max={result['irasa_ap'][0].max():.3f}")
    
    # 测试多通道
    print("\n=== 测试多通道 (2, 1000) ===")
    x_multi = np.vstack([x, x * 0.8])
    result = get_acf(x_multi, fs, rm_ap=True, verbose=1)
    print(f"ACF shape: {result['acf'].shape}")
    print(f"ACF[0, 0] (lag=0): {result['acf'][0, 0]:.4f}")
    print(f"ACF[1, 0] (lag=0): {result['acf'][1, 0]:.4f}")
