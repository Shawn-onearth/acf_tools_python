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
            band_low: float = 0.1,
            band_high: float = 30.0,
            normalize_x: bool = False,
            force_x_positive: bool = False,
            normalize_acf_to_1: bool = True,
            normalize_acf_z: bool = False,
            verbose: int = 0,
            acf_half: bool = True
            ) :

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
        'irasa_ap' : np.ndarray or None
            IRASA估计的1/f幅度谱 (channel, freq)
    """
    
    if not HAS_IRASA and rm_ap:
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
    

    
    # 时间域预处理
    if normalize_x:
        x = stats.zscore(x, axis=-1)
    
    if force_x_positive and np.any(x < 0):
        warnings.warn("强制x为正值")
        x = x - x.min(axis=-1, keepdims=True)
    
    # FFT计算
    nyq = fs / 2
    hN = N // 2 + 1  # 单边谱的长度
    
    # 频率数组
    freq = np.fft.rfftfreq(N, d=1/fs)
    
    # FFT
    X = np.fft.rfft(x) / N  # 归一化FFT


    #对于奇数N，最后一个不是nyquist，也需要乘以二；对于偶数N，最后一个是nyquist，不乘以2
    if N%2 ==1:
        X[:,1:] *= 2
    else :
        X[:,1:-1] *= 2 


    
    # Lag数组 (秒)
    lags = np.arange(hN) / fs
    fulllags = np.arange(N) / fs
    
    # 初始化输出
    ap_linear = None
    irasa_ap = None
    X_norm = None
    x_norm = None
    osc_psd = None
    osc_freqs = None
    fit_params = None
    X_ap_vecs = None
    x_ap_vecs = None

   

    
    # ========== 1/f拟合 (使用YASA的IRASA) ==========
    if rm_ap :
        if verbose:
            print(f"拟合1/f分量 (使用YASA IRASA)")
        
        ap_linear = np.zeros((n_channels, hN))
        irasa_ap = np.zeros((n_channels, hN))
        # nbins calculation removed as it was causing shape mismatch errors
        # osc_psds/osc_freqs removed as they are not returned

        for ch in range(n_channels):
            if verbose:
                print(f"  通道 {ch+1}/{n_channels}")

            # YASA的IRASA接收时域信号
            # winsec默认4秒
            freqs, ap_psd, osc_psd, fit_params = irasa(x[ch], sf=fs, 
                                            band=(band_low, band_high),
                                            hset = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9],
                                            win_sec=4,
                                            return_fit=True,
                                            verbose=False)

            # yasa 0.6.5 irasa returns (1, freqs) even for 1D input
            if ap_psd.ndim == 2:
                ap_psd = ap_psd.flatten()
            
            # 插值到我们的频率分辨率
            # 先把freq限制在band_high之内
            freq_mask = (freq > 0 ) & (freq <= band_high)
            freq_to_fit = freq[freq_mask]
            
            ap_psd_interp = np.interp(freq_to_fit, freqs, ap_psd)
            
            if freqs[0] != 0:
                # 在0 Hz处补0
                ap_psd_interp = np.concatenate(([0], ap_psd_interp))
                freq_to_fit = np.concatenate(([0], freq_to_fit))

            # 补全到hN长度
            if len(ap_psd_interp) < hN:
                n_missing = hN - len(ap_psd_interp)
                ap_psd_interp = np.concatenate((
                    ap_psd_interp,
                    np.zeros(n_missing)
                ))
            
            # 转换为幅度
            #yasa用的是welch，返回的是1/f的PSD (Density, V^2/Hz)
            # 我们需要将其转换为对应全长信号FFT的幅度谱 (X, Volts)
            # 关系: |X|^2 = 2 * PSD * df (where df = fs/N)
            # 所以 |X| = sqrt(2 * fs / N * PSD)
            #注意我们已经在PSD上插值了，因此频率分辨率fs/N就是X的分辨率，就是1/信号总时长
            
            scaling_factor = 2 * fs / N
            ap_linear[ch] = np.sqrt(ap_psd_interp * scaling_factor)
            
            irasa_ap[ch] =  ap_psd_interp
            # osc_psds/osc_freqs assignments removed
            #fit_params[ch] = fit_params
            
            if verbose > 1:
                print(f"    IRASA 1/f提取完成 (freq范围: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz)")
            

    
    # ========== 在频域删除1/f降噪==========
    if rm_ap :
        if verbose:
            print("执行1/f降噪...")
        
        # 保持DC (通常为0)
        ap_for_norm = ap_linear.copy()
        ap_for_norm[:, 0] = 0 

        # 降噪: 从X中减去估计的1/f
        mX_half_spect = np.abs(X)
        mX_half_spect[mX_half_spect == 0] = 1  # 避免除以0
        
        # 频率分量除以模长，等于相位， X/|X|
        X_norm_vecs = X / mX_half_spect
        
        # 1/f向量，用相位乘以1/f的幅度
        ap_half_spect = ap_for_norm
        X_ap_vecs = X_norm_vecs * ap_half_spect
        
        # 初始化降噪后的X
        X_norm = X.copy()
        
        # 掩码: 信号幅度低于噪声幅度的地方
        mask_dont_touch = np.abs(X) - np.abs(X_ap_vecs) < 0
        X_norm[mask_dont_touch] = 0
        
        # 减去1/f分量
        X_norm[~mask_dont_touch] = X[~mask_dont_touch] - X_ap_vecs[~mask_dont_touch]

    else:
        X_norm = X.copy()
        X_ap_vecs = None

    
    # ========== 计算ACF ==========
    if rm_ap:
        if verbose:
            print("计算ACF...")
        
        # Wiener-Khintchin定理: ACF = IFFT(|X|²)
        # (channel, N) -> (channel, N)

        # 1. 减去均值 (Remove DC offset in Frequency Domain)
        # 频域的第0个分量对应直流分量(DC)，置0即等于时域减去均值
        X_norm_no_dc = X_norm.copy()
        X_norm_no_dc[:, 0] = 0
        
        acf_circular = np.fft.irfft(X_norm_no_dc * np.conj(X_norm_no_dc), axis=-1, n=N)

        if acf_half ==True:
            # 截取到hN (只需lag 0到N/2)
            acf = acf_circular[:, :hN]
        else:
            acf = acf_circular

        # 归一化ACF到[-1, 1]
        if normalize_acf_to_1:
            acf = acf / acf[:, 0:1]  # 除以lag=0的值
        
        # zscore归一化
        if normalize_acf_z:
            acf = stats.zscore(acf, axis=-1, keepdims=True)
        
        # 时间域1/f去除信号(可选)

        X_norm = X_norm * N  # 反归一化FFT
        X_ap_vecs = X_ap_vecs * N  # 反归一化FFT
        X = X * N
        #对于奇数N，最后一个不是nyquist，也需要乘以二；对于偶数N，最后一个是nyquist，不乘以2
        if N%2 ==1:
            X_norm[:,1:] /= 2
            X_ap_vecs[:,1:] /= 2
            X[:,1:] /= 2
        else :
            X_norm[:,1:-1] /= 2
            X_ap_vecs[:,1:-1] /= 2
            X[:,1:-1] /= 2

            
        x_norm = np.fft.irfft(X_norm, axis=-1,n=N)
        x_ap_vecs = np.fft.irfft(X_ap_vecs, axis=-1,n=N)

    
    else:
        # 未移除1/f，直接计算ACF
        if verbose:
            print("计算ACF...")
        
        # 1. 减去均值 (Remove DC offset in Frequency Domain)
        X_no_dc = X.copy()
        X_no_dc[:, 0] = 0
        
        acf_circular = np.fft.irfft(X_no_dc * np.conj(X_no_dc), axis=-1, n=N)

        if acf_half ==True:
            # 截取到hN (只需lag 0到N/2)
            acf = acf_circular[:, :hN]
        else:
            acf = acf_circular

        # 归一化ACF到[-1, 1]
        if normalize_acf_to_1:
            acf = acf / acf[:, 0:1]  # 除以lag=0的值
        
        # zscore归一化
        if normalize_acf_z:
            acf = stats.zscore(acf, axis=-1, keepdims=True)
        
        x_norm = None
        X_norm = None
        x_ap_vecs = None
    
    return {
        'acf': acf,
        'lags': lags if acf_half else fulllags,
        'x_norm': x_norm,
        'X_norm': X_norm,
        'x_ap_vecs':x_ap_vecs
    }

