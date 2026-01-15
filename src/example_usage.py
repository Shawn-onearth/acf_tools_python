"""
get_acf.py 使用示例

展示如何使用ACF计算函数处理单通道和多通道脑电信号
"""

import numpy as np
import matplotlib.pyplot as plt
from get_acf import get_acf

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def example_single_channel():
    """单通道脑电信号示例"""
    print("=" * 60)
    print("示例 1: 单通道时序 (1, time)")
    print("=" * 60)
    
    # 生成模拟脑电信号
    fs = 250  # 采样率 250 Hz (典型脑电采样率)
    duration = 4  # 4秒
    N = int(fs * duration)
    t = np.arange(N) / fs
    
    # 创建信号: 1/f噪声 + 10 Hz alpha波段响应
    np.random.seed(123)
    
    # 1/f背景噪声
    f = np.fft.fftfreq(N, 1/fs)[:N//2+1]
    psd = 1 / np.maximum(np.abs(f), 0.5) ** 1.2
    pink_fft = np.sqrt(psd) * (np.random.randn(N//2+1) + 1j * np.random.randn(N//2+1))
    background = np.fft.irfft(pink_fft, n=N)
    
    # 10 Hz alpha响应
    alpha = 2.0 * np.sin(2 * np.pi * 10 * t)
    
    # 合成信号
    signal = background + alpha
    signal = signal[np.newaxis, :]  # (1, N)
    
    # 计算ACF (不去除1/f)
    print("\n1. 计算原始信号的ACF...")
    result_raw = get_acf(signal, fs, rm_ap=False, normalize_acf_to_1=True)
    
    # 计算ACF (去除1/f, FOOOF方法)
    print("\n2. 计算去除1/f后的ACF (FOOOF官方库)...")
    result_denoised = get_acf(
        signal, fs, 
        rm_ap=True, 
        response_f0=10.0,
        fit_knee=False,
        ap_fit_flims=(1.0, fs/2),
        only_use_f0_harmonics=True,
        normalize_acf_to_1=True,
        verbose=0
    )
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 时间域信号
    axes[0, 0].plot(t[:500], signal[0, :500], 'k-', linewidth=0.8)
    axes[0, 0].set_xlabel('时间 (秒)')
    axes[0, 0].set_ylabel('幅度')
    axes[0, 0].set_title('时间域信号 (前2秒)')
    axes[0, 0].grid(alpha=0.3)
    
    # 频域幅度谱
    axes[0, 1].plot(result_raw['freq'], result_raw['mX'][0], 'b-', 
                    linewidth=1, label='原始谱', alpha=0.7)
    if result_denoised['ap_linear'] is not None:
        axes[0, 1].plot(result_denoised['freq'], 
                       result_denoised['ap_linear'][0], 
                       'r--', linewidth=2, label='FOOOF 1/f估计')
    axes[0, 1].set_xlabel('频率 (Hz)')
    axes[0, 1].set_ylabel('幅度')
    axes[0, 1].set_title('幅度谱与1/f拟合')
    axes[0, 1].set_xlim([0, 50])
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # ACF对比
    lag_max = 1.0  # 显示到1秒
    lag_mask = result_raw['lags'] <= lag_max
    
    axes[1, 0].plot(result_raw['lags'][lag_mask], 
                   result_raw['acf'][0, lag_mask], 
                   'k-', linewidth=1.5, label='原始ACF', alpha=0.7)
    axes[1, 0].plot(result_denoised['lags'][lag_mask], 
                   result_denoised['acf'][0, lag_mask], 
                   'r-', linewidth=1.5, label='去1/f (FOOOF)')
    axes[1, 0].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('Lag (秒)')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].set_title('自相关函数对比')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # ACF细节 (放大10 Hz周期)
    lag_detail = (result_raw['lags'] >= 0) & (result_raw['lags'] <= 0.3)
    axes[1, 1].plot(result_raw['lags'][lag_detail], 
                   result_raw['acf'][0, lag_detail], 
                   'k-', linewidth=1.5, label='原始ACF', alpha=0.7)
    axes[1, 1].plot(result_denoised['lags'][lag_detail], 
                   result_denoised['acf'][0, lag_detail], 
                   'r-', linewidth=1.5, label='去1/f (FOOOF)')
    
    # 标记10 Hz周期 (0.1秒)
    for i in range(1, 4):
        axes[1, 1].axvline(i * 0.1, color='blue', linestyle='--', 
                          alpha=0.3, linewidth=1)
    axes[1, 1].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Lag (秒)')
    axes[1, 1].set_ylabel('ACF')
    axes[1, 1].set_title('ACF细节 (0-0.3秒, 蓝线=10Hz周期)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_single_channel.png', dpi=150)
    print("\n✓ 图像已保存: example_single_channel.png")
    plt.close()
    
    return result_raw, result_denoised


def example_multi_channel():
    """多通道脑电信号示例"""
    print("\n" + "=" * 60)
    print("示例 2: 多通道脑电 (Channel, time)")
    print("=" * 60)
    
    # 生成3通道模拟脑电
    fs = 250
    duration = 4
    N = int(fs * duration)
    t = np.arange(N) / fs
    n_channels = 3
    
    np.random.seed(456)
    
    signals = []
    channel_names = ['Ch1: 10Hz强', 'Ch2: 10Hz中', 'Ch3: 10Hz弱']
    alphas = [3.0, 1.5, 0.5]  # 不同通道的alpha强度
    
    for ch in range(n_channels):
        # 1/f背景
        f = np.fft.fftfreq(N, 1/fs)[:N//2+1]
        psd = 1 / np.maximum(np.abs(f), 0.5) ** (1.0 + ch * 0.2)
        pink_fft = np.sqrt(psd) * (np.random.randn(N//2+1) + 1j * np.random.randn(N//2+1))
        background = np.fft.irfft(pink_fft, n=N)
        
        # 10 Hz alpha (不同强度)
        alpha = alphas[ch] * np.sin(2 * np.pi * 10 * t + ch * np.pi / 4)
        
        signals.append(background + alpha)
    
    signals = np.array(signals)  # (3, N)
    
    # 批量处理所有通道
    print("\n处理3个通道...")
    result = get_acf(
        signals, fs,
        rm_ap=True,
        response_f0=10.0,
        only_use_f0_harmonics=True,
        normalize_acf_to_1=True,
        verbose=1
    )
    
    # 可视化多通道结果
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    colors = ['red', 'green', 'blue']
    
    for ch in range(n_channels):
        # 幅度谱
        axes[ch, 0].plot(result['freq'], result['mX'][ch], 
                        color=colors[ch], linewidth=1, alpha=0.7,
                        label=f'{channel_names[ch]}')
        if result['ap_linear'] is not None:
            axes[ch, 0].plot(result['freq'], result['ap_linear'][ch],
                           'k--', linewidth=2, alpha=0.5, label='1/f估计')
        axes[ch, 0].set_xlim([0, 50])
        axes[ch, 0].set_ylabel('幅度')
        axes[ch, 0].set_title(f'{channel_names[ch]} - 幅度谱')
        axes[ch, 0].legend()
        axes[ch, 0].grid(alpha=0.3)
        
        # ACF
        lag_max = 0.5
        lag_mask = result['lags'] <= lag_max
        axes[ch, 1].plot(result['lags'][lag_mask], 
                        result['acf'][ch, lag_mask],
                        color=colors[ch], linewidth=2)
        axes[ch, 1].axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        # 标记10 Hz周期
        for i in range(1, 6):
            axes[ch, 1].axvline(i * 0.1, color='gray', linestyle='--', 
                               alpha=0.2, linewidth=1)
        
        axes[ch, 1].set_ylabel('ACF')
        axes[ch, 1].set_title(f'{channel_names[ch]} - 自相关')
        axes[ch, 1].grid(alpha=0.3)
        
        if ch == n_channels - 1:
            axes[ch, 0].set_xlabel('频率 (Hz)')
            axes[ch, 1].set_xlabel('Lag (秒)')
    
    plt.tight_layout()
    plt.savefig('example_multi_channel.png', dpi=150)
    print("\n✓ 图像已保存: example_multi_channel.png")
    plt.close()
    
    return result


def example_comparison_methods():
    """展示FOOOF的knee参数效果"""
    print("\n" + "=" * 60)
    print("示例 3: FOOOF knee参数对比")
    print("=" * 60)
    
    # 生成测试信号
    fs = 200
    N = 2000
    t = np.arange(N) / fs
    
    np.random.seed(789)
    
    # 1/f噪声
    f = np.fft.fftfreq(N, 1/fs)[:N//2+1]
    psd = 1 / np.maximum(np.abs(f), 0.3) ** 1.5
    pink_fft = np.sqrt(psd) * (np.random.randn(N//2+1) + 1j * np.random.randn(N//2+1))
    background = np.fft.irfft(pink_fft, n=N)
    
    # 12 Hz响应
    response = 1.5 * np.sin(2 * np.pi * 12 * t)
    signal = (background + response)[np.newaxis, :]
    
    # 两种FOOOF模式
    print("\n使用fixed模式(无knee)...")
    result_fixed = get_acf(
        signal, fs,
        rm_ap=True,
        fit_knee=False,
        response_f0=12.0,
        only_use_f0_harmonics=True,
        normalize_acf_to_1=True,
        verbose=0
    )
    
    print("\n使用knee模式...")
    result_knee = get_acf(
        signal, fs,
        rm_ap=True,
        fit_knee=True,
        response_f0=12.0,
        only_use_f0_harmonics=True,
        normalize_acf_to_1=True,
        verbose=0
    )
    
    # 对比可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原始幅度谱
    result_no_rm = get_acf(signal, fs, rm_ap=False)
    axes[0, 0].semilogy(result_no_rm['freq'], result_no_rm['mX'][0], 
                       'k-', linewidth=1, alpha=0.5, label='原始谱')
    
    if result_fixed['ap_linear'] is not None:
        axes[0, 0].semilogy(result_fixed['freq'], 
                           result_fixed['ap_linear'][0],
                           color='blue', linestyle='--',
                           linewidth=2, label='fixed模式')
    
    if result_knee['ap_linear'] is not None:
        axes[0, 0].semilogy(result_knee['freq'], 
                           result_knee['ap_linear'][0],
                           color='red', linestyle='--',
                           linewidth=2, label='knee模式')
    
    axes[0, 0].set_xlim([1, 80])
    axes[0, 0].set_xlabel('频率 (Hz)')
    axes[0, 0].set_ylabel('幅度 (log scale)')
    axes[0, 0].set_title('1/f拟合对比 (对数刻度)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, which='both')
    
    # 放大12 Hz附近
    axes[0, 1].plot(result_no_rm['freq'], result_no_rm['mX'][0], 
                   'k-', linewidth=1.5, alpha=0.7, label='原始谱')
    
    if result_fixed['ap_linear'] is not None:
        axes[0, 1].plot(result_fixed['freq'], 
                       result_fixed['ap_linear'][0],
                       color='blue', linestyle='--',
                       linewidth=2, label='fixed 1/f')
    
    if result_knee['ap_linear'] is not None:
        axes[0, 1].plot(result_knee['freq'], 
                       result_knee['ap_linear'][0],
                       color='red', linestyle='--',
                       linewidth=2, label='knee 1/f')
    
    axes[0, 1].axvline(12, color='purple', linestyle=':', alpha=0.5, linewidth=2)
    axes[0, 1].set_xlim([8, 16])
    axes[0, 1].set_xlabel('频率 (Hz)')
    axes[0, 1].set_ylabel('幅度')
    axes[0, 1].set_title('12 Hz响应放大')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # ACF对比 (全范围)
    lag_mask = result_no_rm['lags'] <= 0.5
    axes[1, 0].plot(result_no_rm['lags'][lag_mask], 
                   result_no_rm['acf'][0, lag_mask],
                   'k-', linewidth=1.5, alpha=0.5, label='原始ACF')
    
    axes[1, 0].plot(result_fixed['lags'][lag_mask],
                   result_fixed['acf'][0, lag_mask],
                   color='blue', linewidth=2,
                   label='fixed模式')
    
    axes[1, 0].plot(result_knee['lags'][lag_mask],
                   result_knee['acf'][0, lag_mask],
                   color='red', linewidth=2,
                   label='knee模式')
    
    axes[1, 0].axhline(0, color='gray', linestyle=':', alpha=0.3)
    axes[1, 0].set_xlabel('Lag (秒)')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].set_title('ACF对比')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # ACF细节 (放大12 Hz周期)
    lag_detail = (result_no_rm['lags'] >= 0) & (result_no_rm['lags'] <= 0.25)
    
    axes[1, 1].plot(result_fixed['lags'][lag_detail],
                   result_fixed['acf'][0, lag_detail],
                   color='blue', linewidth=2,
                   label='fixed模式', marker='o', markersize=3)
    
    axes[1, 1].plot(result_knee['lags'][lag_detail],
                   result_knee['acf'][0, lag_detail],
                   color='red', linewidth=2,
                   label='knee模式', marker='s', markersize=3)
    
    # 标记12 Hz周期 (1/12 ≈ 0.083秒)
    for i in range(1, 4):
        axes[1, 1].axvline(i / 12, color='purple', linestyle='--', 
                          alpha=0.3, linewidth=1)
    
    axes[1, 1].axhline(0, color='gray', linestyle=':', alpha=0.3)
    axes[1, 1].set_xlabel('Lag (秒)')
    axes[1, 1].set_ylabel('ACF')
    axes[1, 1].set_title('ACF细节 (紫线=12Hz周期)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_fooof_comparison.png', dpi=150)
    print("\n✓ 图像已保存: example_fooof_comparison.png")
    plt.close()
    
    # 打印FOOOF参数
    if result_fixed['fooof_results'] and result_fixed['fooof_results'][0]:
        fm_fixed = result_fixed['fooof_results'][0]
        print(f"\nFixed模式参数: offset={fm_fixed.aperiodic_params_[0]:.3f}, "
              f"exponent={fm_fixed.aperiodic_params_[1]:.3f}")
    
    if result_knee['fooof_results'] and result_knee['fooof_results'][0]:
        fm_knee = result_knee['fooof_results'][0]
        print(f"Knee模式参数: offset={fm_knee.aperiodic_params_[0]:.3f}, "
              f"knee={fm_knee.aperiodic_params_[1]:.3f}, "
              f"exponent={fm_knee.aperiodic_params_[2]:.3f}")
    
    return {'fixed': result_fixed, 'knee': result_knee}


if __name__ == '__main__':
    # 运行所有示例
    print("\n🧠 ACF计算示例 - 脑电信号处理\n")
    
    result1_raw, result1_denoised = example_single_channel()
    result2 = example_multi_channel()
    result3 = example_comparison_methods()
    
    print("\n" + "=" * 60)
    print("✓ 所有示例完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - example_single_channel.png")
    print("  - example_multi_channel.png")
    print("  - example_fooof_comparison.png")
    print("\n关键参数说明:")
    print("  • rm_ap=True: 移除1/f背景噪声")
    print("  • fit_knee: FOOOF使用knee参数(更灵活的1/f模型)")
    print("  • response_f0: 响应基频(Hz)，用于谐波对齐")
    print("  • only_use_f0_harmonics: 仅保留基频的谐波")
    print("  • normalize_acf_to_1: ACF归一化到[-1, 1]")
