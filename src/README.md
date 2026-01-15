# ACF 自相关函数计算与1/f降噪 (Python版本)

从MATLAB `get_acf.m` 移植的核心ACF计算功能，专门用于脑电(EEG)信号处理。

## 功能特性

- ✅ 支持单通道 `(1, time)` 和多通道 `(Channel, time)` 脑电信号
- ✅ FFT计算与自相关函数(Wiener-Khintchin定理)
- ✅ **使用官方FOOOF库**进行1/f背景噪声拟合
- ✅ 1/f降噪与谐波保留
- ✅ 多种归一化选项
- ✅ 支持knee参数的灵活1/f建模

## 快速开始

### 安装依赖

```bash
# 基础依赖
conda install numpy scipy matplotlib

# FOOOF库 (必需)
pip install fooof
# 或使用新版本
pip install specparam
```

### 基本用法

```python
import numpy as np
from get_acf import get_acf

# 生成测试信号
fs = 250  # 采样率 250 Hz
t = np.arange(1000) / fs
signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz信号
signal = signal[np.newaxis, :]  # (1, 1000)

# 计算ACF
result = get_acf(signal, fs, rm_ap=False)

print(f"ACF shape: {result['acf'].shape}")
print(f"Lags (seconds): {result['lags'][:10]}")
```

### 去除1/f背景噪声

```python
# 使用FOOOF方法去除1/f
result = get_acf(
    signal, fs,
    rm_ap=True,                    # 移除1/f
    response_f0=10.0,              # 响应基频10 Hz
    fit_knee=False,                # 使用fixed模式(不带knee)
    only_use_f0_harmonics=True,    # 仅保留谐波
    normalize_acf_to_1=True        # ACF归一化到[-1, 1]
)

# 访问结果
acf = result['acf']                   # 自相关函数 (channel, lags)
lags = result['lags']                 # lag时间(秒)
mX = result['mX']                     # 幅度谱
ap_linear = result['ap_linear']       # 1/f估计
fooof_results = result['fooof_results']  # FOOOF拟合对象列表
```

### 访问FOOOF拟合参数

```python
# 获取第一个通道的FOOOF结果
fm = result['fooof_results'][0]

# 查看拟合参数
print(f"Offset: {fm.aperiodic_params_[0]:.3f}")
print(f"Exponent: {fm.aperiodic_params_[1]:.3f}")

# 如果使用knee模式
if len(fm.aperiodic_params_) == 3:
    print(f"Knee: {fm.aperiodic_params_[1]:.3f}")
    print(f"Exponent: {fm.aperiodic_params_[2]:.3f}")

# 检测到的峰
print(f"检测到 {len(fm.peak_params_)} 个峰")
```

### 多通道处理

```python
# 3通道脑电数据
eeg_data = np.random.randn(3, 1000)  # (3, 1000)

result = get_acf(
    eeg_data, fs,
    rm_ap=True,
    response_f0=10.0,
    verbose=1  # 显示处理进度
)

# 每个通道的ACF
for ch in range(3):
    print(f"Ch{ch}: ACF[0]={result['acf'][ch, 0]:.3f}")
    
    # 每个通道的FOOOF参数
    if result['fooof_results'][ch]:
        fm = result['fooof_results'][ch]
        print(f"  Exponent: {fm.aperiodic_params_[1]:.3f}")
```

## 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `x` | ndarray | - | 输入信号 `(1, time)` 或 `(channel, time)` |
| `fs` | float | - | 采样率(Hz) |
| `rm_ap` | bool | False | 是否移除1/f背景噪声 |
| `response_f0` | float | None | 响应基频(Hz)，用于排除已知谐波 |
| `fit_knee` | bool | False | 是否使用knee参数(灵活1/f模型) |
| `max_n_peaks` | int | 10 | FOOOF最大峰数 |
| `peak_threshold` | float | 2.0 | FOOOF峰检测阈值(标准差) |
| `only_use_f0_harmonics` | bool | True | 仅保留基频的谐波 |
| `ap_fit_flims` | tuple | (1, fs/2) | 1/f拟合频率范围 |
| `normalize_acf_to_1` | bool | True | ACF归一化到[-1, 1] |
| `verbose` | int | 0 | 输出详细程度 {0, 1, 2} |

## 返回值

```python
{
    'acf': ndarray,           # 自相关函数 (channel, lags)
    'lags': ndarray,          # lag时间(秒)
    'freq': ndarray,          # 频率数组(Hz)
    'mX': ndarray,            # 幅度谱 (channel, freq)
    'ap_linear': ndarray,     # 1/f估计 (channel, freq) or None
    'x_norm': ndarray,        # 1/f去除后的时间序列 or None
    'X_norm': ndarray,        # 1/f去除后的复数频谱 or None
    'fooof_results': list     # FOOOF对象列表 or None
}
```

## 示例脚本

运行完整示例查看三个演示:

```bash
python example_usage.py
```

生成的图像:
- `example_single_channel.png` - 单通道示例
- `example_multi_channel.png` - 多通道示例  
- `example_fooof_comparison.png` - FOOOF knee参数对比

## FOOOF: 为什么用官方库？

官方FOOOF库 (Donoghue et al., 2020) 提供：

✅ **经过充分验证** - 在多项神经科学研究中使用  
✅ **鲁棒的峰检测** - 自动识别和拟合振荡峰  
✅ **灵活的1/f模型** - 支持固定和knee模式  
✅ **详细的拟合报告** - R²、误差、参数置信区间等  
✅ **持续维护** - 活跃开发和bug修复

### FOOOF 1/f模型

```python
# Fixed模式 (简单)
log10(PSD) = offset - exponent × log10(f)

# Knee模式 (灵活)
log10(PSD) = offset - log10(knee + f^exponent)
```

**参数解释**：
- `offset`: 整体功率水平
- `exponent` (χ): 频谱倾斜度 (通常0.5-2.5)
  - χ ≈ 1: 粉红噪声 (1/f)
  - χ ≈ 2: 布朗噪声 (1/f²)
- `knee`: 低频饱和点(仅knee模式)

## 核心算法

### 自相关计算 (Wiener-Khintchin定理)

```python
# X: 复数频谱
# ACF = IFFT(|X|²)
acf = np.fft.ifft(X * np.conj(X), axis=-1).real
```

### 1/f降噪

```python
# 1. FOOOF拟合1/f分量 (ap_linear)
# 2. 从复数频谱减去1/f向量
X_norm = X - (X / |X|) * ap_linear
# 3. 信号低于噪声的位置设为0
X_norm[|X| < |X_ap|] = 0
```

## 与MATLAB版本的对应关系

| MATLAB | Python + FOOOF |
|--------|---------------|
| 手动曲线拟合 | `FOOOF.fit(freq, psd)` |
| `fit_aperiodic()` | `fm.aperiodic_params_` |
| `ap_par` | `fm` (FOOOF对象) |
| 峰检测 | `fm.peak_params_` |
| R² | `fm.r_squared_` |

## 测试

运行内置测试:

```bash
python get_acf.py
```

输出示例:
```
=== 测试单通道 (1, 1000) ===
拟合1/f分量 (使用FOOOF库)
  通道 1/1
...
FOOOF拟合: offset=-5.348, exponent=1.496
```

## 注意事项

1. **FOOOF库版本**: 官方推荐使用新版`specparam`替代`fooof`
   ```bash
   pip install specparam  # 推荐
   ```

2. **输入格式**: 必须是2D数组 `(channel, time)`

3. **频率范围**: `ap_fit_flims`默认`[1, fs/2]`，避免在0 Hz处拟合

4. **响应基频**: 提供`response_f0`可排除已知响应频率，改善1/f拟合质量

5. **knee参数**: 对于有低频饱和的数据，设置`fit_knee=True`

## 依赖

- **必需**:
  - numpy
  - scipy  
  - fooof (或 specparam)
  
- **可选**:
  - matplotlib (仅示例脚本需要)

## 参考文献

Donoghue, T., Haller, M., Peterson, E. J., et al. (2020). **Parameterizing neural power spectra into periodic and aperiodic components**. *Nature Neuroscience*, 23(12), 1655-1665.

GitHub: https://github.com/fooof-tools/fooof

## 许可

与原始MATLAB代码保持一致
