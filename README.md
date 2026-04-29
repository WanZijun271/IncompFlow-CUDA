# IncompFlow-CUDA

> GPU 加速的不可压缩稳态流动求解器  
> 结构化同位网格 · 有限体积法 · SIMPLE 算法

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-brightgreen)](https://developer.nvidia.com/cuda-toolkit)

---

## 📌 核心特性

| 类别 | 方法 |
|------|------|
| 控制方程 | 不可压缩 Navier‑Stokes，稳态 |
| 空间离散 | 有限体积法，结构化同位网格 |
| 对流项 | 一阶迎风格式 |
| 扩散项 | 中心差分（二阶精度） |
| 压力‑速度耦合 | **SIMPLE** 算法 + Rhie & Chow 动量插值 |
| 线性求解器（GPU） | Point Jacobi / Gauss‑Seidel |
| 硬件加速 | 全 CUDA 内核（网格循环、系数组装、求解） |

---

## 🔧 安装

### 系统要求

- **NVIDIA GPU**：计算能力 6.0 及以上（如 GTX 1060, RTX 2060, Tesla P100, V100, A100 等）
- **操作系统**：Linux（Ubuntu 18.04+ 推荐）
- **编译器**：支持 C++17（g++ 8+ / clang 10+ / MSVC 2019+）
- **构建工具**：CMake 3.15+
- **CUDA 工具包**：11.0 或更高版本

### 安装 CUDA（以 Ubuntu 为例）

```bash
# 下载并安装 CUDA 11.8（可根据需要选择更新版本）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

添加 CUDA 到环境变量（`~/.bashrc`）：

```bash
export PATH=/usr/local/cuda-11/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH
```

### 安装 CMake 及其他工具

```bash
sudo apt update
sudo apt install build-essential cmake git
```

---

## 📥 获取代码

```bash
git clone https://github.com/WanZijun271/IncompFlow-CUDA.git
cd IncompFlow-CUDA
```

---

## 🏗️ 编译

1. **创建 build 目录并运行 CMake**（需要指定你的 GPU 计算能力，例如 RTX 2080 为 `sm_75`，A100 为 `sm_80`）

   ```bash
   mkdir build && cd build
   cmake .. -DCUDA_ARCH=sm_75 -DCMAKE_BUILD_TYPE=Release
   ```

   > 查询 GPU 计算能力：`nvidia-smi --query-gpu=compute_cap --format=csv`，输出如 `7.5` 则对应 `sm_75`。

2. **编译**

   ```bash
   make -j$(nproc)
   ```

   编译成功后，可执行文件 `incompflow` 位于 `build/` 目录下。

---

## ⚙️ 配置算例

**所有运行参数均在 `include/config.h` 头文件中定义**。修改后需要重新编译。

---

## 🚀 运行

```bash
cd build
./incompflow
```

---

## 📄 许可证

本项目采用 MIT 许可证。详情见 [LICENSE](LICENSE) 文件。

---

## 📖 引用

如果你在学术研究或出版物中使用了本求解器，请引用：

```bibtex
@software{IncompFlow_CUDA,
  author = {WanZijun(万子珺)},
  title = {IncompFlow-CUDA: A GPU-Accelerated Finite Volume Solver for Steady Incompressible Flows},
  year = {2026},
  url = {https://github.com/WanZijun271/IncompFlow-CUDA},
  note = {SIMPLE algorithm, collocated grid, CUDA parallelization}
}
```

---

## 🙏 致谢

- 算法设计参考：
  - Moukalled, F., Mangani, L., & Darwish, M. (2016). *The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM® and Matlab*.
- CUDA 编程理念受 NVIDIA 官方文档启发。

---

**Happy simulating!** 🚀
