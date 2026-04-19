# python_demos/ — Python Demo Scripts

教学级 Python 演示脚本，从 3DGS 渲染到训练完整链路，每个步骤都有中文详细注释。

## Files

| File | Description | Maps To Chapters |
|------|-------------|------------------|
| `full_pipeline.py` | Complete rendering → training pipeline | Ch03, Ch04, Ch05, Ch07 |
| `gaussian_math.py` | Standalone Gaussian math (quat→matrix, cov building) | Ch02, Ch06 |
| `sh_color_demo.py` | Spherical Harmonics color evaluation demo | Ch09 (Prob/Info) |

## Prerequisites

```bash
pip install torch numpy matplotlib
```

## Quick Start

```bash
python full_pipeline.py
```

Run outputs training simulation with 20 iterations, densification every 5 steps.

## How to Use

Each script is designed as a **teaching tool** — not production code:
- Comments explain the math behind each operation
- Sections map directly to tutorial chapters
- Simplified versions of complex algorithms (full impl in `diff-gaussian-rasterization`)

To visualize rendered images, add matplotlib at end of scripts:
```python
import matplotlib.pyplot as plt
plt.imshow(rendered.cpu().numpy())
plt.show()
```
