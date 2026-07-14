# data_index — 数据资产对应(单一真相源)

这里固定本实现用到的**全部** split。权威元信息(样本数、sha256、楼数、D 分布、图像来源)在
`manifest.json`。公开仓库通过 `pairuav.index verify-manifest` 校验数量和 SHA256;如需重生更丰富的统计,
必须从原始数据资产单独执行并审核 diff。

三份 `val10_*` 是已经用于 T107 的冻结历史 matched sample。原始 seeded sampler 留下少量重复行:
ongrid `26/2048`,offframe `4/2048`,offgrid `1/2048`(每个 pair 最多出现两次)。为保持已归档 cache、
预测和论文指标可复现,这里不静默去重;`manifest.json` 显式记录总行数、唯一 pair 数和重复行数。
训练集、主验证集和全量验证集继续强制 pair 唯一。

## 六个 index 一览

| index | 对数 | 楼 | 图像来源 | 一行的格式 | D | 用途 |
|---|---|---|---|---|---|---|
| `train_balanced_32768` | 32,768 | 631 | PairUAV `train_tour/` | `<楼>/<帧A>_<帧B>.json` | 全合法 | **训练**(按 D 均衡;占官方 701 楼总量 1.60%,占无泄漏 631 楼训练池 1.78%) |
| `val_quick_2048` | 2,048 | 70 | 同上 | 同上 | 全合法 | **主验证集**(选型/消融/论文表格默认口径) |
| `val_full_204120` | 204,120 | 70 | 同上 | 同上 | 全合法 | 全量验证(70 × 54 × 54,验抽样代表性) |
| `val10_ongrid_2048` | 2,048 | 70 | **UniV 45°/10fps** | `<楼>/<帧A 3位>_<帧B 3位>` | 全合法 | 泛化**对照组**(两帧都是官方帧) |
| `val10_offframe_2048` | 2,048 | 70 | 同上 | 同上 | 全合法 | 新视角,**标签可表示** |
| `val10_offgrid_2048` | 2,048 | 70 | 同上 | 同上 | **全非法** | 新视角 **+ 标签不可表示** |

> 631 + 70 = 701(PairUAV 全部训练楼);train 与 val 的楼**零交叉**。
> 三个 `val10_*` 使用**同 70 栋楼**且 |D| 分布已匹配,可在三档内部做 matched comparison。
> `val10_ongrid` 仅覆盖 10fps 线性区,排除异常的官方第 1 帧,因此不能与 `val_quick_2048`
> 的绝对 MAE 直接比较。

## 两种路径格式(别混)

- **`.json` 结尾** → PairUAV **官方 pair-json** 的相对路径,标签直接读官方文件。
  帧号是官方 54 帧编号(2 位)。
- **无 `.json`** → **10fps 帧号**(3 位)。PairUAV 对这些对**没有官方标注**,标签由采集轨迹的
  **闭式几何**生成(`pairuav/geometry.py`)。

## 帧 ↔ 轨迹步(这套 index 的地基)

```
官方 54 帧 → 步:  f = 1  → step 0
                  f ≥ 2  → step 4 + 5(f−2)          即 {0, 4, 9, …, 264}

2fps ↔ 10fps:     j = 5f − 2      ← **图像相似度实测**,残差 0.00;次近邻距离远 50 倍
                                     (md5 不同 —— 10fps 是重新编码的,但视觉上是同一帧)

10fps 帧 → 步:    step = j − 4    ← 仅在 j ∈ [8, 268] 成立
```

`f = 1`(→ j = 3)是**异常点**:它到 `f = 2` 只跨 4 步,之后每步都跨 5。
所以线性区从 **j = 8** 起 —— 这就是 `val10_*` 只取 j ∈ [8, 268] 的原因。

**合法 D = 211 个值**(含 0)= 官方 54 帧的 step 两两之差。
标签:`heading = wrap180(4·D)`,`range = −0.5·D`。

## 校验

```bash
# 公开资产的总行数 / 唯一 pair 数 / 重复行数 / sha256
python -m pairuav.index verify-manifest \
  --manifest data_index/manifest.json \
  --index-root data_index

# pair-json 类 index 与官方 json 目录对齐
python -m pairuav.index verify-json --index data_index/val_quick_2048.txt --json-dir "$PAIRUAV_TRAIN_JSON"

# 与已抽好的 VGGT 特征 cache 顺序对齐
python -m pairuav.index verify-cache --index data_index/val_quick_2048.txt --cache-dir "$PAIRUAV_CACHE_ROOT/val_nfull_s518"
```
