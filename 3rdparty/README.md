# 3rdparty

这里存放第三方仓库。

当前依赖:

- `vggt/`: 官方 VGGT 仓库,作为 git submodule 固定在 `a288dd0f14786c93483e45524328726ab7b1b4ce`。

约定:

- 不在第三方源码里实现 PairUAV 逻辑;
- 如果将来确实需要第三方补丁,必须在 `docs/` 中明确记录原因和 patch 内容;
- PairUAV 专用代码应放在 `pairuav/`。
