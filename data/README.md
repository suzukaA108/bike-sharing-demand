# 数据来源、许可与本地缓存

## 当前主线：Citi Bike JC + Open-Meteo

行程来源为 [Citi Bike 官方系统数据页](https://citibikenyc.com/system-data) 和其 [月度文件目录](https://s3.amazonaws.com/tripdata/index.html)。项目使用 `JC-` 发布月文件的服务范围计数，不做 NYC 全城或 Jersey City 行政边界过滤。数据包含 ride ID、出发/结束时间、站点、坐标和会员类别等字段。

天气来源为 [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)。实验使用历史网格小时值（温度、相对湿度、降水、风速），不是当时发布的天气预报。最终回测的实际信息可用性和网格对 JC 服务范围的代表性有限。

### 使用与发布边界

- Citi Bike 数据受官方 [Data Sharing Policy](https://www.citibikenyc.com/data-sharing-policy) 约束。仓库不分发原始压缩包或逐行行程，不暗示 Citi Bike 官方背书；保留来源归属。
- Open-Meteo 免费 API 面向非商业用途，API 数据依其条款和 CC BY 4.0 进行署名。查看 [Open-Meteo Terms](https://open-meteo.com/en/terms) 和 API 文档；不要把当前免费 API 使用方式描述为可直接商用。
- 本仓库只保留审计摘要、模型指标、聚合误差切片和图表。逐小时实际/预测明细、行程归档及天气缓存在 Git 忽略路径中。
- 发布或商业使用前，应重新核对上游政策、归属要求与当前服务限制。

### 获取与准备

从仓库根目录依次运行：

```bash
python -m src.jc_download
python -m src.jc_year_audit
python -m src.jc_prepare
```

压缩归档缓存到 `data/raw/jc/`；准备后的小时时间序列缓存到 `data/processed/`。各文件 SHA-256、大小和审计细节写入本地生成的结果文件。

不能仅按文件名决定行程所属月份：相邻月归档可能包含跨月出发记录，须按实际 `started_at` 统计。零出发仅在完整月份归档存在且校验有效时记为 0；源系统漏报仍无法排除。

源时间戳没有 UTC 偏移，按 America/New_York 墙上时间解释。秋季重复小时无法消歧的行程排除，对应 UTC 目标小时不评分。全量重复与 DST 处理见 [全量审计](../results/JC_FULL_AUDIT.md)。

## 历史原型：UCI Bike Sharing

保留的 [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) 仅作为旧方案原型，下载说明由 `python -m src.data --download` 提供；它不属于 JC 主实验结果。UCI 引用：Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.
