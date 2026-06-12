# CogniGuard 数据结构最小版

本文件锁定后续 Demo 的核心字段，避免 LLM、VLM、记忆模块和报告模块各自发明字段。

## 认知域 key

统一使用以下英文 key：

| key | 中文展示名 |
|---|---|
| orientation | 时间定向 |
| memory | 记忆 |
| language | 语言 |
| executive_function | 执行功能 |
| attention | 注意力 |
| visuospatial | 视觉空间 |

## 风险等级

统一使用：

- `low`
- `medium`
- `high`
- `unknown`

## 单次评估结果

```json
{
  "session_id": "demo-session-001",
  "participant_id": "demo-person-normal",
  "created_at": "2026-05-23T10:00:00+08:00",
  "is_mock": true,
  "domain_scores": {
    "orientation": 0.9,
    "memory": 0.8,
    "language": 0.9,
    "executive_function": 0.8,
    "attention": 0.8,
    "visuospatial": 0.7
  },
  "evidence": [
    {
      "domain": "memory",
      "source": "dialog",
      "text": "能回忆早饭内容，但对日期需要提示。"
    }
  ],
  "risk_level": "low",
  "explanation": "本次模拟会话未显示明显连续下降信号。",
  "disclaimer": "本系统仅为技术原型，输出内容仅作认知健康风险提示参考，不构成医学诊断或治疗建议。如有健康疑虑，请咨询专业医生。"
}
```

## 画钟分析结果

```json
{
  "session_id": "demo-session-001",
  "is_mock": true,
  "clock_findings": {
    "number_placement": "数字基本完整，间距略不均匀。",
    "hand_accuracy": "指针方向基本符合题目要求。",
    "visuospatial_evidence": [
      "数字集中在右侧",
      "圆形轮廓略不规则"
    ]
  },
  "risk_level": "low",
  "disclaimer": "本系统仅为技术原型，输出内容仅作认知健康风险提示参考，不构成医学诊断或治疗建议。如有健康疑虑，请咨询专业医生。"
}
```

## 趋势类型

Demo 数据至少包含三类轨迹：

- `normal`：整体稳定，认知域分数无明显下降。
- `mild_decline`：至少 3 次 session 中出现轻度下降趋势，尤其是 `visuospatial` 或 `memory`。
- `fluctuating`：分数有波动，但不简单单调下降。

## 模型失败兜底结果

当 LLM / VLM JSON 解析失败、schema 校验失败、API 调用失败时，禁止编造分数。统一返回：

```json
{
  "domain_scores": {
    "orientation": null,
    "memory": null,
    "language": null,
    "executive_function": null,
    "attention": null,
    "visuospatial": null
  },
  "evidence": [],
  "risk_level": "unknown",
  "explanation": "本次模型输出无法可靠解析，系统未生成有效评估结果。请重试或使用人工检查。",
  "disclaimer": "本系统仅为技术原型，输出内容仅作认知健康风险提示参考，不构成医学诊断或治疗建议。如有健康疑虑，请咨询专业医生。"
}
```
