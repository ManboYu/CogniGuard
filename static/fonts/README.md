# 字体文件放这里

UI「暖调编辑感」需要两套思源字体（woff2 子集），放在本目录后由 Streamlit 静态托管（`enableStaticServing = true`）自动通过 `app/static/fonts/...` 提供。

请下载以下两个文件并放入本目录，**文件名必须一致**：

| 目标文件名 | 用途 | 推荐下载来源（任选其一） |
|---|---|---|
| `NotoSerifSC.woff2` | 标题（衬线） | https://cdn.jsdelivr.net/fontsource/fonts/noto-serif-sc@latest/chinese-simplified-700-normal.woff2 |
| `NotoSansSC.woff2` | 正文（无衬线） | https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-sc@latest/chinese-simplified-400-normal.woff2 |

可选（让标题更顺滑，非必需）：
| `NotoSansSC-Medium.woff2` | 强调/数值 500 | https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-sc@latest/chinese-simplified-500-normal.woff2 |

下载命令示例（在本目录执行）：

```bash
curl -L -o NotoSerifSC.woff2 "https://cdn.jsdelivr.net/fontsource/fonts/noto-serif-sc@latest/chinese-simplified-700-normal.woff2"
curl -L -o NotoSansSC.woff2  "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-sc@latest/chinese-simplified-400-normal.woff2"
curl -L -o NotoSansSC-Medium.woff2 "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-sc@latest/chinese-simplified-500-normal.woff2"
```

**字体缺失也不会报错**：CSS 已配置系统兜底（serif / system-ui），只是少了衬线那点编辑温度。放入文件后刷新页面即可生效。
