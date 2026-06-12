# Cloudflare Quick Tunnel 临时公网部署

本方案用于 CogniGuard 临时公网演示，预计只保留约 15 天。

## 部署架构

- Streamlit 运行在服务器本地：`127.0.0.1:8501`。
- Cloudflare Quick Tunnel 将该本地端口暴露为 HTTPS 临时公网地址：`https://xxxx.trycloudflare.com`。
- 演示时直接在浏览器打开 `trycloudflare.com` 链接展示。

## 为什么选择这个方案

- 不需要域名。
- 不需要备案。
- 不需要 Nginx。
- 默认提供 HTTPS。
- 浏览器麦克风录音需要 HTTPS，因此比裸 IP + HTTP 更适合本项目。

## 局限

- `trycloudflare.com` 链接可能在 tunnel 重启后变化。
- 演示当天需要提前检查链接、麦克风、ASR 和 TTS。
- 适合短期演示，不是长期生产部署。

## 完整服务器操作步骤

以下步骤面向阿里云 2G 服务器。不要把真实 `.env`、API Key、音频缓存或 `data/` 提交到 git。

### A. 进入服务器项目目录

```bash
cd /opt/cogniguard
```

### B. 创建或更新虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### C. 创建服务器 `.env`

```bash
nano .env
```

说明：

- 不要把真实 Key 写进文档、README、脚本或 git。
- 参考 `.env.production.example` 填写服务器本地 `.env`。
- 真实 `.env` 只保存在服务器本地，不提交。
- `QWEN_API_KEY` 只放服务器本地。

### D. 运行测试

```bash
python -m pytest tests
```

### E. 本机启动 Streamlit

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

### F. 本机 smoke test

另开一个终端执行：

```bash
curl http://127.0.0.1:8501
```

### G. 安装 `cloudflared`

```bash
cd /tmp
wget -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/cloudflared
cloudflared --version
```

### H. 启动 Quick Tunnel

```bash
cloudflared tunnel --url http://127.0.0.1:8501
```

### I. 复制临时公网地址

`cloudflared` 输出的 HTTPS 地址就是临时公网地址，例如：

```text
https://xxxx.trycloudflare.com
```

### J. 后台运行方式

```bash
cd /opt/cogniguard
source .venv/bin/activate
nohup python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true > streamlit.log 2>&1 &
nohup cloudflared tunnel --url http://127.0.0.1:8501 > cloudflared.log 2>&1 &
```

### K. 查看临时公网地址

```bash
grep -o 'https://[-a-zA-Z0-9.]*trycloudflare.com' cloudflared.log | tail -1
```

### L. 查看日志

```bash
tail -f streamlit.log
tail -f cloudflared.log
```

### M. 停止服务

```bash
pkill -f streamlit
pkill -f cloudflared
```

### N. 演示当天检查清单

- 访问 `trycloudflare.com` 链接。
- 测试首页。
- 测试对话评估。
- 测试长者简易版。
- 测试演示。
- 测试麦克风授权。
- 测试 ASR。
- 测试 TTS。
- 测试认知简报。
- 如果链接失效，重启 `cloudflared` 并重新复制链接。

