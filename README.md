# 虚拟电厂调度与售电预测平台

这是一个基于 Streamlit 开发的电力调度演示系统，集成了负荷预测、多源数据采集（气象+POI）以及储能削峰填谷策略分析。

## 功能特性
- **数据采集**：集成 Open-Meteo 气象数据与 OpenStreetMap 商业点位数据。
- **负荷预测**：基于 Ridge 回归算法预测区域新增负荷。
- **调度决策**：自动生成削峰填谷策略，优化储能充放电。
- **经济分析**：对比无调度与有调度场景下的成本与营收。

## 快速启动 (本地)

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
streamlit run app.py
```
或者直接双击 `快速启动.bat`。

## 如何部署到网页 (Streamlit Cloud)

本项目已准备好部署所需的配置文件。

1. **上传代码到 GitHub**
   - 注册/登录 [GitHub](https://github.com/)。
   - 创建一个新的仓库 (Repository)，例如命名为 `vpp-system`。
   - 将本项目所有文件上传到该仓库。

2. **部署到 Streamlit Cloud**
   - 访问 [Streamlit Community Cloud](https://streamlit.io/cloud)。
   - 使用 GitHub 账号登录。
   - 点击 "New app"。
   - 选择刚才创建的 GitHub 仓库。
   - **Main file path** 填写 `app.py`。
   - 点击 "Deploy"。

等待几分钟，应用即可上线，你可以将生成的 URL 分享给其他人。
