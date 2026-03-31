# 金融智能体系统

一个基于AI的金融分析和交易策略开发系统，使用多个专业智能体协同工作。

## 项目概述

金融智能体系统是一个模块化架构，旨在通过一组专业的AI智能体自动完成金融分析。每个智能体都有特定的角色，协同工作生成全面的分析报告和交易策略。

## 架构设计

```
project/
 ├── .venv/                      # Python虚拟环境
 ├── .trae/                      # Trae IDE配置
 ├── documents/                  # 项目文档和数据文件
 ├── fin_agents_system/          # 主项目目录
 │   ├── backend/               # 后端服务
 │   │   ├── agents/           # 智能体目录
 │   │   │   ├── backtester.py      # 回测智能体
 │   │   │   ├── base_agent.py      # 基础智能体类
 │   │   │   ├── data_engineer.py   # 数据工程师智能体
 │   │   │   ├── sentiment_analyst.py # 情感分析师智能体
 │   │   │   └── technical_analyst.py # 技术分析师智能体
 │   │   ├── graph/            # 工作流定义
 │   │   │   └── workflow.py        # LangGraph工作流
 │   │   ├── models/           # 数据模型
 │   │   │   ├── database.py        # 数据库模型
 │   │   │   └── schemas.py         # Pydantic数据模型
 │   │   ├── ppo_tensorboard/  # PPO算法的TensorBoard日志
 │   │   ├── services/         # 服务层
 │   │   │   ├── backtest_service.py # 回测服务
 │   │   │   ├── data_service.py    # 数据服务
 │   │   │   └── llm_service.py     # 大模型服务
 │   │   ├── utils/            # 工具函数
 │   │   │   ├── data_fetcher.py    # 数据获取工具
 │   │   │   ├── indicators.py      # 技术指标计算
 │   │   │   ├── logger.py          # 日志工具
 │   │   │   ├── ppo_backtester.py  # PPO回测工具
 │   │   │   ├── trading_env.py     # 交易环境
 │   │   │   ├── transformer_model.py # 转换器模型
 │   │   │   └── web_crawler.py     # 网页爬虫
 │   │   ├── config.py         # 配置文件
 │   │   ├── financial_agents.db    # SQLite数据库文件
 │   │   ├── financial_agents.log   # 日志文件
 │   │   ├── log_config.yaml        # 日志配置文件
 │   │   └── main.py                # FastAPI主应用
 │   └── frontend/              # 前端界面
 │       ├── app.js                # 前端逻辑
 │       ├── index.html            # 主页面
 │       └── style.css             # 样式文件
 ├── init-scripts/              # 初始化脚本
 │   └── start_backend.bat         # 后端启动脚本
 ├── .python-version             # Python版本配置
 ├── financial_agents.db         # 主数据库文件（副本）
 ├── financial_agents.log        # 主日志文件（副本）
 ├── pyproject.toml              # 项目配置文件
 └── uv.lock                     # 依赖锁定文件
```

## 智能体介绍

### 数据工程师智能体
- **职责**: 从外部API获取、清洗和准备金融数据
- **工具**: 使用yfinance进行数据检索，支持web爬虫获取数据
- **输出**: 清洗和结构化的金融时间序列数据

### 情感分析师智能体
- **职责**: 分析新闻和社交媒体的市场情感
- **工具**: 使用在线大模型（如OpenAI GPT系列）进行情感分类
- **输出**: 情感分数和市场情感摘要

### 技术分析师智能体
- **职责**: 对金融数据进行技术分析
- **工具**: 技术指标（SMA、EMA、RSI、MACD、布林带等）
- **输出**: 交易信号和技术分析摘要

### 回测智能体
- **职责**: 针对历史数据测试交易策略
- **工具**: 自定义回测框架，支持PPO强化学习算法
- **输出**: 性能指标和策略评估

## 安装部署

### 前置条件
- Python 3.13+ 已安装
- 已安装uv依赖管理工具
- 已获取在线大模型API Key（如OpenAI API Key）

### 安装步骤

1. **项目已配置虚拟环境**
   项目根目录已包含 `.venv` 虚拟环境，可直接激活使用：
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **安装或更新依赖**
   ```bash
   # 使用uv安装或更新依赖
   uv sync
   ```

## 使用说明

### 启动后端服务

#### 使用启动脚本（推荐）
```bash
# Windows
# 在项目根目录下执行
init-scripts\start_backend.bat
```

#### 手动启动
```bash
# 在backend目录下执行
uvicorn main:app --host 0.0.0.0 --port 8000 --log-config log_config.yaml
```

API将在 `http://localhost:8000` 可用，若要自动重启请添加`--reload`参数

### API端点

#### 工作流
- `POST /run-workflow` - 运行完整分析工作流
  - 参数: symbol（股票代码）, url（数据爬取URL）, time_range（时间范围）, interval（数据间隔）
  - 返回: 包含情感分析、技术分析和回测结果的综合报告

#### 健康检查
- `GET /` - 根端点，返回系统运行状态
  - 返回: `{"message":"Financial Agents System API is running"}`
- `GET /health` - 健康检查端点
  - 返回: `{"status":"healthy"}`

### 前端使用

直接在浏览器中打开 `fin_agents_system/frontend/index.html` 即可访问用户界面。

## 配置说明

### 配置文件

主要配置项位于 `fin_agents_system/backend/config.py`，支持通过环境变量覆盖：

- `DEBUG` - 启用/禁用调试模式（默认: True）
- `LLM_MODEL` - 大语言模型名称（默认: qwen3:8b，使用本地Ollama模型）
- `DATABASE_URL` - 数据库连接URL（默认: sqlite:///./financial_agents.db）
- `LOG_LEVEL` - 日志级别（默认: INFO）
- `LOG_FILE` - 日志文件路径（默认: financial_agents.log）

## 工作流程

1. **数据获取**: 数据工程师智能体从外部来源（yfinance、Web爬虫）检索金融数据
2. **情感分析**: 情感分析师智能体使用本地Ollama大模型分析市场情感
3. **技术分析**: 技术分析师智能体计算多种技术指标并生成交易信号
4. **策略回测**: 回测智能体使用PPO强化学习算法针对历史数据测试策略
5. **结果生成**: 生成综合分析报告，包含情感分析、技术分析和回测结果

## 技术栈

- **后端框架**: FastAPI
- **工作流编排**: LangGraph
- **大模型集成**: LangChain + Ollama
- **金融数据**: yfinance
- **数据处理**: Pandas + Numpy
- **机器学习**: Scikit-learn
- **深度学习**: TensorFlow (用于PPO算法)
- **可视化**: Plotly + Matplotlib
- **数据库**: SQLite + SQLAlchemy
- **数据验证**: Pydantic
- **Web爬虫**: BeautifulSoup4
- **依赖管理**: uv

## 核心功能

1. **多智能体协作** - 专业智能体协同完成金融分析全流程
2. **本地大模型支持** - 使用Ollama本地模型，保护数据隐私
3. **强化学习回测** - 集成PPO算法，智能生成和测试交易策略
4. **多源数据获取** - 支持yfinance和Web爬虫获取金融数据
5. **全面技术分析** - 支持SMA、EMA、RSI、MACD、布林带等多种技术指标
6. **可视化训练** - 使用TensorBoard可视化PPO算法训练过程
7. **统一日志管理** - 所有组件使用统一日志格式，便于分析和调试
8. **模块化架构** - 各组件独立设计，便于扩展和维护

### 代码风格

```bash
# 格式化代码
black .

# 整理导入
isort .

# 检查代码质量
flake8
```

## 项目文件说明

- **fin_agents_system/backend/agents/** - 智能体实现
- **fin_agents_system/backend/graph/workflow.py** - 工作流定义
- **fin_agents_system/backend/ppo_tensorboard/** - PPO训练日志
- **fin_agents_system/frontend/** - 前端界面
- **init-scripts/start_backend.bat** - 后端启动脚本
- **documents/** - 项目文档和数据文件

## 路线图

- [x] 集成本地Ollama大模型
- [x] 实现多智能体协作工作流
- [x] 添加PPO强化学习回测
- [x] 实现Web爬虫数据获取
- [x] 添加统一日志格式
- [x] 数据工程师智能体测试脚本
- [ ] 支持更多金融数据来源
- [ ] 实现更多技术指标
- [ ] 支持实时数据流
- [ ] 添加投资组合优化功能
- [ ] 实现风险管理模块

