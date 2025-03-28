# 蔚蓝档案向量数据库项目

## 项目介绍

这是一个基于向量数据库的蔚蓝档案（Blue Archive）资料检索系统。该系统使用 Milvus 作为向量数据库，通过 DashScope 的文本嵌入服务将文档内容转换为向量，实现高效的语义搜索功能。

## 功能特点

- 自动处理 Markdown 格式的文档资料
- 智能文本分块，保持语义完整性
- 使用阿里云 DashScope 的文本向量嵌入模型
- 基于 Milvus 的高效向量搜索
- 友好的命令行交互界面
- 详细的进度显示和日志记录

## 环境要求

- Python >= 3.12
- 阿里云 DashScope API 密钥
- Milvus 向量数据库（本项目使用嵌入式 Milvus）

## 安装步骤

1. 克隆本仓库：

   ```bash
   git clone https://github.com/yourusername/ba-vector-db.git
   cd ba-vector-db
   ```

2. 安装依赖：
   <details open>
      <summary>使用uv</summary>

   ```bash
   uv sync
   ```

   </details>

   <details>
   <summary>使用pip</summary>

   ```bash
   pip install -e .
   ```

   </details>

3. 配置环境变量：
   - 复制 `.env.example` 为 `.env`
   - 在 `.env` 文件中填入您的 DashScope API 密钥和工作空间 ID

## 使用方法

### 数据准备

将您的 Markdown 文档放在 `data/articles/` 目录下，可以使用子目录进行分类组织。

### 创建向量数据库

<details open>
   <summary>使用uv</summary>

```bash
uv run src/create_vector_db.py
```

</details>

<details>
   <summary>使用pip</summary>

```bash
python src/create_vector_db.py
```

</details>

支持的选项：

- `-t`, `--test`：使用测试模式（仅处理少量文档）

### 查询向量数据库

<details>
   <summary>使用uv<summary>

```bash
uv run src/query_vector_db.py
```

</details>

<details>
   <summary>使用pip</summary>

```bash
python src/query_vector_db.py
```

</details>

在交互式界面中输入您的问题，系统会返回最相关的文档片段。

## 项目结构

```
ba-vector-db/
├── .env.example       # 环境变量示例文件
├── data/
│   ├── articles/      # 存放 Markdown 文档
│   └── ba_milvus.db/  # Milvus 数据库文件（自动创建）
├── src/
│   ├── create_vector_db.py  # 创建向量数据库脚本
│   ├── query_vector_db.py   # 查询向量数据库脚本
│   └── utils.py             # 工具函数
└── pyproject.toml     # 项目配置文件
```

## 技术细节

- 文本分块：使用结合语义边界的重叠分块策略
- 向量嵌入：DashScope text-embedding-v3 模型（1024维向量）
- 向量搜索：基于 Milvus 的 AUTOINDEX 索引
- 中文分词：jieba 分词库

## 数据来源

- [萌娘百科](https://zh.moegirl.org.cn)

## 许可证

[GNU General Public License v3.0](LICENSE)

## 贡献

欢迎提交 Issue 或 Pull Request 来帮助改进本项目。
