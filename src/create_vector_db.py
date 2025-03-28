import argparse
import asyncio
import hashlib
import os
import re
import time
from pathlib import Path

import jieba
from dashscope import TextEmbedding
from dotenv import load_dotenv
from loguru import logger
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)
from pymilvus.milvus_client import IndexParams
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils import setup_logger

load_dotenv()

# 配置日志
logger.add("vector_db_creation.log", rotation="10 MB")
setup_logger()

# 创建Rich控制台
console = Console()

# 配置变量
ARTICLES_PATH = Path(__file__).parents[1] / "data/articles"
CHUNK_SIZE = 500  # 每个文本块的最大字符数
CHUNK_OVERLAP = 100  # 文本块之间的重叠字符数
EMBEDDING_DIM = 1024  # DashScope的text-embedding-v3模型输出1024维向量
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "ba_articles"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "")

# DashScope 并发限制和计数器
MAX_CONCURRENT_REQUESTS = 5  # DashScope API并发限制
semaphore = None  # 将在main函数中初始化
request_counter = 0  # 请求计数器
last_request_time = 0  # 上次请求时间


def load_markdown_files() -> list[dict[str, str]]:
    """加载所有Markdown文件"""
    markdown_files: list[dict[str, str]] = []

    console.print(Panel("扫描Markdown文件...", title="加载数据", border_style="green"))

    # 获取文件列表
    file_paths = list(ARTICLES_PATH.rglob("**/*.md"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]读取文件...", total=len(file_paths))

        # 遍历articles目录下的所有.md文件
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取文件名作为标题
            title = file_path.stem

            # 获取相对路径作为分类
            rel_path = file_path.relative_to(ARTICLES_PATH)
            category = rel_path.parent.as_posix()
            if not category:
                category = "general"

            markdown_files.append(
                {
                    "title": title,
                    "content": content,
                    "path": str(rel_path),
                    "category": category,
                }
            )
            progress.update(task, advance=1)

    logger.info(f"加载了 {len(markdown_files)} 个Markdown文件")
    return markdown_files


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    use_semantic: bool = True,
) -> list[str]:
    """将文本切分为重叠的块"""
    md_blocks = re.split(
        r"(\n#{1,6} .+?\n|\n```.+?```\n|\n- .+?\n)", text, flags=re.DOTALL
    )

    chunks = []
    current_chunk = []
    current_len = 0

    for block in md_blocks:
        if not block.strip():
            continue

        # 处理Markdown语法块（标题、代码块、列表）
        if re.match(r"(\n#{1,6} |\n```|\n- )", block, flags=re.DOTALL):
            if current_len + len(block) > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = current_chunk[
                    -max(0, len(current_chunk) - chunk_overlap // 50) :
                ]
                current_len = sum(len(s) for s in current_chunk)
            current_chunk.append(block)
            current_len += len(block)
            continue

        # 语义分割处理
        if use_semantic:
            sentences = _split_sentences(block)
        else:
            sentences = [block]

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # 当前块容量检查
            if current_len + len(sent) > chunk_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    # 重叠保留策略
                    overlap = max(0, chunk_overlap - 50)
                    current_chunk = (
                        current_chunk[-overlap // 50 :] if overlap > 0 else []
                    )
                    current_len = sum(len(s) for s in current_chunk)

            current_chunk.append(sent)
            current_len += len(sent)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """结合jieba的语义分割"""

    # 使用精确模式+标点分割
    words = list(jieba.cut(text, cut_all=False))
    sentences = []
    current_sent = []

    # 合并词语并识别句子边界
    for word in words:
        current_sent.append(word)
        # 句子结束符检测（支持中文标点）
        if re.match(r"[。！？…；]$", word):
            sentences.append("".join(current_sent))
            current_sent = []

    # 处理剩余内容
    if current_sent:
        sentences.append("".join(current_sent))

    return sentences


def create_milvus_collection() -> MilvusClient:
    """创建Milvus集合"""
    try:
        # 连接Milvus服务
        client = MilvusClient(str(Path(__file__).parents[1] / "data/ba_milvus.db"))

        # 检查集合是否存在，如果存在则删除
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
            logger.info(f"删除了现有的集合: {COLLECTION_NAME}")

        # 定义集合字段
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100
            ),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM
            ),
        ]

        # 创建集合模式
        schema = CollectionSchema(fields, "文章集合")

        # 创建集合
        client.create_collection(COLLECTION_NAME, schema=schema)

        # 创建索引
        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            # index_type="HNSW",
            # metric_type="COSINE",
            # params={"M": 8, "efConstruction": 64},
        )
        client.create_index(COLLECTION_NAME, index_params=index_params)

        logger.info(f"创建了新的Milvus集合: {COLLECTION_NAME}")
        # return collection
        return client

    except Exception as e:
        logger.error(f"创建Milvus集合时出错: {e}")
        raise


async def generate_embedding_with_rate_limit(text: str) -> list[float] | None:
    """
    使用DashScope API生成嵌入向量，并包含速率限制逻辑
    """
    global request_counter, last_request_time, semaphore

    if semaphore is None:
        logger.error("并发器未初始化")
        return

    # 使用信号量控制并发请求数
    async with semaphore:
        # 添加请求间隔以避免触发QPS限制
        current_time = time.time()
        if current_time - last_request_time < 0.2:  # 确保请求间隔至少200毫秒
            await asyncio.sleep(0.2 - (current_time - last_request_time))

        try:
            # 重试逻辑
            for attempt in range(3):
                try:
                    response = TextEmbedding.call(
                        model="text-embedding-v3",
                        input=text,
                        timeout=30,
                        api_key=DASHSCOPE_API_KEY,
                        workspace=DASHSCOPE_WORKSPACE_ID,
                    )

                    if response.status_code == 200:
                        # 请求成功，更新计数器和时间戳
                        request_counter += 1
                        last_request_time = time.time()

                        # 返回嵌入向量
                        embedding = response.output["embeddings"][0]["embedding"]
                        return embedding
                    else:
                        logger.warning(
                            f"DashScope API请求失败，状态码: {response.status_code}, 错误信息: {response.message}"
                        )
                        if "rate limit exceeded" in response.message.lower():
                            # 如果是速率限制错误，等待更长时间
                            await asyncio.sleep(2 + attempt * 2)
                        else:
                            await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"生成嵌入向量时出错: {e}")
                    await asyncio.sleep(1)

            # 所有尝试都失败，返回空向量
            logger.error(f"无法为文本生成嵌入向量(长度: {len(text)}): {text[:50]}...")
            return [0.0] * EMBEDDING_DIM

        except Exception as e:
            logger.error(f"调用DashScope API时发生异常: {e}")
            return [0.0] * EMBEDDING_DIM


async def process_batch(
    batch_data: dict,
    client: MilvusClient,
    progress: Progress,
    embedding_task,
    insert_task,
    chunks_processed: float,
) -> float:
    """处理一批数据并插入到Milvus"""

    if not batch_data["ids"]:
        return chunks_processed

    try:
        # 批量生成嵌入任务
        embedding_tasks = []
        for content in batch_data["contents"]:
            embedding_tasks.append(generate_embedding_with_rate_limit(content))

        # 并行执行所有嵌入任务
        embeddings = await asyncio.gather(*embedding_tasks)

        # 更新进度
        progress.update(embedding_task, advance=len(batch_data["ids"]))
        progress.update(insert_task, visible=True)
        progress.update(insert_task, completed=chunks_processed)

        records = []
        for i in range(len(batch_data["ids"])):
            record = {
                "id": batch_data["ids"][i],
                "title": batch_data["titles"][i],
                "category": batch_data["categories"][i],
                "chunk_index": batch_data["chunk_indices"][i],
                "content": batch_data["contents"][i],
                "embedding": embeddings[i],
            }
            records.append(record)

        # 插入到Milvus
        client.insert(COLLECTION_NAME, records)

        logger.info(f"插入了{len(batch_data['ids'])}个文本块")
        chunks_processed += len(batch_data["ids"])

        # 更新进度
        progress.update(insert_task, completed=chunks_processed)

    except Exception as e:
        logger.error(f"批处理插入时出错: {e}")

    return chunks_processed


async def main_async(is_test: bool):
    global semaphore

    try:
        console.print(
            Panel("启动蔚蓝档案向量数据库创建任务", border_style="blue", title="初始化")
        )

        # 初始化信号量控制并发请求
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        # 加载Markdown文件
        markdown_files = load_markdown_files()
        if is_test:
            markdown_files = markdown_files[:5]

        # 检查DashScope API是否可用
        with console.status("[bold green]测试DashScope API连接...", spinner="dots"):
            try:
                test_response = TextEmbedding.call(
                    model="text-embedding-v3",
                    input="测试连接",
                    api_key=DASHSCOPE_API_KEY,
                    workspace=DASHSCOPE_WORKSPACE_ID,
                )
                if test_response.status_code == 200:
                    console.print("[green]DashScope API连接成功 ✓")
                else:
                    console.print(
                        f"[red]DashScope API返回错误: {test_response.message}"
                    )
                    return
            except Exception as e:
                console.print(f"[red]无法连接到DashScope API: {e}")
                console.print("[yellow]请确保已正确设置DASHSCOPE_API_KEY环境变量")
                return

        # 创建Milvus集合
        with console.status("[bold yellow]创建Milvus集合...", spinner="dots"):
            client = create_milvus_collection()
            console.print("[green]Milvus集合创建完成 ✓")

        # 处理文件并插入数据库
        logger.info("开始处理文件并插入数据库...")

        total_chunks = 0

        # 首先计算总块数以便更精确地显示进度
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]预处理"),
            BarColumn(None),
            TaskProgressColumn(),
        ) as progress:
            count_task = progress.add_task(
                "[yellow]计算文本块数量...", total=len(markdown_files)
            )
            for file_data in markdown_files:
                content = file_data["content"]
                chunks = chunk_text(content)
                total_chunks += len(chunks)
                progress.update(count_task, advance=1)

        console.print(f"[bold]总计需处理 {total_chunks} 个文本块")

        # 处理每一个文件并生成嵌入
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            embedding_task = progress.add_task("[cyan]生成嵌入向量", total=total_chunks)
            insert_task = progress.add_task(
                "[green]插入数据库", total=total_chunks, visible=False
            )

            batch_size = 20  # DashScope处理较小批量更合适
            chunks_processed = 0

            # 处理批处理数据的缓存
            batch_data = {
                "ids": [],
                "titles": [],
                "categories": [],
                "chunk_indices": [],
                "contents": [],
            }

            # 处理每一个文件
            for file_data in markdown_files:
                title = file_data["title"]
                content = file_data["content"]
                category = file_data["category"]

                # 将文本分块
                chunks = chunk_text(content)

                # 更新描述，显示当前处理的文件
                progress.update(
                    embedding_task, description=f"[cyan]生成嵌入向量 - {title}"
                )

                # 对每个块生成嵌入
                for i, chunk in enumerate(chunks):
                    # 生成唯一ID
                    chunk_id = hashlib.md5(f"{title}_{i}".encode()).hexdigest()

                    # 添加到批处理
                    batch_data["ids"].append(chunk_id)
                    batch_data["titles"].append(title)
                    batch_data["categories"].append(category)
                    batch_data["chunk_indices"].append(i)
                    batch_data["contents"].append(chunk)

                    # 当积累了足够多的数据时进行批处理
                    if len(batch_data["ids"]) >= batch_size:
                        # 执行批处理
                        chunks_processed = await process_batch(
                            batch_data,
                            client,
                            progress,
                            embedding_task,
                            insert_task,
                            chunks_processed,
                        )

                        # 清空批处理数据
                        batch_data = {
                            "ids": [],
                            "titles": [],
                            "categories": [],
                            "chunk_indices": [],
                            "contents": [],
                        }

            # 处理剩余的批次
            if batch_data["ids"]:
                chunks_processed = await process_batch(
                    batch_data,
                    client,
                    progress,
                    embedding_task,
                    insert_task,
                    chunks_processed,
                )

        # 刷新集合使数据可搜索
        with console.status("[bold cyan]刷新集合数据...", spinner="dots"):
            client.flush(COLLECTION_NAME)

        # 输出集合统计数据
        console.print(
            f"[bold green]总共插入了 {client.get_collection_stats(COLLECTION_NAME).get('row_count', 0)} 个文本块到集合 {COLLECTION_NAME} 中"
        )

        # 加载集合
        with console.status("[bold magenta]加载集合到内存...", spinner="dots"):
            client.load_collection(COLLECTION_NAME)

        # 测试搜索功能
        console.print("[bold]执行测试查询...", style="yellow")
        test_query = "简要介绍一下蔚蓝档案"

        with console.status(f"[bold]查询: {test_query}", spinner="dots"):
            test_vector = await generate_embedding_with_rate_limit(test_query)
            # search_params = {"metric_type": "COSINE", "params": {"ef": 10}}
            result = client.search(
                COLLECTION_NAME,
                data=[test_vector],
                anns_field="embedding",
                # param=search_params,
                limit=3,
                output_fields=["title", "content"],
            )

        console.print("[bold]查询内容:", style="cyan")
        console.print(f"[green]: {test_query}")
        console.print("[bold]查询结果:", style="cyan")
        for i, hits in enumerate(result):
            for j, hit in enumerate(hits):
                console.print(
                    f"[bold cyan]{j + 1}. {hit.get('entity', {}).get('title', '')} [green](匹配度: {hit.get('distance', 0):.4f})"
                )
                console.print(
                    f"[blue]内容摘要: [white]{hit.get('entity', {}).get('content')[:100]}..."
                )

        console.print(
            Panel("Milvus向量数据库创建并填充成功!", title="完成", border_style="green")
        )

    except Exception as e:
        console.print(f"[bold red]处理失败: {str(e)}")
        logger.error(f"处理失败: {e}")
        raise


def main():
    """同步入口点，调用异步主函数"""
    parser = argparse.ArgumentParser(description="创建BA Milvus数据库")
    parser.add_argument("-t", "--test", action="store_true", help="是否开启测试模式")
    args = parser.parse_args()
    asyncio.run(main_async(args.test))


if __name__ == "__main__":
    main()
