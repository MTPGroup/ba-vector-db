import asyncio
from pathlib import Path
import os

from dashscope import TextEmbedding
from loguru import logger
from pymilvus import MilvusClient
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from utils import setup_logger

# 创建Rich控制台
console = Console()

# 配置
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "ba_articles"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "")


async def generate_embedding(query_text: str) -> list[float] | None:
    """使用DashScope生成文本嵌入向量"""
    try:
        response = TextEmbedding.call(
            model="text-embedding-v3",
            input=query_text,
            timeout=30,
            api_key=DASHSCOPE_API_KEY,
            workspace=DASHSCOPE_WORKSPACE_ID,
        )

        if response.status_code == 200:
            embedding = response.output["embeddings"][0]["embedding"]
            return embedding
        else:
            logger.error(f"DashScope API调用失败: {response.message}")
            return None
    except Exception as e:
        logger.error(f"生成向量嵌入时出错: {e}")
        return None


async def query_vector_db(query_text, top_k=5):
    """查询向量数据库"""
    client: MilvusClient | None = None
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
        ) as progress:
            # 连接到Milvus服务器
            connect_task = progress.add_task("[cyan]连接到Milvus服务器...", total=None)
            client = MilvusClient(str(Path(__file__).parents[1] / "data/ba_milvus.db"))
            progress.update(
                connect_task, description="[green]连接到Milvus服务器 ✓", completed=True
            )

            # 加载集合
            load_task = progress.add_task("[cyan]加载集合...", total=None)
            client.load_collection(COLLECTION_NAME)
            progress.update(load_task, description="[green]加载集合 ✓", completed=True)

            # 初始化模型
            model_task = progress.add_task("[cyan]初始化DashScope API...", total=None)
            progress.update(
                model_task, description="[green]初始化DashScope API ✓", completed=True
            )

            # 将查询文本转换为向量
            vector_task = progress.add_task("[cyan]生成查询向量...", total=None)
            query_vector = await generate_embedding(query_text)

            if query_vector is None:
                progress.update(
                    vector_task, description="[red]生成查询向量失败 ✗", completed=True
                )
                return []

            progress.update(
                vector_task, description="[green]生成查询向量 ✓", completed=True
            )

            # 执行搜索
            search_task = progress.add_task("[cyan]执行向量搜索...", total=None)
            # search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
            results = client.search(
                COLLECTION_NAME,
                data=[query_vector],
                anns_field="embedding",
                # param=search_params,
                limit=top_k,
                output_fields=["title", "category", "content"],
            )
            progress.update(
                search_task, description="[green]执行向量搜索 ✓", completed=True
            )

        # 处理结果
        responses = []
        for hits in results:
            for hit in hits:
                response = {
                    "distance": hit.get("distance", 0.0),
                    "title": hit.get("entity", {}).get("title"),
                    "category": hit.get("entity", {}).get("category"),
                    "content": hit.get("entity", {}).get("content"),
                }
                responses.append(response)

        return responses

    except Exception as e:
        console.print(f"[bold red]查询时出错: {str(e)}")
        logger.error(f"查询时出错: {e}")
        return []
    finally:
        # 释放集合资源
        try:
            if client is not None:
                client.release_collection(COLLECTION_NAME)
        except Exception as e:
            console.print(f"[bold red]释放集合资源时出错: {str(e)}")
            logger.error(f"释放集合资源时出错: {e}")

        # 断开连接
        try:
            if client is not None:
                client.close()
        except Exception as e:
            console.print(f"[bold red]断开连接时出错: {str(e)}")
            logger.error(f"断开连接时出错: {e}")


async def main_async():
    setup_logger()

    console.print(
        Panel.fit(
            "[bold cyan]蔚蓝档案向量数据库查询系统",
            border_style="blue",
            subtitle="输入 'exit' 或 'quit' 退出",
        )
    )

    # 检查DashScope API连接
    try:
        console.print("[bold yellow]正在检查DashScope API连接...")
        test_response = TextEmbedding.call(
            model="text-embedding-v3",
            input="测试连接",
            api_key=DASHSCOPE_API_KEY,
            workspace=DASHSCOPE_WORKSPACE_ID,
        )
        if test_response.status_code == 200:
            console.print("[green]DashScope API连接正常 ✓")
        else:
            console.print(f"[red]DashScope API返回错误: {test_response.message}")
            console.print("[yellow]查询功能可能无法正常工作")
    except Exception as e:
        console.print(f"[bold red]无法连接到DashScope API: {str(e)}")
        console.print("[yellow]请确保已正确设置DASHSCOPE_API_KEY环境变量")
        return

    while True:
        try:
            query = Prompt.ask("\n[bold green]请输入您的问题")

            if query.lower() in ["exit", "quit"]:
                console.print("[yellow]再见!")
                break

            if not query.strip():
                continue

            console.print("\n[bold blue]正在搜索相关内容...")
            results = await query_vector_db(query)

            if not results:
                console.print("[bold red]没有找到相关内容。")
                continue

            console.print("\n[bold green]找到以下相关内容:")

            # 创建结果表格
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("序号", style="dim", width=4)
            table.add_column("标题", style="cyan")
            table.add_column("分类", style="green")
            table.add_column("相关度", style="yellow", justify="right", width=8)

            for i, result in enumerate(results, 1):
                table.add_row(
                    str(i),
                    result["title"],
                    result["category"],
                    f"{result['distance']:.2f}",
                )

            console.print(table)

            # 显示详细内容
            for i, result in enumerate(results, 1):
                console.print(
                    f"\n[bold cyan]{i}. {result['title']} [yellow](相关度: {result['distance']:.2f})"
                )
                console.print(f"[bold green]分类:[/bold green] {result['category']}")

                # 显示内容摘要，使用语法高亮
                content_preview = result["content"][:300] + (
                    "..." if len(result["content"]) > 300 else ""
                )
                console.print(
                    Panel(
                        content_preview,
                        title="内容摘要",
                        title_align="left",
                        border_style="blue",
                    )
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]操作已取消。输入'exit'退出程序。")
            continue
        except Exception as e:
            console.print(f"[bold red]错误: {str(e)}")
            logger.exception("查询过程中出现错误")


def main():
    """同步入口点，调用异步主函数"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
