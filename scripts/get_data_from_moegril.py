import asyncio
import json
import random
import webbrowser
from pathlib import Path

import httpx
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def is_captcha_response(content: str) -> bool:
    """检测响应是否为验证码页面"""
    captcha_indicators = [
        "TCaptcha.js",
        "TencentCaptcha",
        "captcha.qq.com",
        "WafCaptcha",
    ]

    for indicator in captcha_indicators:
        if indicator in content:
            return True
    return False


async def get_data_url(
    search_query: str, max_retries: int = 3, failed_queries: list = []
) -> list[str]:
    """获取萌娘百科搜索结果URL"""
    retry_count = 0
    retry_delay = 2.0  # 初始延迟5秒

    while retry_count <= max_retries:
        try:
            # 添加用户代理和请求头，避免被拒绝
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",  # 添加防缓存头
                "Pragma": "no-cache",
                # 随机 X-Forwarded-For 可能导致更多问题，暂不添加
            }

            # 参数使用params正确编码
            params = {
                "action": "opensearch",
                "search": search_query,
                "limit": "500",
                "format": "json",
            }

            async with httpx.AsyncClient(headers=headers) as ctx:
                # 使用更长的超时时间
                resp = await ctx.get(
                    "https://zh.moegirl.org.cn/api.php", params=params, timeout=30.0
                )

                # 调试输出
                logger.debug(f"请求URL: {resp.url}")
                logger.debug(f"状态码: {resp.status_code}")

                if resp.status_code == 200:
                    # 检查返回内容的前缀，确保是JSON
                    content = resp.text.strip()
                    if content.startswith("[") and content.endswith("]"):
                        data = resp.json()
                        if len(data) >= 4:
                            return data[3]
                        logger.warning(f"响应结构不符合预期: {data}")
                    else:
                        # 检查是否为验证码响应
                        if is_captcha_response(content):
                            logger.error(f"遇到验证码拦截! 搜索查询: '{search_query}'")

                            # 记录验证码失败的查询
                            if failed_queries is not None and search_query not in [
                                q["query"] for q in failed_queries
                            ]:
                                failed_queries.append(
                                    {
                                        "query": search_query,
                                        "reason": "遇到验证码",
                                        "response_sample": content[
                                            :200
                                        ],  # 保存更多信息以便分析
                                    }
                                )

                            # 打开验证码页面
                            current_url = str(resp.url)
                            logger.info(f"正在打开浏览器处理验证码页面: {current_url}")
                            webbrowser.open(current_url)

                            # 验证码要求更长的等待时间
                            if retry_count < max_retries:
                                retry_count += 1
                                # 验证码情况下使用更长的随机等待时间 (15-30秒)
                                wait_time = random.uniform(180, 240)
                                logger.warning(
                                    f"遇到验证码，等待更长时间: {wait_time:.1f}秒后第 {retry_count}/{max_retries} 次重试..."
                                )
                                await asyncio.sleep(wait_time)
                                continue
                        else:
                            logger.error(f"非JSON响应: {content[:100]}...")
                            # 记录非JSON响应的查询
                            if failed_queries is not None and search_query not in [
                                q["query"] for q in failed_queries
                            ]:
                                failed_queries.append(
                                    {
                                        "query": search_query,
                                        "reason": "非JSON响应",
                                        "response_sample": content[:100],
                                    }
                                )
                else:
                    logger.error(f"HTTP错误: {resp.status_code}")

                # 如果状态码不是200，且重试次数未达上限，则重试
                if resp.status_code != 200 and retry_count < max_retries:
                    retry_count += 1
                    logger.warning(
                        f"请求失败，准备第 {retry_count}/{max_retries} 次重试，延迟 {retry_delay} 秒..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    continue

                # 如果是状态码200但非JSON响应，或者已达到重试上限，则退出循环
                break

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(
                    f"请求超时或连接错误: {e}，准备第 {retry_count}/{max_retries} 次重试，延迟 {retry_delay} 秒..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                logger.error(f"达到最大重试次数，请求失败: {e}")
                # 记录失败的查询
                if failed_queries is not None and search_query not in [
                    q["query"] for q in failed_queries
                ]:
                    failed_queries.append(
                        {
                            "query": search_query,
                            "reason": f"连接错误或超时: {str(e)}",
                            "response_sample": "",
                        }
                    )
                break
        except Exception as e:
            logger.error(f"获取数据错误：{e}")
            # 记录失败的查询
            if failed_queries is not None and search_query not in [
                q["query"] for q in failed_queries
            ]:
                failed_queries.append(
                    {
                        "query": search_query,
                        "reason": f"未知错误: {str(e)}",
                        "response_sample": "",
                    }
                )
            break

    return []


async def main():
    try:
        # 读取数据
        search_queries = ["蔚蓝档案"]
        students_file = Path(__file__).parents[1] / "data/students_min.json"

        # 用于记录失败的查询
        failed_queries = []

        with open(students_file, "r", encoding="utf-8") as f:
            students = json.load(f)

        # 收集所有角色全名
        students_set = []
        for student in students:
            if "fullName" in student and student["fullName"] not in students_set:
                students_set.append(student["fullName"])
                search_queries.append(student["fullName"])

        logger.info(f"共搜索 {len(search_queries)} 个关键词")
        data_urls = []

        # 显示进度条
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(None),
            TextColumn("[bold green]{task.completed}/{task.total}"),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("正在生成wiki URLs", total=len(search_queries))

            for search_query in search_queries:
                # 使用随机延迟 (1.5-3.5秒) 避免过快请求被限制
                delay = random.uniform(1.5, 3.5)
                await asyncio.sleep(delay)

                cur_urls = await get_data_url(
                    search_query, max_retries=3, failed_queries=failed_queries
                )
                if cur_urls:
                    data_urls.extend(cur_urls)
                    logger.info(f"搜索 '{search_query}' 找到 {len(cur_urls)} 条结果")
                else:
                    logger.warning(f"搜索 '{search_query}' 未找到结果")

                progress.update(task, advance=1)

                # 如果遇到验证码或其他错误，增加更长的冷却时间
                if any(q["query"] == search_query for q in failed_queries):
                    cool_down = random.uniform(30, 60)  # 30-60秒冷却
                    logger.warning(f"上次查询失败，增加 {cool_down:.1f} 秒冷却时间...")
                    await asyncio.sleep(cool_down)

        # 保存结果
        output_file = Path(__file__).parents[1] / "data/wiki_urls.json"
        logger.info(f"共获取 {len(data_urls)} 条URL，正在保存...")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_urls, f, ensure_ascii=False, indent=2)

        logger.success(f"成功保存数据到 {output_file}")

        # 保存失败的查询
        if failed_queries:
            failed_queries_file = Path(__file__).parents[1] / "data/failed_queries.json"
            logger.info(f"有 {len(failed_queries)} 个查询失败，正在保存...")

            with open(failed_queries_file, "w", encoding="utf-8") as f:
                json.dump(failed_queries, f, ensure_ascii=False, indent=2)

            logger.warning(f"已保存失败查询到 {failed_queries_file}")

    except Exception as e:
        logger.error(f"主程序错误: {e}")
        logger.exception("详细错误信息:")


if __name__ == "__main__":
    asyncio.run(main())
