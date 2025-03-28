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
    TimeElapsedColumn,
    TimeRemainingColumn,
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


def convert_to_markdown(text: str) -> str:
    """将MediaWiki文本转换为Markdown格式"""
    import re

    # 转换标题格式 == 标题 == -> ## 标题
    text = re.sub(r"^=== (.*?) ===$", r"### \1", text, flags=re.MULTILINE)
    text = re.sub(r"^== (.*?) ==$", r"## \1", text, flags=re.MULTILINE)
    text = re.sub(r"^= (.*?) =$", r"# \1", text, flags=re.MULTILINE)

    # 处理空行（保持段落格式）
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


async def get_mds(title: str, progress=None, task_id=None):
    """
    获取指定标题的萌娘百科内容并保存为Markdown格式

    Args:
        title: 页面标题
        progress: Rich进度条对象
        task_id: 任务ID用于更新进度
    """
    try:
        logger.info(f"正在获取页面: {title}")
        base_dir = Path(__file__).parents[1] / "data"
        base_dir.mkdir(exist_ok=True)

        # 添加用户代理和请求头，避免被拒绝
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://zh.moegirl.org.cn/",
            "Origin": "https://zh.moegirl.org.cn/",
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exlimit": 1,
            "format": "json",
        }

        async with httpx.AsyncClient(headers=headers) as ctx:
            resp = await ctx.get(
                "https://zh.moegirl.org.cn/api.php", params=params, timeout=30.0
            )

            if resp.status_code == 200:
                # 检查是否为验证码页面
                content = resp.text
                if is_captcha_response(content):
                    logger.warning(f"遇到验证码页面: {title}")
                    # 打开验证码页面让用户处理
                    current_url = str(resp.url)
                    logger.info(f"正在打开浏览器处理验证码: {current_url}")
                    webbrowser.open(current_url)

                    # 等待一段时间让用户处理验证码
                    wait_time = 30  # 等待30秒
                    logger.info(f"等待 {wait_time} 秒以便完成验证码...")
                    await asyncio.sleep(wait_time)

                    if progress and task_id:
                        progress.update(
                            task_id, description=f"[yellow]需验证码: {title}"
                        )
                    return False

                try:
                    data = resp.json()
                    pages = data.get("query", {}).get("pages", {})
                    if not pages:
                        logger.warning(
                            f"获取页面失败，未找到页面内容: {title}\nurl: {resp.url}"
                        )
                        if progress and task_id:
                            progress.update(task_id, description=f"[red]失败: {title}")
                        return False

                    page_id = next(iter(pages))
                    if "extract" not in pages[page_id]:
                        logger.warning(f"页面内容为空: {title}")
                        if progress and task_id:
                            progress.update(
                                task_id, description=f"[yellow]空内容: {title}"
                            )
                        return False

                    raw_text = pages[page_id]["extract"]
                    markdown_text = convert_to_markdown(raw_text)

                    output_file = base_dir / (
                        f"MomoTalk/{title.split('/')[0]}.md"
                        if "MomoTalk" in title
                        else f"articles/{title}.md"
                    )
                    output_file.parent.mkdir(exist_ok=True, parents=True)
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(markdown_text)

                    logger.success(f"已保存: {output_file}")
                    if progress and task_id:
                        progress.update(task_id, description=f"[green]完成: {title}")
                    return True
                except json.JSONDecodeError:
                    logger.error(f"返回内容不是有效的JSON: {title}")
                    # 可能是验证码或其他非JSON响应，尝试直接访问页面
                    direct_url = f"https://zh.moegirl.org.cn/{title}"
                    logger.info(f"尝试直接打开页面: {direct_url}")
                    webbrowser.open(direct_url)
                    if progress and task_id:
                        progress.update(task_id, description=f"[red]格式错误: {title}")
                    return False
            else:
                logger.error(f"请求失败，状态码: {resp.status_code}, 页面: {title}")
                if progress and task_id:
                    progress.update(task_id, description=f"[red]失败: {title}")
                return False

    except Exception as e:
        logger.error(f"处理 {title} 时出错: {e}")
        if progress and task_id:
            progress.update(task_id, description=f"[red]错误: {title}")
        return False


async def main():
    try:
        # 读取wiki页面标题，如果是URL列表则转换为标题
        wiki_file_path = Path(__file__).parents[1] / "data/wiki_titles.json"
        with open(wiki_file_path, "r", encoding="utf-8") as f:
            titles = json.load(f)

        total = len(titles)
        logger.info(f"准备获取 {total} 个页面")

        successful = 0
        failed = 0

        # 创建一个Rich进度条
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("获取萌娘百科页面", total=total)

            for title in titles:
                # 添加随机延时（1.5-4秒），避免请求过快被限制
                delay = random.uniform(1.5, 4.0)
                logger.debug(f"等待 {delay:.2f} 秒")
                await asyncio.sleep(delay)

                result = await get_mds(title, progress, task_id)
                progress.update(task_id, advance=1)
                if result:
                    successful += 1
                else:
                    failed += 1
                    # 如果失败，增加额外冷却时间避免被封
                    await asyncio.sleep(random.uniform(3.0, 6.0))

        # 总结
        logger.info(f"任务完成! 成功: {successful}, 失败: {failed}, 总计: {total}")
    except Exception as e:
        logger.error(f"主程序错误: {e}")
        logger.exception("详细错误信息:")


if __name__ == "__main__":
    # 配置日志格式
    asyncio.run(main())
