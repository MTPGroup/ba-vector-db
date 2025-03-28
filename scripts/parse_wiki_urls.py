import json
from pathlib import Path
import urllib.parse
from loguru import logger

def parse_wiki_urls(input_file_path, output_file_path):
    """
    解析萌娘百科URL并提取中文标题
    
    Args:
        input_file_path: 输入的URL JSON文件路径
        output_file_path: 输出的解析结果JSON文件路径
    """
    logger.info(f"开始解析萌娘百科URL: {input_file_path}")
    
    # 读取URL文件
    with open(input_file_path, 'r', encoding='utf-8') as f:
        urls = json.load(f)
    
    logger.info(f"读取到 {len(urls)} 个URL")
    
    # 创建URL->中文名映射
    url_to_title = []
    
    for url in urls:
        if url.startswith("https://zh.moegirl.org.cn/"):
            # 提取URL后面的部分
            encoded_part = url[len("https://zh.moegirl.org.cn/"):]
            
            # 解码URL编码
            title = urllib.parse.unquote(encoded_part)
            
            # 存储映射
            url_to_title.append(title)
        else:
            logger.warning(f"跳过非萌娘百科URL: {url}")
    
    logger.info(f"成功解析 {len(url_to_title)} 个URL")
    
    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(url_to_title, f, ensure_ascii=False, indent=2)
    
    logger.success(f"解析结果已保存至: {output_file_path}")
    return url_to_title

def main():
    # 设置文件路径
    input_file = Path(__file__).parents[1] / "data/wiki_urls_1.json"
    output_file = Path(__file__).parents[1] / "data/wiki_titles.json"
    
    # 解析URL
    url_to_title = parse_wiki_urls(input_file, output_file)
    
if __name__ == "__main__":
    main()
