import sys
from loguru import logger


def setup_logger():
    """配置日志记录器"""
    logger.remove()  # 移除默认处理器
    logger.add(sys.stderr, level="INFO")  # 添加标准错误输出处理器
    logger.add("query_logs.log", rotation="10 MB")  # 添加文件处理器
