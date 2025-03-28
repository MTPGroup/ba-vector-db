import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from query_vector_db import generate_embedding, query_vector_db
from utils import setup_logger

# 配置日志
setup_logger()

# 创建FastAPI应用
app = FastAPI(
    title='蔚蓝档案向量数据库检索API',
    description='使用向量数据库进行语义化查询蔚蓝档案知识库',
    version='0.1.0',
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # 允许所有来源，生产环境中应该设置为特定域名
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class SearchResult(BaseModel):
    """搜索结果模型"""

    distance: float = Field(..., description='相似度分数')
    title: str = Field(..., description='文档标题')
    category: str = Field(..., description='文档分类')
    content: str = Field(..., description='文档内容')


class SearchResponse(BaseModel):
    """搜索响应模型"""

    query: str = Field(..., description='查询文本')
    results: list[SearchResult] = Field(
        default_factory=list, description='搜索结果列表'
    )
    total: int = Field(..., description='结果总数')


@app.get('/', include_in_schema=False)
async def root():
    """API根路径响应"""
    return {'message': '欢迎使用蔚蓝档案向量数据库检索API'}


@app.get('/api/search', response_model=SearchResponse)
async def search(
    q: str = Query(..., description='查询文本'),
    limit: int = Query(5, description='返回结果数量', ge=1, le=50),
):
    """
    使用向量数据库进行语义化搜索

    - **q**: 查询文本
    - **limit**: 返回结果数量(1-50)

    返回与查询语义最相关的文档列表
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail='查询文本不能为空')

    try:
        # 执行向量数据库查询
        results = await query_vector_db(q, limit)

        if not results:
            logger.info(f"未找到与查询相关的结果: '{q}'")
            return SearchResponse(query=q, results=[], total=0)

        # 格式化响应
        search_results = [
            SearchResult(
                distance=result['distance'],
                title=result['title'],
                category=result['category'],
                content=result['content'],
            )
            for result in results
        ]

        logger.info(f"查询 '{q}' 返回了 {len(search_results)} 个结果")
        return SearchResponse(
            query=q, results=search_results, total=len(search_results)
        )

    except Exception as e:
        logger.error(f'搜索查询过程中出错: {str(e)}')
        raise HTTPException(status_code=500, detail=f'搜索过程中发生错误: {str(e)}')


@app.get('/api/healthcheck')
async def health_check():
    """健康检查接口"""
    try:
        # 检查DashScope API连接
        embedding = await generate_embedding('测试连接')

        if embedding is None:
            return {
                'status': 'warning',
                'message': 'DashScope API连接异常',
                'services': {'api': 'ok', 'dashscope': 'error'},
            }

        # 尝试执行简单查询来检查Milvus连接
        await query_vector_db('测试', 2)

        return {
            'status': 'ok',
            'message': '所有服务运行正常',
            'services': {'api': 'ok', 'dashscope': 'ok', 'milvus': 'ok'},
        }
    except Exception as e:
        logger.error(f'健康检查失败: {e}')
        return {
            'status': 'error',
            'message': f'健康检查失败: {str(e)}',
            'services': {'api': 'ok', 'dashscope': 'unknown', 'milvus': 'unknown'},
        }


def start():
    """启动FastAPI服务器"""
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)


if __name__ == '__main__':
    start()
