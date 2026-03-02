"""
pytest 配置与公共 fixture
"""
import logging

# 项目根由 pyproject.toml [tool.pytest.ini_options] pythonpath = ["."] 提供

# 测试时配置 logging，避免未配置时无输出
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
