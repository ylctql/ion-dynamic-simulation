"""
电极电压配置模块
从 JSON 文件加载电压配置，构建 voltage_list 供 FieldSettings 使用。
无量纲化常数由 init_from_config 返回的 Config 对象提供。
注意：__init__ 仅导出 constants，避免导入 loader 时产生循环依赖。
"""
from .constants import Config, init_from_config

__all__ = ["Config", "init_from_config"]
