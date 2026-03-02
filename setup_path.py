"""
统一设置 sys.path，确保 ionsim 可被导入。
须在导入 ionsim 或 backend 之前调用。无其他项目依赖，可安全作为首个导入。
"""
import sys
from pathlib import Path


def ensure_build_in_path(root: Path | None = None) -> Path:
    """
    将项目根目录及 build（若含 ionsim*.so）加入 sys.path。

    Parameters
    ----------
    root : Path | None
        项目根目录；None 时使用本文件所在目录

    Returns
    -------
    Path
        项目根目录
    """
    if root is None:
        root = Path(__file__).resolve().parent
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    build_dir = (root / "build").resolve()
    if build_dir.exists() and build_dir.is_dir():
        so_files = list(build_dir.glob("ionsim*.so"))
        if so_files and str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
    return root
