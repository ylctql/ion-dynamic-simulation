"""
collision_pressure — H2 elastic collision pressure estimation

Subcommands:
  python -m collision_pressure scan          Build configuration library
  python -m collision_pressure simulate      Run collision simulation

注：包初始化不导入子模块，避免 matplotlib 等副作用影响 main.py --plot。
请从具体子模块导入，例如 ``collision_pressure.species``、``collision_pressure.simulation``。
"""
