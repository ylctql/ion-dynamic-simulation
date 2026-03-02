# ioncpp 核心库源文件（由根 CMake 定义 ioncpp 目标）
target_sources(ioncpp PRIVATE
  Coulomb.cpp
  Coulomb.hpp
  numerical_integration.cpp
  numerical_integration.hpp
  types.hpp
)

target_include_directories(ioncpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA 源文件（可选）
if(USE_CUDA)
  target_sources(ioncpp PRIVATE Coulomb_cuda.cu)
  target_link_libraries(ioncpp PRIVATE CUDA::cudart)
endif()
