target_sources(
	ioncpp PRIVATE

	ioncpp/RungeKutta.hpp
	ioncpp/RungeKutta.cpp
	ioncpp/Grid.hpp
	ioncpp/Grid.cpp
	ioncpp/types.hpp
    ioncpp/CoulombInteraction.cu
)

target_include_directories(ioncpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})