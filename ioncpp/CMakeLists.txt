target_sources(
	ioncpp PRIVATE

	ioncpp/RungeKutta.hpp
	ioncpp/RungeKutta.cpp
	ioncpp/Grid.hpp
	ioncpp/Grid.cpp
	ioncpp/types.hpp
)

target_include_directories(ioncpp PUBLIC .)