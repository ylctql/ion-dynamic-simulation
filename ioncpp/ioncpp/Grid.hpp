#ifndef ION_SIMULATION_GRID_HPP
#define ION_SIMULATION_GRID_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "types.hpp"

namespace ioncpp
{


/**
 * @brief Calculate interpolated value for given grid data
 * 
 */
class Grid
{
private:
	std::vector<data_t> _grid_x;
	std::vector<data_t> _grid_y;
	std::vector<data_t> _grid_z;
	std::vector<data_t> _value;

	std::array<size_t, 3> _dim;

public:
	/**
	* @brief Description of coordinates relative to a certain grid
	* 
	* Applying on different grid structures will yield wrong result
	*/
	struct GridCoord
	{
		size_t x;
		size_t y;
		size_t z;

		data_t px;
		data_t py;
		data_t pz;
	};


	/**
	 * @brief Constructor
	 * 
	 * @param grid_x x coordinates of the grid. Sorted in ascending order.
	 * @param grid_y y coordinates of the grid. Sorted in ascending order.
	 * @param grid_z z coordinates of the grid. Sorted in ascending order.
	 * @param value values on the grid point. The order is given by v(x, y, z) = value[x * ly * lz + y * lz + z].
	 */
	Grid(
		std::vector<data_t> grid_x, std::vector<data_t> grid_y, std::vector<data_t> grid_z,
		std::vector<data_t> value
	);


	/**
	 * @brief Check if the points are inside the grid
	 * 
	 */
	bool in_bounds(data_t x, data_t y, data_t z) const;


	/**
	 * @brief Translate coordinates relative to grid.
	 * 
	 */
	GridCoord get_coord(data_t x, data_t y, data_t z) const;


	/**
	 * @brief Interpolate grid at given position. UB if position is outside of the grid.
	 * 
	 */
	data_t interpolate(data_t x, data_t y, data_t z) const;


	/**
	 * @brief Interpolate grid at given grid coordinate.
	 * 
	 */
	data_t interpolate(const GridCoord& coord) const;
};


}

#endif
