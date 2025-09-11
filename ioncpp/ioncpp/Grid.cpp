#include "Grid.hpp"

#include <algorithm>
#include <iterator>
#include <vector>

namespace ioncpp
{


Grid::Grid(
	std::vector<data_t> grid_x, std::vector<data_t> grid_y, std::vector<data_t> grid_z,
	std::vector<data_t> value
): 
_grid_x(std::move(grid_x)), 
_grid_y(std::move(grid_y)), 
_grid_z(std::move(grid_z)), 
_value(std::move(value)),
_dim({_grid_x.size(), _grid_y.size(), _grid_z.size()})
{
}


bool Grid::in_bounds(data_t x, data_t y, data_t z) const
{
	return (x >= this->_grid_x.front()) && (x <= this->_grid_x.back())
	&& (y >= this->_grid_y.front()) && (y <= this->_grid_y.back())
	&& (z >= this->_grid_z.front()) && (z <= this->_grid_z.back());
}


Grid::GridCoord Grid::get_coord(data_t x, data_t y, data_t z) const
{
	auto find_pos = [](const std::vector<data_t>& arr, data_t i)
	{
		return std::distance(arr.begin(), std::lower_bound(arr.begin(), arr.end(), i));
	};

	size_t xidx = find_pos(this->_grid_x, x);
	size_t yidx = find_pos(this->_grid_y, y);
	size_t zidx = find_pos(this->_grid_z, z);
	return GridCoord{
		xidx, yidx, zidx,
		(x - this->_grid_x[xidx - 1]) / (this->_grid_x[xidx] - this->_grid_x[xidx - 1]),
		(y - this->_grid_y[yidx - 1]) / (this->_grid_y[yidx] - this->_grid_y[yidx - 1]),
		(z - this->_grid_z[zidx - 1]) / (this->_grid_z[zidx] - this->_grid_z[zidx - 1])
	};
}


data_t Grid::interpolate(data_t x, data_t y, data_t z) const
{
	return this->interpolate(this->get_coord(x, y, z));
}


data_t Grid::interpolate(const GridCoord& coord) const
{
	auto stridey = this->_dim[2];
	auto stridex = this->_dim[1] * stridey;
	auto pos = (coord.x - 1) * stridex + (coord.y - 1) * stridey + (coord.z - 1);
	return this->_value[pos] * (1 - coord.px) * (1 - coord.py) * (1 - coord.pz)
	+ this->_value[pos + stridex] * (coord.px) * (1 - coord.py) * (1 - coord.pz)
	+ this->_value[pos + stridey] * (1 - coord.px) * (coord.py) * (1 - coord.pz)
	+ this->_value[pos + 1] * (1 - coord.px) * (1 - coord.py) * (coord.pz)
	+ this->_value[pos + stridex + stridey] * (coord.px) * (coord.py) * (1 - coord.pz)
	+ this->_value[pos + stridex + 1] * (coord.px) * (1 - coord.py) * (coord.pz)
	+ this->_value[pos + stridey + 1] * (1 - coord.px) * (coord.py) * (coord.pz)
	+ this->_value[pos + stridex + stridey + 1] * (coord.px) * (coord.py) * (coord.pz);
}


}