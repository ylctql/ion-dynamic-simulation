#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <ioncpp/RungeKutta.hpp>
#include <ioncpp/Grid.hpp>
#include <ioncpp/types.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace ioncpp;
using namespace pybind11::literals;
using namespace std::string_literals;

// Wrapper around the original function, mainly dealing with storage orders
auto CalcTrajRK_wrapper(
    int device,
	const pybind11::EigenDRef<const VecType>& init_r,
	const pybind11::EigenDRef<const VecType>& init_v,
	CRef<ArrayType>& charge,
	CRef<ArrayType>& mass,
	size_t step,
	data_t time_start,
	data_t time_end,
	pybind11::function force
)
{
	// Validate input
	if (time_start > time_end)
		throw std::invalid_argument(
			"time_end "s + std::to_string(time_end) 
			+ " must be greater than time_start "s + std::to_string(time_start)
		);
	
	auto len_r = init_r.rows();
	if (init_v.rows() != len_r)
		throw std::invalid_argument(
			"init_v have different length "s + std::to_string(init_v.rows()) 
			+ " compared to init_r "s + std::to_string(len_r)
		);
	if (charge.rows() != len_r)
		throw std::invalid_argument(
			"charge have different length "s + std::to_string(init_v.rows()) 
			+ " compared to init_r "s + std::to_string(len_r)
		);
	if (mass.rows() != len_r)
		throw std::invalid_argument(
			"charge have different length "s + std::to_string(init_v.rows()) 
			+ " compared to init_r "s + std::to_string(len_r)
		);

	return CalcTrajRK(
        device,
		init_r, 
		init_v, 
		charge, 
		mass, 
		step, 
		time_start, 
		time_end, 
		[force=std::move(force)](CRef<VecType>& r, CRef<VecType>& v, data_t t) -> VecType
		{
			pybind11::buffer result = force(r, v, t);
			using Strides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> ;

			/* Request a buffer descriptor from Python */
			pybind11::buffer_info info = result.request();

			/* Some basic validation checks ... */
			if (info.format != pybind11::format_descriptor<VecType::Scalar>::format())
				throw std::runtime_error("Incompatible format: expected a double array!");

			if (info.ndim != 2)
				throw std::runtime_error("Incompatible buffer dimension!");

			auto strides = Strides(
				info.strides[1] / (pybind11::ssize_t)sizeof(VecType::Scalar),
				info.strides[0] / (pybind11::ssize_t)sizeof(VecType::Scalar)
			);

			auto map = Eigen::Map<VecType, 0, Strides>(
			static_cast<VecType::Scalar *>(info.ptr), info.shape[0], info.shape[1], strides);

			return VecType(map);
		}
	);
}


template<typename Callable>
auto Vectorize_3D(pybind11::array_t<data_t>& xi, Callable&& f)
{
	using Ret = std::result_of_t<Callable(data_t, data_t, data_t)>;
	auto ndim = xi.ndim();
	const auto* shape = xi.shape();
	if (shape[ndim - 1] != 3)
	{
		std::string shape = "(";
		for(long long i = 0; i < ndim; i++) shape += std::to_string(xi.shape(i)) + ",";
		shape += ")";
		throw std::invalid_argument("expect input to have shape (..., 3), but is "s + shape + " instead"s);
	}

	pybind11::array_t<data_t> xi_list = xi.reshape({-1, 3});
	auto data = xi_list.unchecked<2>();
	auto len = data.shape(0);
	auto result = pybind11::array_t<Ret>(len);
	auto result_ptr = static_cast<Ret *>(result.request().ptr);

	for (long long i = 0; i < len; i++)
		result_ptr[i] = std::forward<Callable>(f)(data(i, 0), data(i, 1), data(i, 2));

	return result.reshape(std::vector<std::ptrdiff_t>(shape, shape + ndim - 1));

}


// NOLINTNEXTLINE
PYBIND11_MODULE(ionsim, m) 
{
	#ifdef VERSION_INFO
		m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
	#else
		m.attr("__version__") = "dev";
	#endif


	m.def(
		"calculate_trajectory", 
		&CalcTrajRK_wrapper,
        "device"_a,
		"init_r"_a.noconvert(),
		"init_v"_a.noconvert(),
		"charge"_a.noconvert(),
		"mass"_a.noconvert(),
		"step"_a,
		"time_start"_a,
		"time_end"_a,
		"force"_a,
		pybind11::return_value_policy::move
	);

	PYBIND11_NUMPY_DTYPE(Grid::GridCoord, x, y, z, px, py, pz);
	auto m_grid = pybind11::class_<Grid>(m, "Grid");
	auto m_gridcoord = pybind11::class_<Grid::GridCoord>(m_grid, "GridCoord");


	m_grid
	.def(
		pybind11::init([](
			pybind11::array_t<data_t>& x, 
			pybind11::array_t<data_t>& y, 
			pybind11::array_t<data_t>& z, 
			pybind11::array_t<data_t>& value
		){
			pybind11::array_t<data_t> x_view = x.reshape({-1});
			pybind11::array_t<data_t> y_view = y.reshape({-1});
			pybind11::array_t<data_t> z_view = z.reshape({-1});
			pybind11::array_t<data_t> value_view = value.reshape({-1});


			if (value_view.shape(0) != x_view.shape(0) * y_view.shape(0) * z_view.shape(0))
				throw std::invalid_argument(
					"dismatched dimension between value ("s
					+ std::to_string(value_view.shape(0)) + ") and grid ("s
					+ std::to_string(x_view.shape(0)) + "x"s
					+ std::to_string(y_view.shape(0)) + "x"s
					+ std::to_string(z_view.shape(0)) + ")"s
				);

			auto vec_x = x_view.cast<std::vector<data_t>>();
			auto vec_y = y_view.cast<std::vector<data_t>>();
			auto vec_z = z_view.cast<std::vector<data_t>>();
			auto vec_value = value_view.cast<std::vector<data_t>>();

			if (
				!std::is_sorted(vec_x.begin(), vec_x.end()) 
				|| !std::is_sorted(vec_y.begin(), vec_y.end())
				|| !std::is_sorted(vec_z.begin(), vec_z.end())
			)
				throw std::invalid_argument("grid coordinates not sorted in ascending order");
			return std::make_unique<Grid>(std::move(vec_x), std::move(vec_y), std::move(vec_z), std::move(vec_value));
		}),
		"x"_a.noconvert(), 
		"y"_a.noconvert(), 
		"z"_a.noconvert(), 
		"value"_a.noconvert()
	)

	.def("in_bounds", [](const Grid& g, pybind11::array_t<data_t>& xi) 
	{
		return Vectorize_3D(xi, [&g](data_t x, data_t y, data_t z) { return g.in_bounds(x, y, z); });
	},
	"xi"_a.noconvert(),
	pybind11::return_value_policy::move)

	.def("in_bounds", &Grid::in_bounds, "x"_a, "y"_a, "z"_a)

	.def("get_coord", [](const Grid& g, pybind11::array_t<data_t>& xi) 
	{
		return Vectorize_3D(xi, [&g](data_t x, data_t y, data_t z) { return g.get_coord(x, y, z); });
	},
	"xi"_a.noconvert(),
	pybind11::return_value_policy::move)

	.def("get_coord", &Grid::get_coord, "x"_a, "y"_a, "z"_a)

	.def("interpolate", [](const Grid& g, pybind11::array_t<data_t>& xi) 
	{
		return Vectorize_3D(xi, [&g](data_t x, data_t y, data_t z) { return g.interpolate(x, y, z); });
	},
	"xi"_a.noconvert(),
	pybind11::return_value_policy::move)

	.def("interpolate", pybind11::overload_cast<data_t, data_t, data_t>(&Grid::interpolate, pybind11::const_), "x"_a, "y"_a, "z"_a)

	.def("interpolate", [](const Grid& g, pybind11::array_t<Grid::GridCoord>& xi) 
	{
		auto ndim = xi.ndim();
		const auto* shape = xi.shape();

		pybind11::array_t<Grid::GridCoord> xi_list = xi.reshape({-1});
		auto data = xi_list.unchecked<1>();
		auto len = data.shape(0);
		auto result = pybind11::array_t<data_t>(len);
		auto result_ptr = static_cast<data_t *>(result.request().ptr);

		for (long long i = 0; i < len; i++)
			result_ptr[i] = g.interpolate(data(i));

		return result.reshape(std::vector<std::ptrdiff_t>(shape, shape + ndim));
	},
	"xi"_a.noconvert(),
	pybind11::return_value_policy::move)

	.def("interpolate", pybind11::overload_cast<const Grid::GridCoord&>(&Grid::interpolate, pybind11::const_), "coord"_a);



	m_gridcoord
	.def(pybind11::init<size_t, size_t, size_t, data_t, data_t, data_t>(),
	"x"_a = 0, "y"_a = 0, "z"_a = 0, "px"_a = 0.0, "py"_a = 0.0, "pz"_a = 0.0)

	.def_readwrite("x", &Grid::GridCoord::x)
	.def_readwrite("y", &Grid::GridCoord::y)
	.def_readwrite("z", &Grid::GridCoord::z)
	.def_readwrite("px", &Grid::GridCoord::px)
	.def_readwrite("py", &Grid::GridCoord::py)
	.def_readwrite("pz", &Grid::GridCoord::pz)

	.def_static("from_numpy", [](pybind11::array_t<Grid::GridCoord>& a)
	{
		pybind11::array_t<Grid::GridCoord> view = a.reshape({-1});
		auto data = view.unchecked<1>();
		auto len = data.shape(0);
		pybind11::list list(len);

		for (long long i = 0; i < len; i++)
			list[i] = data(i);
		return list;
	},
	"a"_a.noconvert(),
	pybind11::return_value_policy::move)

	.def_static("to_numpy", [](pybind11::list& a)
	{
		auto len = a.size();
		auto result = pybind11::array_t<Grid::GridCoord>(pybind11::ssize_t(len));
		auto result_ptr = static_cast<Grid::GridCoord *>(result.request().ptr);
		for (auto it = a.begin(); it != a.end(); it++, result_ptr++)
		{
			*result_ptr = it->cast<Grid::GridCoord>();
		}
		return result;
	},
	"a"_a.noconvert(),
	pybind11::return_value_policy::move);
}