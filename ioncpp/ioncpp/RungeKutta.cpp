#include "RungeKutta.hpp"

#include <iostream>
#include <chrono>

namespace ioncpp
{

using namespace std;
using namespace std::literals;

namespace
{

// NOLINTNEXTLINE
chrono::microseconds elapsed1 = 0us;

VecType CoulombInteraction(
	CRef<VecType> r, 
	CRef<ArrayType>& charge
)
{
	auto begin = chrono::steady_clock::now();

	const auto N = r.rows();
	Eigen::Matrix<data_t, 1, Eigen::Dynamic> ones_col = Eigen::Matrix<data_t, 1, Eigen::Dynamic>::Ones(1, N);
	Eigen::Matrix<data_t, Eigen::Dynamic, 1> ones_row = Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Ones(N, 1);

	Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2 = 
		r.rowwise().squaredNorm().matrix() * ones_col
		+ ones_row * r.rowwise().squaredNorm().transpose().matrix()
		- 2 * (r.matrix() * r.matrix().transpose());
	dist2.diagonal() = ones_row;

	VecType result(N, 3);
	for (uint8_t i = 0; i < 3; i++)
	{
		result.col(i) = (
			(r.col(i).matrix() * ones_col - ones_row * r.col(i).transpose().matrix()).array()
			/ (dist2.array().sqrt() * dist2.array()) * (charge.matrix() * charge.transpose().matrix()).array()
		).rowwise().sum();
	}

	auto end = std::chrono::steady_clock::now();
	elapsed1 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin);

	return result;
}

}

std::pair<std::vector<VecType>, std::vector<VecType>> CalcTrajRK(
	CRef<VecType>& init_r,
	CRef<VecType>& init_v,
	CRef<ArrayType>& charge,
	CRef<ArrayType>& mass,
	size_t step,
	data_t time_start,
	data_t time_end,
	ForceCallback force
)
{
	auto begin = chrono::steady_clock::now();
	auto tmp = chrono::steady_clock::now();
	elapsed1 = 0us;
	chrono::microseconds elapsed2 = 0us;

	data_t dt = (time_end - time_start) / (data_t)step;

	VecType a(init_r.rows(), DIM);
	vector<VecType> r_ret, v_ret;
	r_ret.reserve(step);
	v_ret.reserve(step);

	VecType r = init_r;
	VecType v = init_v;

	VecType r_tmp(init_r.rows(), DIM);
	VecType v_tmp(init_r.rows(), DIM);

	VecType r_k1(init_r.rows(), DIM);
	VecType r_k2(init_r.rows(), DIM);
	VecType r_k3(init_r.rows(), DIM);
	VecType r_k4(init_r.rows(), DIM);

	VecType v_k1(init_r.rows(), DIM);
	VecType v_k2(init_r.rows(), DIM);
	VecType v_k3(init_r.rows(), DIM);
	VecType v_k4(init_r.rows(), DIM);

	for (size_t i = 0; i < step; i++)
	{
		double t = time_start + dt * (double)i;
		
		tmp = std::chrono::steady_clock::now();
		v_k1 = (force(r, v, t)+ CoulombInteraction(r, charge)).colwise() / mass * dt;
		elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tmp);
		r_k1 = v * dt;
		r_tmp = r + r_k1 / 2.0;
		v_tmp = v + v_k1 / 2.0;

		tmp = std::chrono::steady_clock::now();
		v_k2 = (force(r_tmp, v_tmp, t + dt / 2.0)+ CoulombInteraction(r_tmp, charge)).colwise() / mass * dt;
		elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tmp);
		r_k2 = v_tmp * dt;
		r_tmp = r + r_k2 / 2.0;
		v_tmp = v + v_k2 / 2.0;

		tmp = std::chrono::steady_clock::now();
		v_k3 = (force(r_tmp, v_tmp, t + dt / 2.0)+ CoulombInteraction(r_tmp, charge)).colwise() / mass * dt;
		elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tmp);
		r_k3 = v_tmp * dt;
		r_tmp = r + r_k3;
		v_tmp = v + v_k3;

		tmp = std::chrono::steady_clock::now();
		v_k4 = (force(r_tmp, v_tmp, t + dt)+ CoulombInteraction(r_tmp, charge)).colwise() / mass * dt;
		elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tmp);
		r_k4 = v_tmp * dt;
		v += v_k1 / 6.0 + v_k2 / 3.0 + v_k3 / 3.0 + v_k4 / 6.0;
		r += r_k1 / 6.0 + r_k2 / 3.0 + r_k3 / 3.0 + r_k4 / 6.0;

		r_ret.push_back(r);
		v_ret.push_back(v);
	}

	auto end = chrono::steady_clock::now();

	std::cout << "Time elapsed in CoulombInteraction: " << elapsed1.count() << "[us]" << std::endl;
	std::cout << "Time elapsed in force callback: " << elapsed2.count() - elapsed1.count() << "[us]" << std::endl;
	std::cout << "Time elapsed in CalcTrajRK: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

	return {r_ret, v_ret};
}


}