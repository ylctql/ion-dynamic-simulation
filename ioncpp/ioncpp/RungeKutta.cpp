#include "RungeKutta.hpp"

#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace ioncpp
{

using namespace std;
using namespace std::literals;

namespace
{

// NOLINTNEXTLINE
chrono::microseconds elapsed1 = 0us;

// 缓存可复用的矩阵（线程局部存储，避免多线程问题）
struct CoulombCache {
	Eigen::Matrix<data_t, 1, Eigen::Dynamic> ones_col;
	Eigen::Matrix<data_t, Eigen::Dynamic, 1> ones_row;
	Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> charge_matrix;
	size_t cached_N = 0;
	const data_t* cached_charge_data = nullptr;
	
	void resize(size_t N) {
		if (cached_N != N) {
			ones_col = Eigen::Matrix<data_t, 1, Eigen::Dynamic>::Ones(1, N);
			ones_row = Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Ones(N, 1);
			cached_N = N;
			cached_charge_data = nullptr; // 强制重新计算电荷矩阵
		}
	}
	
	void updateChargeMatrix(CRef<ArrayType>& charge) {
		// 如果 N 变化或 charge 数据指针变化，重新计算电荷矩阵
		// 使用数据指针而不是引用指针，因为 CRef 是 Eigen::Ref 类型
		const data_t* charge_data = charge.data();
		if (cached_charge_data != charge_data || charge_matrix.rows() != cached_N) {
			charge_matrix = (charge.matrix() * charge.transpose().matrix()).array();
			cached_charge_data = charge_data;
		}
	}
};

// 使用线程局部存储，确保线程安全
thread_local CoulombCache coulomb_cache;

VecType CoulombInteraction(
	CRef<VecType> r, 
	CRef<ArrayType>& charge
)
{
	auto begin = chrono::steady_clock::now();

	const auto N = r.rows();
	
	// 更新缓存（如果需要）
	coulomb_cache.resize(N);
	coulomb_cache.updateChargeMatrix(charge);
	
	auto& ones_col = coulomb_cache.ones_col;
	auto& ones_row = coulomb_cache.ones_row;
	auto& charge_matrix = coulomb_cache.charge_matrix;

	// Calculation of Coulomb interaction - 优化距离矩阵计算
	Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2 = 
		r.rowwise().squaredNorm().matrix() * ones_col
		+ ones_row * r.rowwise().squaredNorm().transpose().matrix()
		- 2 * (r.matrix() * r.matrix().transpose());
	dist2.diagonal() = ones_row;

	// 预计算 dist2 的平方根和立方（避免在循环中重复计算）
	Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2_array = dist2.array();
	Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2_sqrt = dist2_array.sqrt();
	Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> dist2_cubed = dist2_sqrt * dist2_array;

	VecType result(N, 3);
	for (uint8_t i = 0; i < 3; i++)
	{
		// 使用预计算的电荷矩阵和距离矩阵（关键优化：电荷矩阵不再在循环内计算）
		result.col(i) = (
			(r.col(i).matrix() * ones_col - ones_row * r.col(i).transpose().matrix()).array()
			/ dist2_cubed * charge_matrix
		).rowwise().sum();
	}

	// Finish Coulomb calculation

	auto end = std::chrono::steady_clock::now();
	elapsed1 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin);

	return result;
}

}

void saveArray(const VecType& arr, const std::string& filename)
	{
		std::ofstream ofs(filename, std::ios::binary);
		ofs.write(reinterpret_cast<const char*>(arr.data()), arr.size() * sizeof(data_t));
	}

void loadArray(VecType& arr, const std::string& filename)
	{
		std::ifstream ifs(filename, std::ios::binary);
		ifs.read(reinterpret_cast<char*>(arr.data()), arr.size() * sizeof(data_t));
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

	//RK4
	
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
	
	
	VecType a_last(init_r.rows(), DIM);

	for (size_t i = 0; i < step; i++)
	{
		double t = time_start + dt * (double)i;
		
		// RK4
		
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

		//End RK4
		

		// Velocity Verlet
		/*
		tmp = std::chrono::steady_clock::now();
		std::filesystem::path a_file = "../data_cache/a.bin";
		if (!std::filesystem::exists(a_file))
		{
			a = (force(r, v, t) + CoulombInteraction(r, charge)).colwise() / mass;
			saveArray(a, a_file.string());
			r += v * dt + a * (dt * dt / 2.0);
			v += a * dt;
		}
		else{
			loadArray(a_last, a_file.string());
			r += v * dt + a_last * (dt * dt / 2.0);
			a = (force(r, v, t + dt) + CoulombInteraction(r, charge)).colwise() / mass;
			saveArray(a, a_file.string());
			v += (a + a_last) * (dt / 2.0);
		}
		elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tmp);
		*/
		//End Velocity Verlet
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