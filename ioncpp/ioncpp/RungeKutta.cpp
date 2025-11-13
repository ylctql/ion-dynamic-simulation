#include "RungeKutta.hpp"

#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cuda_runtime.h>

using data_t = double; 
constexpr int DIM = 3;
constexpr int CPU = 0;
constexpr int CUDA = 1;

extern "C" void computeCoulombInteraction(
    const data_t* r, const data_t* charge, data_t* result, int N, int grid_size, int block_size);

namespace ioncpp
{

using namespace std;
using namespace std::literals;

namespace
{

chrono::microseconds elapsed1 = 0us;

VecType CoulombInteractionCuda(
    CRef<VecType> r, 
    CRef<ArrayType>& charge
) {
    auto begin = chrono::steady_clock::now();

    const auto N = r.rows();
    VecType result(N, DIM);
    result.setZero();

    data_t* d_r;
    data_t* d_charge;
    data_t* d_result;
    cudaMalloc(&d_r, N * DIM * sizeof(data_t));
    cudaMalloc(&d_charge, N * sizeof(data_t));
    cudaMalloc(&d_result, N * DIM * sizeof(data_t));

    cudaMemcpy(d_r, r.data(), N * DIM * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_charge, charge.data(), N * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, N * DIM * sizeof(data_t));

    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeCoulombInteraction(d_r, d_charge, d_result, N, blocksPerGrid, threadsPerBlock);

    cudaMemcpy(result.data(), d_result, N * DIM * sizeof(data_t), cudaMemcpyDeviceToHost);

    cudaFree(d_r);

    cudaFree(d_charge);
    cudaFree(d_result);

    auto end = std::chrono::steady_clock::now();
    elapsed1 += std::chrono::duration_cast<std::chrono::microseconds>(end - begin);

    return result;
}

VecType CoulombInteractionCpu(
	CRef<VecType> r, 
	CRef<ArrayType>& charge
)
{
	auto begin = chrono::steady_clock::now();

	const auto N = r.rows();
	Eigen::Matrix<data_t, 1, Eigen::Dynamic> ones_col = Eigen::Matrix<data_t, 1, Eigen::Dynamic>::Ones(1, N);
	Eigen::Matrix<data_t, Eigen::Dynamic, 1> ones_row = Eigen::Matrix<data_t, Eigen::Dynamic, 1>::Ones(N, 1);

	// Calculation of Coulomb interaction
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
    int device,
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
    // printf("Using %s for computation.\n", device == CUDA ? "CUDA" : "CPU");
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
	/*
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
	*/
	
	VecType a_last(init_r.rows(), DIM);

	for (size_t i = 0; i < step; i++)
	{
		double t = time_start + dt * (double)i;
		
		// RK4
		/*
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
		*/

		// Velocity Verlet
		tmp = std::chrono::steady_clock::now();
		// 优化 Velocity Verlet
        static VecType a_last(init_r.rows(), DIM); // 使用静态变量缓存 a_last，避免频繁分配内存
        static bool a_last_initialized = false;   // 标记是否已初始化 a_last

        if (!a_last_initialized) {
            if (device == CUDA) {
                a = (force(r, v, t) + CoulombInteractionCuda(r, charge)).colwise() / mass;
            } else {
                a = (force(r, v, t) + CoulombInteractionCpu(r, charge)).colwise() / mass;
            }
            a_last = a; 
            a_last_initialized = true;

            r += v * dt + a * (dt * dt / 2.0);
            v += a * dt;
        } else {
            r += v * dt + a_last * (dt * dt / 2.0);
            if (device == CUDA) {
                a = (force(r, v, t) + CoulombInteractionCuda(r, charge)).colwise() / mass;
            } else {
                a = (force(r, v, t) + CoulombInteractionCpu(r, charge)).colwise() / mass;
            }
            v += (a + a_last) * (dt / 2.0);
            a_last = a;
        }

        elapsed2 += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tmp);

		r_ret.push_back(r);
		v_ret.push_back(v);
	}

	auto end = chrono::steady_clock::now();

	// std::cout << "Time elapsed in CoulombInteraction: " << elapsed1.count() << "[us]" << std::endl;
	// std::cout << "Time elapsed in force callback: " << elapsed2.count() - elapsed1.count() << "[us]" << std::endl;
	// std::cout << "Time elapsed in CalcTrajRK: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;

	return {r_ret, v_ret};
}


}