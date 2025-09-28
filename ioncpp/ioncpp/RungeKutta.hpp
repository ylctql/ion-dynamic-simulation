#ifndef ION_SIMULATION_RUNGE_KUTTA_HPP
#define ION_SIMULATION_RUNGE_KUTTA_HPP

#include <Eigen/Dense>
#include <vector>

#include "types.hpp"

namespace ioncpp
{

void saveArray(const VecType& arr, const std::string& filename);
void loadArray(VecType& arr, const std::string& filename);
void savetxt(const VecType& arr, const std::string& filename);


using ForceCallback = std::function<VecType(CRef<VecType>& r, CRef<VecType>& v, data_t t)>;

/**
 * @brief Calculate ion trajectory using 4th order Runge-Kutta
 * 
 * @param init_r the initial position of ions
 * @param init_v the initial velocity of ions
 * @param charge the charge of ions
 * @param mass the mass of ions
 * @param step iteration step
 * @param time_start start time
 * @param time_end end time
 * @param force dynamic function for external force based on input position, velocity, time, excluding
 * the Coulomb interaction between ions which has already taken into account
 * @return std::pair<std::vector<ArrayType>, std::vector<ArrayType>>
 */
std::pair<std::vector<VecType>, std::vector<VecType>> CalcTrajRK(
	CRef<VecType>& init_r,
	CRef<VecType>& init_v,
	CRef<ArrayType>& charge,
	CRef<ArrayType>& mass,
	size_t step,
	data_t time_start,
	data_t time_end,
	ForceCallback force
);

}

#endif