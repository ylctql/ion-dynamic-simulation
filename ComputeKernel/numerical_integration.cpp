/**
 * 数值积分：RK4 与 Velocity Verlet
 * 参考 outline.md 与 ism-hybrid/ism-cuda RungeKutta.cpp
 */
#include "numerical_integration.hpp"

#include "Coulomb.hpp"

#include <stdexcept>

namespace ioncpp {

namespace {

TensorType acceleration(CRef<TensorType>& r, CRef<TensorType>& v,
                        CRef<VectorType>& charge, CRef<VectorType>& mass,
                        data_t t, ForceCallback& force, bool use_cuda) {
#ifdef IONCPP_SKIP_COULOMB
    TensorType F_total = force(r, v, t);
#else
    TensorType F_total = force(r, v, t) + CoulombInteraction(r, charge, use_cuda);
#endif
    return F_total.colwise() / mass;
}

}  // namespace

std::pair<std::vector<TensorType>, std::vector<TensorType>> CalcTrajRK(
    CRef<TensorType>& init_r, CRef<TensorType>& init_v,
    CRef<VectorType>& charge, CRef<VectorType>& mass, size_t step,
    data_t time_start, data_t time_end, ForceCallback force, bool use_cuda) {

    const data_t dt = (time_end - time_start) / static_cast<data_t>(step);
    const size_t N = init_r.rows();

    TensorType r = init_r;
    TensorType v = init_v;

    TensorType r_tmp(N, DIM);
    TensorType v_tmp(N, DIM);
    TensorType r_k1(N, DIM), r_k2(N, DIM), r_k3(N, DIM), r_k4(N, DIM);
    TensorType v_k1(N, DIM), v_k2(N, DIM), v_k3(N, DIM), v_k4(N, DIM);

    std::vector<TensorType> r_ret, v_ret;
    r_ret.reserve(step);
    v_ret.reserve(step);

    for (size_t i = 0; i < step; ++i) {
        const data_t t = time_start + dt * static_cast<data_t>(i);

        v_k1 = acceleration(r, v, charge, mass, t, force, use_cuda) * dt;
        r_k1 = v * dt;

        r_tmp = r + r_k1 * 0.5;
        v_tmp = v + v_k1 * 0.5;
        v_k2 = acceleration(r_tmp, v_tmp, charge, mass, t + dt * 0.5, force,
                            use_cuda) *
               dt;
        r_k2 = v_tmp * dt;

        r_tmp = r + r_k2 * 0.5;
        v_tmp = v + v_k2 * 0.5;
        v_k3 = acceleration(r_tmp, v_tmp, charge, mass, t + dt * 0.5, force,
                            use_cuda) *
               dt;
        r_k3 = v_tmp * dt;

        r_tmp = r + r_k3;
        v_tmp = v + v_k3;
        v_k4 = acceleration(r_tmp, v_tmp, charge, mass, t + dt, force,
                            use_cuda) *
               dt;
        r_k4 = v_tmp * dt;

        v += (v_k1 + 2 * v_k2 + 2 * v_k3 + v_k4) / 6.0;
        r += (r_k1 + 2 * r_k2 + 2 * r_k3 + r_k4) / 6.0;

        r_ret.push_back(r);
        v_ret.push_back(v);
    }

    return {r_ret, v_ret};
}

std::pair<std::vector<TensorType>, std::vector<TensorType>> CalcTrajVV(
    CRef<TensorType>& init_r, CRef<TensorType>& init_v,
    CRef<VectorType>& charge, CRef<VectorType>& mass, size_t step,
    data_t time_start, data_t time_end, ForceCallback force, bool use_cuda) {

    const data_t dt = (time_end - time_start) / static_cast<data_t>(step);
    const size_t N = init_r.rows();

    TensorType r = init_r;
    TensorType v = init_v;
    TensorType a(N, DIM);
    TensorType a_last(N, DIM);
    bool a_last_initialized = false;

    std::vector<TensorType> r_ret, v_ret;
    r_ret.reserve(step);
    v_ret.reserve(step);

    for (size_t i = 0; i < step; ++i) {
        const data_t t = time_start + dt * static_cast<data_t>(i);

        if (!a_last_initialized) {
            a = acceleration(r, v, charge, mass, t, force, use_cuda);
            a_last = a;
            a_last_initialized = true;

            r += v * dt + a * (dt * dt * 0.5);
            v += a * dt;
        } else {
            r += v * dt + a_last * (dt * dt * 0.5);
            a = acceleration(r, v, charge, mass, t + dt, force, use_cuda);
            v += (a + a_last) * (dt * 0.5);
            a_last = a;
        }

        r_ret.push_back(r);
        v_ret.push_back(v);
    }

    return {r_ret, v_ret};
}

std::pair<std::vector<TensorType>, std::vector<TensorType>> CalcTraj(
    CRef<TensorType>& init_r, CRef<TensorType>& init_v,
    CRef<VectorType>& charge, CRef<VectorType>& mass, size_t step,
    data_t time_start, data_t time_end, ForceCallback force, bool use_cuda,
    const std::string& calc_method) {

    if (calc_method == "RK4") {
        return CalcTrajRK(init_r, init_v, charge, mass, step, time_start,
                         time_end, force, use_cuda);
    }
    if (calc_method == "VV") {
        return CalcTrajVV(init_r, init_v, charge, mass, step, time_start,
                         time_end, force, use_cuda);
    }
    throw std::invalid_argument("calc_method 须为 \"RK4\" 或 \"VV\"");
}

}  // namespace ioncpp
