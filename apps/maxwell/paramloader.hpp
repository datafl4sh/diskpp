/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *  
 * Matteo Cicuttin (C) 2021
 * matteo.cicuttin@uliege.be
 *
 * University of Liège - Montefiore Institute
 * Applied and Computational Electromagnetics group  
 */

#include "sol/sol.hpp"

namespace priv {

template<typename T>
struct EH_complex_field
{
    T   Ex_re;
    T   Ex_im;
    T   Ey_re;
    T   Ey_im;
    T   Ez_re;
    T   Ez_im;
    T   Hx_re;
    T   Hx_im;
    T   Hy_re;
    T   Hy_im;
    T   Hz_re;
    T   Hz_im;

    EH_complex_field()
        : Ex_re(0), Ex_im(0), Ey_re(0), Ey_im(0), Ez_re(0), Ez_im(0),
          Hx_re(0), Hx_im(0), Hy_re(0), Hy_im(0), Hz_re(0), Hz_im(0)
    {}
};

template<typename T>
struct E_complex_field
{
    T   Ex_re;
    T   Ex_im;
    T   Ey_re;
    T   Ey_im;
    T   Ez_re;
    T   Ez_im;

    E_complex_field()
        : Ex_re(0), Ex_im(0), Ey_re(0), Ey_im(0), Ez_re(0), Ez_im(0)
    {}
};

template<typename T>
void
register_EH_complex_field_ut(sol::state& lua)
{
    sol::usertype<EH_complex_field<T>> EH_complex_field_type =
        lua.new_usertype<EH_complex_field<T>>("EH_complex_field",
            sol::constructors<EH_complex_field<T>()>()
        );

    EH_complex_field_type["Ex_re"] = &EH_complex_field<T>::Ex_re;
    EH_complex_field_type["Ex_im"] = &EH_complex_field<T>::Ex_im;
    EH_complex_field_type["Ey_re"] = &EH_complex_field<T>::Ey_re;
    EH_complex_field_type["Ey_im"] = &EH_complex_field<T>::Ey_im;
    EH_complex_field_type["Ez_re"] = &EH_complex_field<T>::Ez_re;
    EH_complex_field_type["Ez_im"] = &EH_complex_field<T>::Ez_im;
    EH_complex_field_type["Hx_re"] = &EH_complex_field<T>::Hx_re;
    EH_complex_field_type["Hx_im"] = &EH_complex_field<T>::Hx_im;
    EH_complex_field_type["Hy_re"] = &EH_complex_field<T>::Hy_re;
    EH_complex_field_type["Hy_im"] = &EH_complex_field<T>::Hy_im;
    EH_complex_field_type["Hz_re"] = &EH_complex_field<T>::Hz_re;
    EH_complex_field_type["Hz_im"] = &EH_complex_field<T>::Hz_im;
}

template<typename T>
void
register_E_complex_field_ut(sol::state& lua)
{
    sol::usertype<EH_complex_field<T>> EH_complex_field_type =
        lua.new_usertype<EH_complex_field<T>>("E_complex_field",
            sol::constructors<EH_complex_field<T>()>()
        );

    EH_complex_field_type["Ex_re"] = &EH_complex_field<T>::Ex_re;
    EH_complex_field_type["Ex_im"] = &EH_complex_field<T>::Ex_im;
    EH_complex_field_type["Ey_re"] = &EH_complex_field<T>::Ey_re;
    EH_complex_field_type["Ey_im"] = &EH_complex_field<T>::Ey_im;
    EH_complex_field_type["Ez_re"] = &EH_complex_field<T>::Ez_re;
    EH_complex_field_type["Ez_im"] = &EH_complex_field<T>::Ez_im;
}

} // namespace priv

template<typename T>
class parameter_loader
{
    using point_type = disk::point<T,3>;
    using complex_type = std::complex<T>;
    using real_type = T;

    sol::state lua;

public:
    static constexpr double eps0 = 8.8541878128e-12;
    static constexpr double mu0 = 4e-7*M_PI;

    parameter_loader()
    {
        lua.open_libraries(sol::lib::base, sol::lib::math);
        lua["eps0"] = eps0;
        lua["mu0"] = mu0;
        lua["boundary"] = lua.create_table();

        priv::register_E_complex_field_ut<T>(lua);
    }

    bool load(const std::string& fn)
    {
        bool success = true;
        lua.script_file(fn);
        
        if ( not lua["epsilon"].valid() )
        {
            std::cout << "CONFIG PROBLEM: Function epsilon() not defined" << std::endl;
            success = false;
        }

        if ( not lua["mu"].valid() )
        {
            std::cout << "CONFIG PROBLEM: Function mu() not defined" << std::endl;
            success = false;
        }

        if ( not lua["frequency"].valid() )
        {
            std::cout << "CONFIG PROBLEM: Parameter frequency not defined" << std::endl;
            success = false;
        }

        return success;
    }

    complex_type epsilon(size_t tag)
    {
        double re, im;
        sol::tie(re, im) = lua["epsilon"](tag);
        return std::complex<double>(re, im);
    }

    complex_type mu(size_t tag)
    {
        double re, im;
        sol::tie(re, im) = lua["mu"](tag);
        return std::complex<double>(re, im);
    }

    complex_type volume_source(size_t tag, const point_type& pt)
    {
        return complex_type(0.0);
    }

    real_type frequency()
    {
        return lua["frequency"];
    }

    priv::E_complex_field<real_type>
    plane_wave_source(size_t tag, const point_type& pt)
    {
        auto bnd = lua["boundary"][tag];
        if (not bnd.valid())
        {
            std::cout << "Can't access data for boundary " << tag << std::endl;
            throw std::invalid_argument("No boundary data");
        }

        auto field = bnd["source"](tag, pt.x(), pt.y(), pt.z());
        return field;
    }

    bool is_impedance(size_t bndnum)
    {
        auto bnd_data = lua["boundary"][bndnum];
        if (not bnd_data.valid())
            return false;

        std::string bndtype = bnd_data["kind"];
        if (bndtype == "impedance")
            return true;

        return false;
    }

    bool is_impedance_like(size_t bndnum)
    {
        auto bnd_data = lua["boundary"][bndnum];
        if (not bnd_data.valid())
            return false;

        std::string bndtype = bnd_data["kind"];
        if (bndtype == "impedance" or bndtype == "plane_wave")
            return true;

        return false;
    }

    bool is_plane_wave(size_t bndnum)
    {
        auto bnd_data = lua["boundary"][bndnum];
        if (not bnd_data.valid())
            return false;

        std::string bndtype = bnd_data["kind"];
        if (bndtype == "plane_wave")
            return true;

        return false;
    }

    bool is_magnetic_like(size_t bndnum)
    {
        auto bnd_data = lua["boundary"][bndnum];
        if (not bnd_data.valid())
            return false;

        std::string bndtype = bnd_data["kind"];
        if (bndtype == "perfect_magnetic")
            return true;

        return false;
    }
};