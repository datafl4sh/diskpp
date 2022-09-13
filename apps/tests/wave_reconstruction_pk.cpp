/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *
 * Matteo Cicuttin (C) 2020
 * matteo.cicuttin@uliege.be
 *
 * University of Liège - Montefiore Institute
 * Applied and Computational Electromagnetics group
 */

#include <iomanip>
#include <iostream>
#include <regex>

#include <unistd.h>

#include "diskpp/loaders/loader.hpp"
#include "diskpp/methods/hho"
#include "diskpp/methods/implementation_hho/curl.hpp"


#include "common.hpp"

template<typename ScalT, typename Mesh>
struct test_functor_wave_reconstruction
{
    typename Mesh::coordinate_type
    operator()(const Mesh& msh, size_t degree) const
    {
        return 0;
    }

    size_t
    expected_rate(size_t k)
    {
        return k + 50;
    }
};

template<typename ScalT, template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct test_functor_wave_reconstruction<ScalT, Mesh<T,3,Storage>>
{
    using mesh_type = Mesh<T,3,Storage>;
    /* Expect k+1 convergence (hho stabilization) */
    typename mesh_type::coordinate_type
    operator()(const mesh_type& msh, size_t degree) const
    {
        auto kappa = 1.0;
        typedef typename mesh_type::cell            cell_type;
        typedef typename mesh_type::face            face_type;
        typedef typename mesh_type::coordinate_type coord_type;
        typedef ScalT            scalar_type;
        typedef typename mesh_type::point_type      point_type;

        scalar_type i = scalar_type(0.,1.);
        
        typedef Matrix<scalar_type, mesh_type::dimension, 1> ret_type;

        auto f = [&](const point_type& pt) -> Matrix<scalar_type,3,1> {
            Matrix<scalar_type,3,1> ret;
            ret(0) = 0.0;
            ret(1) = std::exp(-i*kappa*pt.x());
            ret(2) = 0.0;
            return ret;
        };

        auto sol = [&](const point_type& pt) -> Matrix<scalar_type,3,1> {
            Matrix<scalar_type,3,1> ret;
            ret(0) = 0.0;
            ret(1) = 0.0;
            ret(2) = -i*kappa*std::exp(-i*kappa*pt.x());
            return ret;
        };

        size_t fd = degree;
        size_t cd = degree;
        size_t rd = degree;

        disk::hho_degree_info hdi(disk::priv::hdi_named_args{.rd = rd, .cd = cd, .fd = fd});

        scalar_type error = 0.0;
        for (const auto& cl : msh)
        {
            auto CR = disk::wave_reconstruction_pk<scalar_type>(msh, cl, hdi, 1.0);
            auto proj = disk::project_tangent<scalar_type>(msh, cl, hdi, f, 1);
            Matrix<scalar_type, Dynamic, 1> rf = CR.first * proj;

            auto rb = disk::make_vector_monomial_basis(msh, cl, rd);

            auto qps = integrate(msh, cl, 2*std::max(rd,cd)+1);
            for (const auto& qp : qps)
            {
                auto rphi = rb.eval_functions(qp.point());
                Matrix<scalar_type,3,1> rval = Matrix<scalar_type,3,1>::Zero();
                for (size_t i = 0; i < rb.size(); i++)
                    rval += rf(i)*rphi.block(i,0,1,3).transpose();
                Matrix<scalar_type,3,1> diff = rval - sol(qp.point());
                //std::cout << "rf  : " << rf.transpose() << std::endl;
                //std::cout << "RVAL: " << rval.transpose() << std::endl;
                //std::cout << "SOL : " << sol(qp.point()).transpose() << std::endl;
                error += qp.weight() * diff.dot(diff);
            }
        }

        std::cout << error << std::endl;

        return real(std::sqrt(error));
    }

    size_t
    expected_rate(size_t k)
    {
        return k + 1;
    }
};

template<typename Mesh>
using test_functor_wave_reconstruction_eo = test_functor_wave_reconstruction<std::complex<double>, Mesh>;

template<typename Mesh>
using test_functor_wave_reconstruction_mo = test_functor_wave_reconstruction<std::complex<double>, Mesh>;

int
main(void)
{
    std::cout << red << "Test HHO wave reconstruction operator" << std::endl;
    // face order: k, cell order: k
    std::cout << cyan << "Face order: k and Cell order: k" << std::endl;
    tester<test_functor_wave_reconstruction_eo> tstr1;
    tstr1.run();

    std::cout << cyan << "Face order: k and Cell order: k+1" << std::endl;
    tester<test_functor_wave_reconstruction_mo> tstr2;
    tstr2.run();

    return 0;
}
