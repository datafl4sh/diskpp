/*
 *       /\         DISK++, a template library for DIscontinuous SKeletal
 *      /__\        methods.
 *     /_\/_\
 *    /\    /\      Matteo Cicuttin (C) 2016, 2017, 2018
 *   /__\  /__\     matteo.cicuttin@enpc.fr
 *  /_\/_\/_\/_\    École Nationale des Ponts et Chaussées - CERMICS
 *
 * This file is copyright of the following authors:
 * Matteo Cicuttin (C) 2016, 2017, 2018         matteo.cicuttin@enpc.fr
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

#pragma once

#include "common/eigen.hpp"
#include "quadratures/quadratures.hpp"

using namespace Eigen;

namespace disk {



template<typename Mesh, typename Element, typename Basis>
Matrix<typename Basis::scalar_type, Dynamic, Dynamic>
make_mass_matrix(const Mesh& msh, const Element& elem, const Basis& basis, size_t di = 0)
{
	auto degree		= basis.degree();
    auto basis_size = basis.size();

    using T = typename Basis::scalar_type;

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(basis_size, basis_size);

    auto qps = integrate(msh, elem, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = basis.eval_functions(qp.point());
        ret += qp.weight() * priv::outer_product(phi, phi);
    }

    return ret;
}

template<typename Mesh, typename Element, typename Basis>
Matrix<typename Basis::scalar_type, Dynamic, Dynamic>
make_stiffness_matrix(const Mesh& msh, const Element& elem, const Basis& basis, size_t di = 0)
{
	auto degree		= basis.degree();
    auto basis_size = basis.size();

    using T = typename Basis::scalar_type;

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(basis_size, basis_size);

    auto qps = integrate(msh, elem, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto dphi = basis.eval_gradients(qp.point());
        ret += qp.weight() * priv::outer_product(dphi, dphi);
    }

    return ret;
}

template<typename Mesh, typename Element, typename Basis, typename Function>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
make_rhs(const Mesh& msh, const Element& elem,
         const Basis& basis, const Function& rhs_fun, size_t di = 0)
{
    auto degree		= basis.degree();
    auto basis_size = basis.size();

    using T = typename Basis::scalar_type;

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(basis_size);

    auto qps = integrate(msh, elem, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = basis.eval_functions(qp.point());
        auto f = rhs_fun(qp.point());
        ret += qp.weight() * priv::inner_product(f, phi);
    }

    return ret;
}

template<typename Mesh>
using SRT = typename Mesh::coordinate_type;

template<typename Mesh>
using VRT = Matrix<typename Mesh::coordinate_type, Mesh::dimension, 1>;

template<typename Mesh>
using MRT = Matrix<typename Mesh::coordinate_type, Mesh::dimension, Mesh::dimension>;

template<typename Mesh>
using PT = typename Mesh::point_type;


template<typename Mesh>
using scalar_rhs_function = std::function<SRT<Mesh>(PT<Mesh>)>;

template<typename Mesh>
using vector_rhs_function = std::function<VRT<Mesh>(PT<Mesh>)>;

template<typename Mesh>
using matrix_rhs_function = std::function<MRT<Mesh>(PT<Mesh>)>;

template<typename Mesh, typename Element>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
project_function(const Mesh& msh, const Element& elem,
                 size_t degree,
                 const scalar_rhs_function<Mesh>& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    auto basis = make_scalar_monomial_basis(msh, elem, degree);
    Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, elem, basis, di);
    Matrix<T, Dynamic, 1> rhs = make_rhs(msh, elem, basis, f, di);
    return mass.llt().solve(rhs);
}

template<typename Mesh, typename Element>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
project_function(const Mesh& msh, const Element& elem,
                 size_t degree,
                 const vector_rhs_function<Mesh>& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    auto basis = make_vector_monomial_basis(msh, elem, degree);
    Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, elem, basis, di);
    Matrix<T, Dynamic, 1> rhs = make_rhs(msh, elem, basis, f, di);
    return mass.llt().solve(rhs);
}

template<typename Mesh, typename Element>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
project_gradient(const Mesh& msh, const Element& elem,
                 size_t degree,
                 const matrix_rhs_function<Mesh>& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    auto basis = make_vector_monomial_basis(msh, elem, degree);
    size_t N = Mesh::dimension;

    auto basis_size = vector_basis_size(degree, Mesh::dimension, Mesh::dimension);

    Matrix<T, Dynamic, Dynamic> stiff = make_stiffness_matrix(msh, elem, basis, di);
    Matrix<T, Dynamic, 1> rhs = Matrix<T, Dynamic, 1>::Zero(basis_size);

    auto qps = integrate(msh, elem, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto dphi = basis.eval_gradients(qp.point());
        auto rhs_func = f(qp.point());
        rhs += qp.weight() * priv::outer_product(dphi, rhs_func);
    }

    return stiff.block(N,N, basis_size-N, basis_size-N).llt().solve(rhs.tail(basis_size-N));
}

template<typename T>
T
eval(const dynamic_vector<T>& tab_coeff, const dynamic_vector<T>& base)
{
   assert(tab_coeff.rows() == base.rows());

   return tab_coeff.dot(base);
}

template<typename T, int DIM>
static_vector<T, DIM>
eval(const dynamic_vector<T>& tab_coeff, const Matrix<T, Dynamic, DIM>& base)
{
   static_vector<T, DIM> ret = static_vector<T, DIM>::Zero();
   assert(tab_coeff.size() == (base.rows()));

   for (int i = 0; i < tab_coeff.size(); i++) {
      ret += tab_coeff(i) * (base.row(i)).transpose();
   }

   return ret;
}

template<typename T, int DIM>
static_matrix<T, DIM, DIM>
eval(const dynamic_vector<T>&                                      tab_coeff,
     const eigen_compatible_stdvector<static_matrix<T, DIM, DIM>>& base,
     const size_t begin = 0)
{
   static_matrix<T, DIM, DIM> ret = static_matrix<T, DIM, DIM>::Zero();
   assert(tab_coeff.size() == (base.size()-begin));

   for (int i = 0; i < tab_coeff.size(); i++) {
      ret += tab_coeff(i) * base[i+begin];
   }

   return ret;
}

} //end disk