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
 * Karol Cascavita (C) 2018                     karol.cascavita@enpc.fr
 * Nicolas Pignet  (C) 2018, 2019               nicolas.pignet@enpc.fr
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

#include <vector>

#include "bases_scalar.hpp"
#include "bases_vector.hpp"
#include "common/eigen.hpp"

namespace disk
{

/* Compute the size of a matrix basis of degree k in dimension d. */
size_t
matrix_basis_size(size_t k, size_t sd, size_t md)
{
    size_t num = 1;
    size_t den = 1;

    for (size_t i = 1; i <= sd; i++)
    {
        num *= k + i;
        den *= i;
    }

    return md * md * (num / den);
}

/* Generic template for bases. */
template<typename MeshType, typename Element>
struct scaled_monomial_matrix_basis
{
    static_assert(sizeof(MeshType) == -1, "scaled_monomial_matrix_basis: not suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "scaled_monomial_matrix_basis: not suitable for the requested kind of element");
};

/* Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_matrix_monomial_basis(const MeshType&    msh,
                           const ElementType& elem,
                           size_t             degree,
                           bool               use_inertia_axes = USE_INERTIA_AXES)
{
    return scaled_monomial_matrix_basis<MeshType, ElementType>(msh, elem, degree, use_inertia_axes);
}

/* Specialization for 3D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_matrix_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 3, Storage>,
                                          typename Mesh<T, 3, Storage>::cell,
                                          typename Mesh<T, 3, Storage>::coordinate_type>
{

  public:
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef static_matrix<scalar_type, 3, 3>    function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

  public:
    scaled_monomial_matrix_basis(const mesh_type& msh,
                                 const cell_type& cl,
                                 size_t           degree,
                                 bool             use_inertia_axes = false) :
      scalar_basis(msh, cl, degree, use_inertia_axes),
      base(msh, cl, use_inertia_axes)
    {
        basis_degree = degree;
        basis_size   = matrix_basis_size(degree, 3, 3);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        const auto phi = scalar_basis.eval_functions(pt);

        for (int k = 0; k < scalar_basis.size(); k++)
        {
            function_type fc;

            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    fc       = function_type::Zero();
                    fc(i, j) = phi(k);
                    ret.push_back(fc);
                }
            }
        }

        assert(ret.size() == basis_size);

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};


/* Specialization for 2D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_matrix_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 2, Storage>,
                                          typename Mesh<T, 2, Storage>::cell,
                                          typename Mesh<T, 2, Storage>::coordinate_type>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef static_matrix<scalar_type, 2, 2>    function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

  public:
    scaled_monomial_matrix_basis(const mesh_type& msh,
                                 const cell_type& cl,
                                 size_t           degree,
                                 bool             use_inertia_axes = false) :
      scalar_basis(msh, cl, degree, use_inertia_axes),
      base(msh, cl, use_inertia_axes)
    {
        basis_degree = degree;
        basis_size   = matrix_basis_size(degree, 2, 2);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        const auto phi = scalar_basis.eval_functions(pt);

        for (int k = 0; k < scalar_basis.size(); k++)
        {
            function_type fc;

            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    fc       = function_type::Zero();
                    fc(i, j) = phi(k);
                    ret.push_back(fc);
                }
            }
        }

        assert(ret.size() == basis_size);

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};


/* Compute the size of a matrix basis of degree k in dimension d. */
size_t
sym_matrix_basis_size(size_t k, size_t sd, size_t vd)
{
    size_t md = 0;

    switch (vd)
    {
        case 3: md = 6; break;

        case 2: md = 3; break;

        default: std::logic_error("Expected 3 >= dim > 1");
    }

    if (k == 0)
        return md;

    size_t num = 1;
    size_t den = 1;

    for (size_t i = 1; i <= sd; i++)
    {
        num *= k + i;
        den *= i;
    }

    return md * (num / den);
}

/* Generic template for bases. */
template<typename MeshType, typename Element>
struct scaled_monomial_sym_matrix_basis
{
    static_assert(sizeof(MeshType) == -1,
                  "scaled_monomial_sym_matrix_basis: not "
                  "suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "scaled_monomial_sym_matrix_basis: not "
                  "suitable for the requested kind of element");
};

/* Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_sym_matrix_monomial_basis(const MeshType&    msh,
                               const ElementType& elem,
                               size_t             degree,
                               bool               use_inertia_axes = USE_INERTIA_AXES)
{
    return scaled_monomial_sym_matrix_basis<MeshType, ElementType>(msh, elem, degree, use_inertia_axes);
}

/* Specialization for 3D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_sym_matrix_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 3, Storage>,
                                          typename Mesh<T, 3, Storage>::cell,
                                          typename Mesh<T, 3, Storage>::coordinate_type>
{

  public:
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef static_matrix<scalar_type, 3, 3>    function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

  public:
    scaled_monomial_sym_matrix_basis(const mesh_type& msh,
                                     const cell_type& cl,
                                     size_t           degree,
                                     bool             use_inertia_axes = false) :
      scalar_basis(msh, cl, degree, use_inertia_axes),
      base(msh, cl, use_inertia_axes)
    {
        basis_degree = degree;
        basis_size   = sym_matrix_basis_size(degree, 3, 3);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        auto              phi     = scalar_basis.eval_functions(pt);
        const scalar_type invrac2 = 1.0 / std::sqrt(2.0);

        for (int k = 0; k < scalar_basis.size(); k++)
        {
            function_type fc;

            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < j; i++)
                {
                    fc       = function_type::Zero();
                    fc(i, j) = phi(k) * invrac2;
                    fc(j, i) = phi(k) * invrac2;
                    ret.push_back(fc);
                }

                fc       = function_type::Zero();
                fc(j, j) = phi(k);
                ret.push_back(fc);
            }
        }

        assert(ret.size() == basis_size);
        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};


/* Specialization for 2D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_sym_matrix_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell,
                                          typename Mesh<T, 2, Storage>::coordinate_type>
{
  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef static_matrix<scalar_type, 2, 2>    function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

  public:
    scaled_monomial_sym_matrix_basis(const mesh_type& msh,
                                     const cell_type& cl,
                                     size_t           degree,
                                     bool             use_inertia_axes = false) :
      scalar_basis(msh, cl, degree, use_inertia_axes),
      base(msh, cl, use_inertia_axes)
    {
        basis_degree = degree;
        basis_size   = sym_matrix_basis_size(degree, 2, 2);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        auto              phi     = scalar_basis.eval_functions(pt);
        const scalar_type invrac2 = 1.0 / std::sqrt(2.0);

        for (int k = 0; k < scalar_basis.size(); k++)
        {
            function_type fc;

            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < j; i++)
                {
                    fc       = function_type::Zero();
                    fc(i, j) = phi(k) * invrac2;
                    fc(j, i) = phi(k) * invrac2;
                    ret.push_back(fc);
                }

                fc       = function_type::Zero();
                fc(j, j) = phi(k);
                ret.push_back(fc);
            }
        }

        assert(ret.size() == basis_size);
        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};


///////////////////////////////////////////////////
//// Raviart-Thomas elements on simplicial ////////
///////////////////////////////////////////////////

/* Compute the size of a vector basis of degree k in dimension d. */
size_t
matrix_basis_size_RT(size_t k, size_t sd, size_t md)
{
    if (k <= 0)
        throw std::invalid_argument("Raviart-Thomas basis: degree has to be > 0");

    return sd * vector_basis_size_RT(k, sd, md);
}

// RT^{k}(T; R^{dxd}) = P^{k-1}(T;R^{dxd}) +  P^{k-1,H}(T;R^d) otimes x)

/* Generic template for RT bases. */
template<typename MeshType, typename Element>
struct scaled_monomial_matrix_basis_RT
{
    static_assert(sizeof(MeshType) == -1,
                  "scaled_monomial_matrix_basis_RT: not suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "scaled_monomial_matrix_basis_RT: not suitable for the requested kind of element");
};

/* RT Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_matrix_monomial_basis_RT(const MeshType& msh, const ElementType& elem, const size_t degree)
{
    return scaled_monomial_matrix_basis_RT<MeshType, ElementType>(msh, elem, degree);
}

/* Specialization for 3D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_matrix_basis_RT<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::cell,
                                          typename Mesh<T, 3, Storage>::coordinate_type>
{

  public:
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, 3, 3>           function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;
    point_type cell_bar;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

    typedef scaled_monomial_matrix_basis<mesh_type, cell_type> matrix_basis_type;
    matrix_basis_type                                          matrix_basis;

  public:
    scaled_monomial_matrix_basis_RT(const mesh_type& msh, const cell_type& cl, const size_t degree) :
      basis_degree(degree), basis_size(matrix_basis_size_RT(degree, 3, 3)), scalar_basis(msh, cl, degree - 1, false),
      matrix_basis(msh, cl, degree - 1, false), base(msh, cl, false)
    {
        if (basis_degree <= 0)
            throw std::invalid_argument("Raviart-Thomas basis: degree has to be > 0");

        if (points(msh, cl).size() != 4)
            throw std::invalid_argument("Raviart-Thomas basis: available only on tetrahedron");

        cell_bar = barycenter(msh, cl);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        const auto mb = matrix_basis.eval_functions(pt);

        std::move(mb.begin(), mb.end(), std::back_inserter(ret));

        const auto sphi = scalar_basis.eval_functions(pt);
        size_t     beg  = 0;
        if (basis_degree >= 2)
            beg = scalar_basis_size(basis_degree - 2, 3);

        Matrix<scalar_type, 1, 3> row;

        const auto bx = (pt.x() - cell_bar.x());
        const auto by = (pt.y() - cell_bar.y());
        const auto bz = (pt.z() - cell_bar.z());

        row(0) = bx;
        row(1) = by;
        row(2) = bz;

        // compute P^(k-1)_H otimes X (monomial of degree exactly k - 1)
        for (size_t i = beg; i < scalar_basis.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                function_type mat = function_type::Zero();
                mat.row(j)        = sphi(i) * row;
                ret.push_back(mat);
            }
        }

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

/* Specialization for 2D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_matrix_basis_RT<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 2, Storage>,
                                          typename Mesh<T, 2, Storage>::cell,
                                          typename Mesh<T, 2, Storage>::coordinate_type>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, 2, 2>           function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;
    point_type cell_bar;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

    typedef scaled_monomial_matrix_basis<mesh_type, cell_type> matrix_basis_type;
    matrix_basis_type                                          matrix_basis;

  public:
    scaled_monomial_matrix_basis_RT(const mesh_type& msh, const cell_type& cl, const size_t degree) :
      basis_degree(degree), basis_size(matrix_basis_size_RT(degree, 2, 2)), scalar_basis(msh, cl, degree - 1, false),
      matrix_basis(msh, cl, degree - 1, false), base(msh, cl, false)
    {
        if (basis_degree <= 0)
            throw std::invalid_argument("Raviart-Thomas basis: degree has to be > 0");

        if (points(msh, cl).size() != 3)
            throw std::invalid_argument("Raviart-Thomas basis: available only on triangle");

        cell_bar = barycenter(msh, cl);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        const auto mb = matrix_basis.eval_functions(pt);

        std::move(mb.begin(), mb.end(), std::back_inserter(ret));

        const auto sphi = scalar_basis.eval_functions(pt);
        size_t     beg  = 0;
        if (basis_degree >= 2)
            beg = scalar_basis_size(basis_degree - 2, 2);

        Matrix<scalar_type, 1, 2> row;

        const auto bx = (pt.x() - cell_bar.x());
        const auto by = (pt.y() - cell_bar.y());
        row(0) = bx;
        row(1) = by;

        // compute P^(k-1)_H otimes X (monomial of degree exactly k - 1)
        for (size_t i = beg; i < scalar_basis.size(); i++)
        {
            for (size_t j = 0; j < 2; j++)
            {
                function_type mat = function_type::Zero();
                mat.row(j)        = sphi(i) * row;
                ret.push_back(mat);
            }
        }

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

///////////////////////////////////////////////////
//// Symmetric Raviart-Thomas elements on simplicial ////////
///////////////////////////////////////////////////

/* Compute the size of a vector basis of degree k in dimension d. */
size_t
sym_matrix_basis_size_RT(size_t k, size_t sd, size_t md)
{
    if (k <= 0)
        throw std::invalid_argument("Raviart-Thomas basis: degree has to be > 0");

    const auto smb_size = sym_matrix_basis_size(k - 1, sd, md);

    size_t beg = 0;
    if (k >= 2)
        beg = scalar_basis_size(k - 2, sd);
    const auto end = scalar_basis_size(k - 1, sd);

    return smb_size + sd * (end - beg);
}

// RT^{k}(T; R^{dxd}_sym) = P^{k-1}(T;R^{dxd}_sym) + Sym( P^{k-1,H}(T;R^d) otimes x)

/* Generic template for RT bases. */
template<typename MeshType, typename Element>
struct scaled_monomial_sym_matrix_basis_RT
{
    static_assert(sizeof(MeshType) == -1,
                  "scaled_monomial_sym_matrix_basis_RT: not suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "scaled_monomial_sym_matrix_basis_RT: not suitable for the requested kind of element");
};

/* RT Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_sym_matrix_monomial_basis_RT(const MeshType& msh, const ElementType& elem, const size_t degree)
{
    return scaled_monomial_sym_matrix_basis_RT<MeshType, ElementType>(msh, elem, degree);
}

/* Specialization for 3D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_sym_matrix_basis_RT<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::cell,
                                          typename Mesh<T, 3, Storage>::coordinate_type>
{

  public:
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, 3, 3>           function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;
    point_type cell_bar;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

    typedef scaled_monomial_sym_matrix_basis<mesh_type, cell_type> sym_matrix_basis_type;
    sym_matrix_basis_type                                          sym_matrix_basis;

  public:
    scaled_monomial_sym_matrix_basis_RT(const mesh_type& msh, const cell_type& cl, const size_t degree) :
      basis_degree(degree), basis_size(sym_matrix_basis_size_RT(degree, 3, 3)),
      scalar_basis(msh, cl, degree - 1, false), sym_matrix_basis(msh, cl, degree - 1, false),
      base(msh, cl, false)
    {
        if (degree <= 0)
            throw std::invalid_argument("Raviart-Thomas basis: degree has to be > 0");

        if (points(msh, cl).size() != 4)
            throw std::invalid_argument("Raviart-Thomas basis: available only on tetrahedron");

        cell_bar = barycenter(msh, cl);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        const auto smb = sym_matrix_basis.eval_functions(pt);

        std::move(smb.begin(), smb.end(), std::back_inserter(ret));

        const auto sphi = scalar_basis.eval_functions(pt);
        size_t     beg  = 0;
        if (basis_degree >= 2)
            beg = scalar_basis_size(basis_degree - 2, 2);

        const auto bx = (pt.x() - cell_bar.x());
        const auto by = (pt.y() - cell_bar.y());
        const auto bz = (pt.z() - cell_bar.z());

        function_type mat1 = function_type::Zero();

        mat1(0, 0) = bx;
        mat1(0, 1) = by / scalar_type(2);
        mat1(1, 0) = mat1(0, 1);
        mat1(0, 2) = bz / scalar_type(2);
        mat1(2, 0) = mat1(0, 2);

        function_type mat2 = function_type::Zero();

        mat2(0, 1) = bx / scalar_type(2);
        mat2(1, 0) = mat2(0, 1);
        mat2(1, 1) = by;
        mat2(1, 2) = bz / scalar_type(2);
        mat2(2, 1) = mat2(1, 2);

        function_type mat3 = function_type::Zero();

        mat3(0, 2) = bx / scalar_type(2);
        mat3(2, 0) = mat3(0, 2);
        mat3(1, 2) = by / scalar_type(2);
        mat3(2, 1) = mat3(1, 2);
        mat3(2, 2) = bz;

        // compute sym(P^(k-1)_H otimes X) (monomial of degree exactly k - 1)
        for (size_t i = beg; i < scalar_basis.size(); i++)
        {
            ret.push_back(sphi(i) * mat1);
            ret.push_back(sphi(i) * mat2);
            ret.push_back(sphi(i) * mat3);
        }

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

/* Specialization for 2D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_sym_matrix_basis_RT<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell>
  : public scaled_monomial_abstract_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell,
                                          typename Mesh<T, 2, Storage>::coordinate_type>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, 2, 2>           function_type;

    using base = scaled_monomial_abstract_basis<mesh_type, cell_type, scalar_type>;

  private:
    size_t basis_degree, basis_size;
    point_type cell_bar;

    typedef scaled_monomial_scalar_basis<mesh_type, cell_type, scalar_type> scalar_basis_type;
    scalar_basis_type                                          scalar_basis;

    typedef scaled_monomial_sym_matrix_basis<mesh_type, cell_type> sym_matrix_basis_type;
    sym_matrix_basis_type                                          sym_matrix_basis;

  public:
    scaled_monomial_sym_matrix_basis_RT(const mesh_type& msh, const cell_type& cl, const size_t degree) :
      basis_degree(degree), basis_size(sym_matrix_basis_size_RT(degree, 2, 2)),
      scalar_basis(msh, cl, degree - 1, false), sym_matrix_basis(msh, cl, degree - 1, false),
      base(msh, cl, false)
    {
        if (degree <= 0)
            throw std::invalid_argument("Raviart-Thomas basis: degree has to be > 0");

        if (points(msh, cl).size() != 3)
            throw std::invalid_argument("Raviart-Thomas basis: available only on triangles");

        cell_bar = barycenter(msh, cl);
    }

    eigen_compatible_stdvector<function_type>
    eval_functions(const point_type& pt) const
    {
        eigen_compatible_stdvector<function_type> ret;
        ret.reserve(basis_size);

        const auto smb = sym_matrix_basis.eval_functions(pt);
        std::move(smb.begin(), smb.end(), std::back_inserter(ret));

        const auto sphi = scalar_basis.eval_functions(pt);
        size_t     beg  = 0;
        if (basis_degree >= 2)
            beg = scalar_basis_size(basis_degree - 2, 2);

        const auto bx = (pt.x() - cell_bar.x());
        const auto by = (pt.y() - cell_bar.y());

        function_type mat1 = function_type::Zero();

        mat1(0, 0) = bx;
        mat1(0, 1) = by / scalar_type(2);
        mat1(1, 0) = mat1(0, 1);

        function_type mat2 = function_type::Zero();

        mat2(0, 1) = bx / scalar_type(2);
        mat2(1, 0) = mat2(0, 1);
        mat2(1, 1) = by;

        // compute sym(P^(k-1)_H otimes X) (monomial of degree exactly k - 1)
        for (size_t i = beg; i < scalar_basis.size(); i++)
        {
            ret.push_back(sphi(i) * mat1);
            ret.push_back(sphi(i) * mat2);
        }

        if (ret.size() != basis_size)
        {
            std::cout << ret.size() << " vs " << basis_size << std::endl;
            throw std::invalid_argument("wrong size");
        }

        // std::cout << "ret size: " << ret.size() << std::endl;

        // for (auto& elem : ret)
        //     std::cout << elem << std::endl;

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

} // namespace disk
