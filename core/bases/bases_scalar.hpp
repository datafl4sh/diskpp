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
 * Nicolas Pignet  (C) 2018                     nicolas.pignet@enpc.fr
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

//#define POWER_CACHE

#ifdef POWER_CACHE
#define PC_OFS_2D_X(d) (2 * d)
#define PC_OFS_2D_Y(d) (2 * d + 1)
#define PC_OFS_3D_X(d) (3 * d)
#define PC_OFS_3D_Y(d) (3 * d + 1)
#define PC_OFS_3D_Z(d) (3 * d + 2)
#endif

#include <vector>

#include "common/eigen.hpp"
#include "mesh/mesh.hpp"

using namespace Eigen;

namespace disk
{

/* Perform exponentiation by integer exponent. */
template<typename T>
T
iexp_pow(T x, size_t n)
{
    if (n == 0)
        return 1;

    T y = 1;
    while (n > 1)
    {
        if (n % 2 == 0)
        {
            x = x * x;
            n = n / 2;
        }
        else
        {
            y = x * y;
            x = x * x;
            n = (n - 1) / 2;
        }
    }

    return x * y;
}

/* Compute the size of a scalar basis of degree k in dimension d. */
size_t
scalar_basis_size(size_t k, size_t d)
{
    size_t num = 1;
    size_t den = 1;

    for (size_t i = 1; i <= d; i++)
    {
        num *= k + i;
        den *= i;
    }

    return num / den;
}

/* Generic template for bases. */
template<typename MeshType, typename Element>
struct scaled_monomial_scalar_basis
{
    static_assert(sizeof(MeshType) == -1, "scaled_monomial_scalar_basis: not suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "scaled_monomial_scalar_basis: not suitable for the requested kind of element");
};

/* Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_scalar_monomial_basis(const MeshType& msh, const ElementType& elem, size_t degree)
{
    return scaled_monomial_scalar_basis<MeshType, ElementType>(msh, elem, degree);
}

/***************************************************************************************************/
/***************************************************************************************************/

/* Specialization for 2D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_scalar_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, Dynamic, 2>     gradient_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

  private:
    point_type                 cell_bar;
    std::array<scalar_type, 2> box_h;
    size_t                     basis_degree, basis_size;

#ifdef POWER_CACHE
    mutable std::vector<scalar_type> power_cache;
#endif

  public:
    scaled_monomial_scalar_basis(const mesh_type& msh, const cell_type& cl, size_t degree)
    {
        cell_bar     = barycenter(msh, cl);
        box_h        = diameter_boundingbox(msh, cl);
        basis_degree = degree;
        basis_size   = scalar_basis_size(degree, 2);
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        const auto bx = (pt.x() - cell_bar.x()) / (0.5 * box_h[0]);
        const auto by = (pt.y() - cell_bar.y()) / (0.5 * box_h[1]);

#ifdef POWER_CACHE
        if (power_cache.size() != (basis_degree + 1) * 2)
            power_cache.resize((basis_degree + 1) * 2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[PC_OFS_2D_X(i)] = bx * power_cache[PC_OFS_2D_X(i - 1)];
            power_cache[PC_OFS_2D_Y(i)] = by * power_cache[PC_OFS_2D_Y(i - 1)];
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                const auto pow_x = k - i;
                const auto pow_y = i;
#ifdef POWER_CACHE
                const auto px = power_cache[PC_OFS_2D_X(pow_x)];
                const auto py = power_cache[PC_OFS_2D_Y(pow_y)];
#else
                const auto px = iexp_pow(bx, pow_x);
                const auto py = iexp_pow(by, pow_y);
#endif
                ret(pos++) = px * py;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    gradient_type
    eval_gradients(const point_type& pt) const
    {
        gradient_type ret = gradient_type::Zero(basis_size, 2);

        const auto ihx = 2.0 / box_h[0];
        const auto ihy = 2.0 / box_h[1];

        const auto bx = (pt.x() - cell_bar.x()) / (0.5 * box_h[0]);
        const auto by = (pt.y() - cell_bar.y()) / (0.5 * box_h[1]);

#ifdef POWER_CACHE
        if (power_cache.size() != (basis_degree + 1) * 2)
            power_cache.resize((basis_degree + 1) * 2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[PC_OFS_2D_X(i)] = bx * power_cache[PC_OFS_2D_X(i - 1)];
            power_cache[PC_OFS_2D_Y(i)] = by * power_cache[PC_OFS_2D_Y(i - 1)];
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                const auto pow_x = k - i;
                const auto pow_y = i;
#ifdef POWER_CACHE
                const auto px = power_cache[PC_OFS_2D_X(pow_x)];
                const auto py = power_cache[PC_OFS_2D_Y(pow_y)];
                const auto dx = (pow_x == 0) ? 0 : pow_x * ihx * power_cache[PC_OFS_2D_X(pow_x - 1)];
                const auto dy = (pow_y == 0) ? 0 : pow_y * ihy * power_cache[PC_OFS_2D_Y(pow_y - 1)];
#else
                const auto px = iexp_pow(bx, pow_x);
                const auto py = iexp_pow(by, pow_y);
                const auto dx = (pow_x == 0) ? 0 : pow_x * ihx * iexp_pow(bx, pow_x - 1);
                const auto dy = (pow_y == 0) ? 0 : pow_y * ihy * iexp_pow(by, pow_y - 1);
#endif
                ret(pos, 0) = dx * py;
                ret(pos, 1) = px * dy;
                pos++;
            }
        }

        assert(pos == basis_size);

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

/* Specialization for 2D meshes, faces */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_scalar_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::face>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type      point_type;
    typedef typename mesh_type::face            face_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

  private:
    point_type  face_bar;
    point_type  base;
    scalar_type face_h;
    size_t      basis_degree, basis_size;

#ifdef POWER_CACHE
    mutable std::vector<scalar_type> power_cache;
#endif

  public:
    scaled_monomial_scalar_basis(const mesh_type& msh, const face_type& fc, size_t degree)
    {
        face_bar     = barycenter(msh, fc);
        face_h       = diameter(msh, fc);
        basis_degree = degree;
        basis_size   = degree + 1;

        const auto pts = points(msh, fc);

        base = face_bar - pts[0];
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        const auto v   = base.to_vector();
        const auto t   = (pt - face_bar).to_vector();
        const auto dot = v.dot(t);
        const auto ep  = 4.0 * dot / (face_h * face_h);

        for (size_t i = 0; i <= basis_degree; i++)
        {
            const auto bv = iexp_pow(ep, i);
            ret(i)        = bv;
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

/***************************************************************************************************/
/***************************************************************************************************/

/* Specialization for 3D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_scalar_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::cell>
{

  public:
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, Dynamic, 3>     gradient_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

  private:
    point_type                 cell_bar;
    std::array<scalar_type, 3> box_h;
    size_t                     basis_degree, basis_size;

#ifdef POWER_CACHE
    mutable std::vector<scalar_type> power_cache;
#endif

  public:
    scaled_monomial_scalar_basis(const mesh_type& msh, const cell_type& cl, size_t degree)
    {
        cell_bar     = barycenter(msh, cl);
        box_h        = diameter_boundingbox(msh, cl);
        basis_degree = degree;
        basis_size   = scalar_basis_size(degree, 3);
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        const auto bx = (pt.x() - cell_bar.x()) / (0.5 * box_h[0]);
        const auto by = (pt.y() - cell_bar.y()) / (0.5 * box_h[1]);
        const auto bz = (pt.z() - cell_bar.z()) / (0.5 * box_h[2]);

#ifdef POWER_CACHE
        if (power_cache.size() != (basis_degree + 1) * 3)
            power_cache.resize((basis_degree + 1) * 3);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        power_cache[2] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[PC_OFS_3D_X(i)] = bx * power_cache[PC_OFS_3D_X(i - 1)];
            power_cache[PC_OFS_3D_Y(i)] = by * power_cache[PC_OFS_3D_Y(i - 1)];
            power_cache[PC_OFS_3D_Z(i)] = bz * power_cache[PC_OFS_3D_Z(i - 1)];
        }
#endif
        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t pow_x = 0; pow_x <= k; pow_x++)
            {
                for (size_t pow_y = 0, pow_z = k - pow_x; pow_y <= k - pow_x; pow_y++, pow_z--)
                {
#ifdef POWER_CACHE
                    const auto px = power_cache[PC_OFS_3D_X(pow_x)];
                    const auto py = power_cache[PC_OFS_3D_Y(pow_y)];
                    const auto pz = power_cache[PC_OFS_3D_Z(pow_z)];
#else
                    const auto px = iexp_pow(bx, pow_x);
                    const auto py = iexp_pow(by, pow_y);
                    const auto pz = iexp_pow(bz, pow_z);
#endif
                    ret(pos++) = px * py * pz;
                }
            }
        }
        assert(pos == basis_size);

        return ret;
    }

    gradient_type
    eval_gradients(const point_type& pt) const
    {
        gradient_type ret = gradient_type::Zero(basis_size, 3);

        const auto bx = (pt.x() - cell_bar.x()) / (0.5 * box_h[0]);
        const auto by = (pt.y() - cell_bar.y()) / (0.5 * box_h[1]);
        const auto bz = (pt.z() - cell_bar.z()) / (0.5 * box_h[2]);

        const auto ihx = 2.0 / box_h[0];
        const auto ihy = 2.0 / box_h[1];
        const auto ihz = 2.0 / box_h[2];

#ifdef POWER_CACHE
        if (power_cache.size() != (basis_degree + 1) * 3)
            power_cache.resize((basis_degree + 1) * 3);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        power_cache[2] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[PC_OFS_3D_X(i)] = bx * power_cache[PC_OFS_3D_X(i - 1)];
            power_cache[PC_OFS_3D_Y(i)] = by * power_cache[PC_OFS_3D_Y(i - 1)];
            power_cache[PC_OFS_3D_Z(i)] = bz * power_cache[PC_OFS_3D_Z(i - 1)];
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t pow_x = 0; pow_x <= k; pow_x++)
            {
                for (size_t pow_y = 0, pow_z = k - pow_x; pow_y <= k - pow_x; pow_y++, pow_z--)
                {
#ifdef POWER_CACHE
                    const auto px = power_cache[PC_OFS_3D_X(pow_x)];
                    const auto py = power_cache[PC_OFS_3D_Y(pow_y)];
                    const auto pz = power_cache[PC_OFS_3D_Z(pow_z)];
                    const auto dx = (pow_x == 0) ? 0 : pow_x * ihx * power_cache[PC_OFS_3D_X(pow_x - 1)];
                    const auto dy = (pow_y == 0) ? 0 : pow_y * ihy * power_cache[PC_OFS_3D_Y(pow_y - 1)];
                    const auto dz = (pow_z == 0) ? 0 : pow_z * ihz * power_cache[PC_OFS_3D_Z(pow_z - 1)];
#else
                    const auto px = iexp_pow(bx, pow_x);
                    const auto py = iexp_pow(by, pow_y);
                    const auto pz = iexp_pow(bz, pow_z);
                    const auto dx = (pow_x == 0) ? 0 : pow_x * ihx * iexp_pow(bx, pow_x - 1);
                    const auto dy = (pow_y == 0) ? 0 : pow_y * ihy * iexp_pow(by, pow_y - 1);
                    const auto dz = (pow_z == 0) ? 0 : pow_z * ihz * iexp_pow(bz, pow_z - 1);
#endif
                    ret(pos, 0) = dx * py * pz;
                    ret(pos, 1) = px * dy * pz;
                    ret(pos, 2) = px * py * dz;
                    pos++;
                }
            }
        }

        assert(pos == basis_size);

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

template<typename Mesh, typename Element>
class scaled_monomial_abstract_face_basis;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_abstract_face_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::face>
{
public:
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type      point_type;
    typedef typename mesh_type::face            face_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

private:
    point_type  face_bar;
    scalar_type face_h;

    /* Local reference frame */
    typedef static_vector<T, 3>                 vector_type;
    vector_type                                 e0, e1;


    /* It takes two edges of an element's face and uses them as the coordinate
     * axis of a 2D reference system. Those two edges are accepted only if they have an angle
     * between them greater than 8 degrees, then they are orthonormalized via G-S. */

    void
    compute_axis(const mesh_type& msh, const face_type& fc, vector_type& e0, vector_type& e1)
    {
        vector_type v0;
        vector_type v1;

        bool ok = false;

        const auto pts = points(msh, fc);

        const size_t npts = pts.size();
        for (size_t i = 1; i <= npts; i++)
        {
            const size_t i0 = (i + 1) % npts;
            const size_t i1 = (i - 1) % npts;
            v0              = (pts[i0] - pts[i]).to_vector();
            v1              = (pts[i1] - pts[i]).to_vector();

            const vector_type v0n = v0 / v0.norm();
            const vector_type v1n = v1 / v1.norm();

            if (v0n.dot(v1n) < 0.99) // we want at least 8 degrees angle
            {
                ok = true;
                break;
            }
        }

        if (!ok)
            throw std::invalid_argument("Degenerate polyhedron, cannot proceed");

        /* Don't normalize, in order to keep axes of the same order of lenght
         * of v in make_face_point_3d_to_2d() */
        e0 = v0 / v0.norm();
        e1 = v1 - (v1.dot(v0) * v0) / (v0.dot(v0));
        e1 = e1 / e1.norm();
    }

protected:
    /* This function maps a 3D point on a face to a 2D reference system, to compute the
     * face basis. It takes two edges of an element's face and uses them as the coordinate
     * axis of a 2D reference system. Those two edges are accepted only if they have an angle
     * between them greater than 8 degrees, then they are orthonormalized via G-S. */
    point<T, 2>
    map_face_point_3d_to_2d(const point_type& pt) const
    {
        vector_type v = (pt - face_bar).to_vector();

        const auto eta = v.dot(e0);
        const auto xi  = v.dot(e1);

        return point<T, 2>({eta, xi});
    }

    auto face_barycenter() const
    {
        return face_bar;
    }

    auto face_diameter() const
    {
        return face_h;
    }

    auto reference_frame() const
    {
        return std::make_pair(e0, e1);
    }

public:
    scaled_monomial_abstract_face_basis(const mesh_type& msh, const face_type& fc)
    {
        face_bar     = barycenter(msh, fc);
        face_h       = diameter(msh, fc);
        compute_axis(msh, fc, e0, e1);
    }
};


/* Specialization for 3D meshes, faces */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_monomial_scalar_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::face>
    : public scaled_monomial_abstract_face_basis<Mesh<T, 3, Storage>, typename Mesh<T, 3, Storage>::face>
{

public:
    typedef Mesh<T, 3, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type      point_type;
    typedef typename mesh_type::face            face_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

    using base = scaled_monomial_abstract_face_basis<mesh_type, face_type>;

private:
    point_type  face_bar;
    scalar_type face_h;
    size_t      basis_degree, basis_size;

  public:
    scaled_monomial_scalar_basis(const mesh_type& msh, const face_type& fc, size_t degree)
        : base(msh, fc)
    {
        face_bar     = this->face_barycenter();
        face_h       = this->face_diameter();
        basis_degree = degree;
        basis_size   = scalar_basis_size(degree, 2);
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        const auto ep = this->map_face_point_3d_to_2d(pt);
        const auto bx = ep.x();
        const auto by = ep.y();

        function_type ret = function_type::Zero(basis_size);
        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                const auto pow_x = k - i;
                const auto pow_y = i;
                const auto px = iexp_pow(bx, pow_x);
                const auto py = iexp_pow(by, pow_y);
                ret(pos++) = px * py;
            }
        }

        assert(pos == basis_size);

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

template<typename MeshType, typename Element>
struct scaled_legendre_scalar_basis
{
    static_assert(sizeof(MeshType) == -1, "scaled_monomial_scalar_basis: not suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "scaled_monomial_scalar_basis: not suitable for the requested kind of element");
};

/* Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_scalar_legendre_basis(const MeshType& msh, const ElementType& elem, size_t degree)
{
    return scaled_legendre_scalar_basis<MeshType, ElementType>(msh, elem, degree);
}

/* Specialization for 2D meshes, faces */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class scaled_legendre_scalar_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::face>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type      point_type;
    typedef typename mesh_type::face            face_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

  private:
    point_type  face_bar;
    point_type  base;
    scalar_type face_h;
    size_t      basis_degree, basis_size;

    scalar_type
    eval_poly(const point_type& pt, size_t num)
    {
        const auto x = pt.x();

        std::array<scalar_type, 11> pows;
        pows[0] = 1;
        for (size_t i = 1; i <= 10; i++)
            pows[i] = x * pows[i - 1];

        switch (num)
        {
            case 0: return 1;
            case 1: return x;
            case 2: return (3 * pows[2] - 1) / 2;
            case 3: return (5 * pows[3] - 3 * pows[1]) / 2;
            case 4: return (35 * pows[4] - 30 * pows[2]) / 8;
            case 5: return (63 * pows[5] - 70 * pows[3] + 15 * pows[1]) / 8;
            case 6: return (231 * pows[6] - 315 * pows[4] + 105 * pows[2] - 5) / 16;
        }
    }

  public:
    scaled_legendre_scalar_basis(const mesh_type& msh, const face_type& fc, size_t degree)
    {
        face_bar     = barycenter(msh, fc);
        face_h       = diameter(msh, fc);
        basis_degree = degree;
        basis_size   = degree + 1;

        if (degree > 10)
            throw std::invalid_argument("Sorry, I don't have a Legendre basis of order > 10.");

        const auto pts = points(msh, fc);
        base           = face_bar - pts[0];
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        const auto v   = base.to_vector();
        const auto t   = (pt - face_bar).to_vector();
        const auto dot = v.dot(t);
        const auto ep  = 4.0 * dot / (face_h * face_h);

        for (size_t i = 0; i <= basis_degree; i++)
        {
            auto bv = iexp_pow(ep, i);
            ret(i)  = bv;
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

/*
template<typename Mesh, typename Element, typename Basis>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
compute_averages(const Mesh& msh, const Element& elem, const Basis& basis)
{
    using T = typename Mesh::coordinate_type;
    auto meas = measure(msh, elem);

    Matrix<T, Dynamic, 1> avgs = Matrix<T, Dynamic, 1>::Zero(basis.size());

    auto qps = integrate(msh, elem, basis.degree());
    for (auto& qp : qps)
        avgs += qp.weight() * basis.eval_functions(qp.point());

    avgs(0) = 0;

    return avgs * (1/meas);
}
*/
}
