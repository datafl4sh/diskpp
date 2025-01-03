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

#include <iostream>
#include <iomanip>
#include <regex>
#include <unistd.h>

#include "diskpp/methods/hho"
#include "diskpp/loaders/loader.hpp"
#include "diskpp/solvers/solver.hpp"
#include "diskpp/output/silo.hpp"

#include "mumps.hpp"

template<typename Mesh, typename Velocity, typename Pressure, typename Assembler>
auto
compute_errors(const Mesh& msh,
                const disk::dynamic_vector<typename Mesh::coordinate_type>& sol,
                const typename disk::hho_degree_info & hdi,
                const Velocity& velocity,
                const Pressure& pressure,
                const Assembler& assembler,
                const bool& use_sym_grad)
{
    typedef Mesh mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;

    auto dim =  Mesh::dimension;

    scalar_type factor = (use_sym_grad)? 2. : 1.;

    scalar_type error(0), error_vel(0), error_pres(0);

    for (auto& cl : msh)
    {
    	auto bar = barycenter(msh, cl);
    	Matrix<scalar_type, Dynamic, 1> p = project_function(msh, cl, hdi, velocity);
    	auto cbs = disk::vector_basis_size(hdi.cell_degree(), dim, dim);
    	auto cell_ofs = disk::offset(msh, cl);
    	Matrix<scalar_type, Dynamic, 1> s = sol.block(cell_ofs * cbs, 0, cbs, 1);
    	Matrix<scalar_type, Dynamic, 1> diff = s - p.head(cbs);
    	auto cb = disk::make_vector_monomial_basis(msh, cl, hdi.cell_degree());
    	Matrix<scalar_type, Dynamic, Dynamic> mm = disk::make_mass_matrix(msh, cl, cb);
    	error += diff.dot(mm * diff);
    	//ofs << bar.x() << " " << bar.y() << " " << s(0) << " " << s(1) << std::endl;

        //pressure error
        Matrix<scalar_type, Dynamic, 1> ppres = disk::project_function(msh, cl, hdi.face_degree(), pressure);
        auto fbs = disk::vector_basis_size(hdi.face_degree(), dim - 1, dim);
        auto pbs = disk::scalar_basis_size(hdi.face_degree(), dim);
        auto pb  = disk::make_scalar_monomial_basis(msh, cl, hdi.face_degree());
        auto num_other_faces = assembler.num_assembled_faces();
        auto pres_ofs = cbs * msh.cells_size() + fbs * num_other_faces + pbs * cell_ofs;

        Matrix<scalar_type, Dynamic, 1> spres = sol.block(pres_ofs, 0, pbs, 1);
    	Matrix<scalar_type, Dynamic, 1> diff_pres = spres - ppres.head(pbs);
    	Matrix<scalar_type, Dynamic, Dynamic> scalar_mm = disk::make_mass_matrix(msh, cl, pb);
    	error_pres += diff_pres.dot(scalar_mm*diff_pres);

        //energy error
        auto num_faces = howmany_faces(msh, cl);
        Matrix<scalar_type, Dynamic, 1> svel(cbs + num_faces * fbs );
        svel.block(0, 0, cbs, 1) = sol.block(cell_ofs * cbs, 0, cbs, 1);
        auto fcs = faces(msh, cl);
        for (size_t i = 0; i < fcs.size(); i++)
        {
            auto fc = fcs[i];

            if (msh.is_boundary(fc))
            {
                svel.block(cbs + i * fbs, 0, fbs, 1) = project_function(msh, fc, hdi.face_degree(), velocity, 2);
            }
            else
            {
                auto face_offset = assembler.global_face_offset(msh, fc);
                svel.block(cbs + i*fbs, 0, fbs, 1) = sol.block(face_offset, 0, fbs, 1);
            }
        }
        Matrix<scalar_type, Dynamic, 1> diff_vel = svel - p;
        auto gr = disk::make_hho_stokes(msh, cl, hdi, use_sym_grad);
        Matrix<scalar_type, Dynamic, Dynamic> stab;
        stab = make_vector_hho_stabilization(msh, cl, gr.first, hdi);
        error_vel += diff_vel.dot(factor * (gr.second + stab)*diff_vel);
    }

    //ofs.close();
    return std::make_pair(std::sqrt(error_vel), std::sqrt(error_pres));
}

template<disk::mesh_2D Mesh>
struct test_case_0 {

    using mesh_type = Mesh;
    using point_type = typename mesh_type::point_type;
    using T = typename mesh_type::coordinate_type;

    disk::static_vector<T,2>
    rhs(const point_type& pt) const
    {
        disk::static_vector<T,2> ret;

        auto x1 = pt.x();
        auto x2 = x1 * x1;
        auto y1 = pt.y();
        auto y2 = y1 * y1;

        auto ax =  x2 * (x2 - 2. * x1 + 1.);
        auto ay =  y2 * (y2 - 2. * y1 + 1.);
        auto bx =  x1 * (4. * x2 - 6. * x1 + 2.);
        auto by =  y1 * (4. * y2 - 6. * y1 + 2.);
        auto cx = 12. * x2 - 12.* x1 + 2.;
        auto cy = 12. * y2 - 12.* y1 + 2.;
        auto dx = 24. * x1 - 12.;
        auto dy = 24. * y1 - 12.;

        ret(0) = - cx * by - ax * dy + 5.* x2 * x2;
        ret(1) = + cy * bx + ay * dx + 5.* y2 * y2;

        return ret;
    }

    T psol(const point_type& pt) const
    {
        return std::pow(pt.x(), 5.)  +  std::pow(pt.y(), 5.)  - 1./3.;
    }

    disk::static_vector<T,2>
    vsol(const point_type& pt) const
    {
        disk::static_vector<T,2> ret;

        auto x1 = pt.x();
        auto x2 = x1 * x1;
        auto y1 = pt.y();
        auto y2 = y1 * y1;

        ret(0) =  x2 * (x2 - 2. * x1 + 1.)  * y1 * (4. * y2 - 6. * y1 + 2.);
        ret(1) = -y2 * (y2 - 2. * y1 + 1. ) * x1 * (4. * x2 - 6. * x1 + 2.);

        return ret;
    }
};

template<disk::mesh_2D Mesh>
struct test_case_1 {

    using mesh_type = Mesh;
    using point_type = typename mesh_type::point_type;
    using T = typename mesh_type::coordinate_type;

    disk::static_vector<T,2>
    rhs(const point_type& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        auto ax = std::sin(2.0*M_PI*x)*std::cos(2.0*M_PI*x);
        auto ay = std::sin(2.0*M_PI*y)*std::cos(2.0*M_PI*y);

        auto coscosx = std::pow(std::cos(2.0*M_PI*x), 2);
        auto coscosy = std::pow(std::cos(2.0*M_PI*y), 2);
        auto sinsinx = std::pow(std::sin(2.0*M_PI*x), 2);

        auto sinsiny = std::pow(std::sin(2.0*M_PI*y), 2);
        
        auto d2xu =  4.0*M_PI*M_PI*(1.0 - 2.0*sinsinx)*ay;
        auto d2yu = -8.0*M_PI*M_PI*sinsinx*ay;

        auto d2yv = -4.0*M_PI*M_PI*(1.0 - 2.0*sinsiny)*ax;
        auto d2xv =  8.0*M_PI*M_PI*sinsiny*ax;

        return {-d2xu-d2yu, -d2xv-d2yv};
    }

    T psol(const point_type& pt) const
    {
        return 0.0;
    }

    disk::static_vector<T,2>
    vsol(const point_type& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        auto vx = 0.5*std::pow(std::sin(2.0*M_PI*x),2) *
            std::sin(2.0*M_PI*y) *
            std::cos(2.0*M_PI*y);

        auto vy = -0.5*std::pow(std::sin(2.0*M_PI*y),2) *
            std::sin(2.0*M_PI*x) *
            std::cos(2.0*M_PI*x);


        return {vx, vy};
    }
};

template<disk::mesh_2D Mesh>
struct test_case_2 {

    using mesh_type = Mesh;
    using point_type = typename mesh_type::point_type;
    using T = typename mesh_type::coordinate_type;

    disk::static_vector<T,2>
    rhs(const point_type& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        
        auto fx = -(12./5.)*std::pow(x,5)*std::pow(y,2);
        auto fy = 8.0*std::pow(x,4)*std::pow(y,3) + (12./5.)*std::pow(x,2)*std::pow(y,5);

        return {fx, fy};
    }

    T psol(const point_type& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        return std::pow(x,4)*std::pow(y,4) - (1./25.);
    }

    disk::static_vector<T,2>
    vsol(const point_type& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        auto vx =  0.2*std::pow(x, 5)*std::pow(y, 4);
        auto vy = -0.2*std::pow(x, 4)*std::pow(y, 5);

        return {vx, vy};
    }
};


template<typename Mesh>
auto
run_stokes(const Mesh& msh, size_t degree, bool use_sym_grad = true)
{
    typedef Mesh mesh_type;
    typedef typename mesh_type::cell        cell_type;
    typedef typename mesh_type::face        face_type;
    typedef typename mesh_type::coordinate_type scalar_type;

    typedef disk::dynamic_matrix<scalar_type>     matrix_type;
    typedef disk::vector_boundary_conditions<mesh_type> boundary_type;

    using point_type = typename mesh_type::point_type;

    test_case_2<Mesh> tp;

    auto rhs_fun = [&](const point_type& p) {
        return tp.rhs(p);
    };
    auto velocity = [&](const point_type& p) {
        return tp.vsol(p);
    };
    auto pressure =  [&](const point_type& p) {
        return tp.psol(p);
    };

    typename disk::hho_degree_info hdi(degree + 1, degree);
    boundary_type bnd(msh);
    bnd.addDirichletEverywhere(velocity);

    auto assembler = disk::make_stokes_assembler(msh, hdi, bnd);

    scalar_type factor = (use_sym_grad)? 2. : 1.;

    for (auto cl : msh)
    {
        auto gr = disk::make_hho_stokes(msh, cl, hdi, use_sym_grad);
        Matrix<scalar_type, Dynamic, Dynamic> stab;
        stab = make_vector_hho_stabilization(msh, cl, gr.first, hdi);
        auto dr = make_hho_divergence_reconstruction_rhs(msh, cl, hdi);
        auto cell_basis = disk::make_vector_monomial_basis(msh, cl, hdi.cell_degree());
        auto rhs = make_rhs(msh, cl, cell_basis, rhs_fun);
        assembler.assemble(msh, cl, factor * (gr.second + stab), -dr, rhs);
    }

    assembler.finalize();

    //dump_sparse_matrix(assembler.LHS, "stokes.txt");

    size_t systsz = assembler.LHS.rows();
    size_t nnz = assembler.LHS.nonZeros();

    disk::dynamic_vector<scalar_type> sol = disk::dynamic_vector<scalar_type>::Zero(systsz);

    sol = mumps_lu(assembler.LHS, assembler.RHS);
    //std::ofstream ofs("velocity.dat");

    disk::silo_database db;
    db.create("stokes.silo");
    db.add_mesh(msh, "mesh");

    std::vector<scalar_type> ps;
    using mXD = Matrix<scalar_type, Dynamic, Mesh::dimension>;
    mXD vs = mXD::Zero(msh.cells_size(), Mesh::dimension);
    size_t i = 0;
    for (auto cl : msh) {
        auto v = assembler.take_velocity(msh, cl, sol);
        auto p = assembler.take_pressure(msh, cl, sol);
        ps.push_back(p(0));
        vs.row(i++) = v.template head<Mesh::dimension>();
    }

    db.add_variable("mesh", "p", ps, disk::zonal_variable_t);
    db.add_variable("mesh", "v", vs, disk::zonal_variable_t);
    auto error = compute_errors(msh, sol, hdi, velocity, pressure, assembler, use_sym_grad);

    return error;
}

void convergence_test_typ1(void)
{
    using T = double;
    bool use_sym_grad = false;
    std::vector<std::string> meshfiles;

    /*
    meshfiles.push_back("../../../meshes/2D_triangles/fvca5/mesh1_1.typ1");
    meshfiles.push_back("../../../meshes/2D_triangles/fvca5/mesh1_2.typ1");
    meshfiles.push_back("../../../meshes/2D_triangles/fvca5/mesh1_3.typ1");
    meshfiles.push_back("../../../meshes/2D_triangles/fvca5/mesh1_4.typ1");
    meshfiles.push_back("../../../meshes/2D_triangles/fvca5/mesh1_5.typ1");
    //meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_6.typ1");
    */
    /*
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_1.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_2.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_3.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_4.typ1");
    meshfiles.push_back("../../../diskpp/meshes/2D_quads/fvca5/mesh2_5.typ1");
    */
    
    
    meshfiles.push_back("../../../meshes/2D_hex/fvca5/hexagonal_1.typ1");
    meshfiles.push_back("../../../meshes/2D_hex/fvca5/hexagonal_2.typ1");
    meshfiles.push_back("../../../meshes/2D_hex/fvca5/hexagonal_3.typ1");
    meshfiles.push_back("../../../meshes/2D_hex/fvca5/hexagonal_4.typ1");
    meshfiles.push_back("../../../meshes/2D_hex/fvca5/hexagonal_5.typ1");
    
    std::cout << "                   velocity H1-error";
    std::cout << "    -     pressure L2-error "<< std::endl;

    for (size_t k = 0; k < 5; k++)
    {
        std::cout << "DEGREE " << k << std::endl;

        std::vector<T> mesh_hs;
        std::vector<std::pair<T,T>> errors;

        for (size_t i = 0; i < meshfiles.size(); i++)
        {
            typedef disk::generic_mesh<T, 2>  mesh_type;

            mesh_type msh;
            disk::fvca5_mesh_loader<T, 2> loader;
            if (!loader.read_mesh(meshfiles.at(i)))
            {
                std::cout << "Problem loading mesh." << std::endl;
                continue;
            }
            loader.populate_mesh(msh);

            auto error = run_stokes(msh, k, use_sym_grad);

            mesh_hs.push_back( disk::average_diameter(msh) );
            errors.push_back(error);
        }

        for (size_t i = 0; i < mesh_hs.size(); i++)
        {
            if (i == 0)
            {
                std::cout << "    ";
                std::cout << std::scientific << std::setprecision(4) << mesh_hs.at(i) << "    ";
                std::cout << std::scientific << std::setprecision(4) << errors.at(i).first;
                std::cout << "     -- " << "          ";
                std::cout << std::scientific << std::setprecision(4) << errors.at(i).second;
                std::cout << "     -- " << std::endl;
            }
            else
            {
                auto rate = std::log( errors.at(i).first/errors.at(i-1).first ) /
                            std::log( mesh_hs.at(i)/mesh_hs.at(i-1) );
                std::cout << "    ";
                std::cout << std::scientific  << std::setprecision(4) << mesh_hs.at(i) << "    ";
                std::cout << std::scientific  << std::setprecision(4) << errors.at(i).first << "    ";
                std::cout << std::fixed<< std::setprecision(2) << rate << "          ";

                auto pres_rate = std::log( errors.at(i).second/errors.at(i-1).second ) /
                            std::log( mesh_hs.at(i)/mesh_hs.at(i-1) );
                std::cout << std::scientific  << std::setprecision(4) << errors.at(i).second << "    ";
                std::cout << std::fixed << std::setprecision(2) << pres_rate << std::endl;
            }
        }
    }
}

int main(void)
{
    convergence_test_typ1();
}

#if 0

int main(int argc, char **argv)
{
    using RealType = double;

    char    *filename       = nullptr;
    int ch;
    size_t degree = 1;

    while ( (ch = getopt(argc, argv, "k:")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    if (argc != 1)
    {
        std::cout << "Please specify a 2D mesh" << std::endl;

        return 0;
    }

    filename = argv[0];

    if (std::regex_match(filename, std::regex(".*\\.typ1$") ))
    {
        std::cout << "Guessed mesh format: FVCA5 2D" << std::endl;

        typedef disk::generic_mesh<RealType, 2>  mesh_type;

        mesh_type msh;
        disk::fvca5_mesh_loader<RealType, 2> loader;
        if (!loader.read_mesh(filename))
        {
            std::cout << "Problem loading mesh." << std::endl;
            return 1;
        }
        loader.populate_mesh(msh);

        run_stokes(msh, degree);
    }

    if (std::regex_match(filename, std::regex(".*\\.mesh2d$") ))
    {
        std::cout << "Guessed mesh format: Netgen 2D" << std::endl;

        typedef disk::simplicial_mesh<RealType, 2>  mesh_type;

        mesh_type msh;
        disk::netgen_mesh_loader<RealType, 2> loader;
        if (!loader.read_mesh(filename))
        {
            std::cout << "Problem loading mesh." << std::endl;
            return 1;
        }
        loader.populate_mesh(msh);

        run_stokes(msh, degree);
    }

/*
    if (std::regex_match(filename, std::regex(".*\\.quad$") ))
    {
        std::cout << "Guessed mesh format: Cartesian 2D" << std::endl;

        typedef disk::cartesian_mesh<RealType, 2>   mesh_type;

        mesh_type msh;
        disk::cartesian_mesh_loader<RealType, 2> loader;
        if (!loader.read_mesh(filename))
        {
            std::cout << "Problem loading mesh." << std::endl;
            return 1;
        }
        loader.populate_mesh(msh);

        test_bases(msh);
    }
    */

    return 0;
}
#endif
