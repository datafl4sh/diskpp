/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *
 * Matteo Cicuttin (C) 2025
 * matteo.cicuttin@polito.it
 *
 * Politecnico di Torino - DISMA
 * Dipartimento di Matematica
 */

#include <iostream>

#include "diskpp/mesh/mesh.hpp"
#include "diskpp/mesh/meshgen.hpp"
#include "diskpp/bases/bases.hpp"
#include "diskpp/methods/hho"
#include "diskpp/methods/implementation_hho/curl.hpp"
#include "diskpp/methods/hho_slapl.hpp"
#include "diskpp/methods/hho_assemblers.hpp"
#include "diskpp/solvers/direct_solvers.hpp"
#include "diskpp/common/timecounter.hpp"
#include "diskpp/output/silo.hpp"
#include "asm.hpp"
#include "minimal_hho.hpp"

/***************************************************************
 * Sources for the test problems
 */
template<typename Mesh>
struct source_functor;

template<disk::mesh_2D Mesh>
struct source_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        //return 2.0*M_PI*M_PI*sx*sy;
        return M_PI*M_PI*sx;
    }
};

template<disk::mesh_3D Mesh>
struct source_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        auto sz = std::sin(M_PI*pt.z());
        return 3.0*M_PI*M_PI*sx*sy*sz;
    }
};


/***************************************************************
 * Analytical solutions for the test problems
 */
template<typename Mesh>
struct solution_functor;

template<disk::mesh_2D Mesh>
struct solution_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        //return sx*sy;
        return sx;
    }
};

template<disk::mesh_3D Mesh>
struct solution_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        auto sz = std::sin(M_PI*pt.z());
        return sx*sy*sz;
    }
};

/***************************************************************
 * Helpers
 */
template<typename Mesh>
auto make_rhs_function(const Mesh& msh)
{
    return source_functor<Mesh>();
}

template<typename Mesh>
auto make_solution_function(const Mesh& msh)
{
    return solution_functor<Mesh>();
}




template<typename Mesh>
auto
minimal_hho_solver(const Mesh& msh, size_t degree, const std::vector<bc>& bcs)
{
    bool compute_cond = true;
    using scalar_type = typename Mesh::coordinate_type;

    scalar_type eta = 1.0;

    auto zerofun = [](const typename Mesh::point_type&) {
        return 0.0;
    };

    auto onefun = [](const typename Mesh::point_type&) {
        return 1.0;
    };

    auto u = [](const typename Mesh::point_type& pt) {
        auto x = pt.x();
        auto y = pt.y();
        return std::exp(x) * std::sin(M_PI*y) + x*x + y;
        //return x*(1-x)*y;
        //return std::cos(M_PI*x);
    };

    auto f = [](const typename Mesh::point_type& pt) {
        auto x = pt.x();
        auto y = pt.y();
        return (M_PI*M_PI - 1) * std::exp(x)*std::sin(M_PI*y) - 2;
        //return 2*y;
        //return M_PI*M_PI*std::cos(M_PI*x);
    };    

    auto gD = [&](const typename Mesh::point_type& pt) {
        return u(pt);
    };

    auto gN = [](const typename Mesh::point_type& pt) {
        auto x = pt.x();
        auto y = pt.y();
        return M_E * std::sin(M_PI*y) + 2;
        //return x*(1-x);
    };

    auto gR = [](const typename Mesh::point_type& pt) {
        auto x = pt.x();
        auto y = pt.y();
        return (-M_PI*std::exp(x) - 1.0 + x*x);
    };


    disk::hho::slapl::degree_info di(degree+1, degree);

    const static size_t DIM = Mesh::dimension;
    auto cbs = disk::scalar_basis_size(degree+1, DIM);
    auto fbs = disk::scalar_basis_size(degree, DIM-1);

    std::vector<bool> dirfaces;
    dirfaces.resize( bcs.size() );

    auto df = [&](bc b) {
        return (b == bc::dirichlet) or (b == bc::neumann) or (b == bc::robin);
    };

    std::transform(bcs.begin(), bcs.end(), dirfaces.begin(), df);

    condensed_assembler assm(msh, fbs, dirfaces);


    timecounter tc;
    tc.tic();

    using MT = disk::dynamic_matrix<scalar_type>;
    using VT = disk::dynamic_vector<scalar_type>;
    std::vector<std::pair<MT, VT>> lcs;

    auto rhsfun = make_rhs_function(msh);

    for (auto& cl : msh) {
        auto [R, A] = hho_minimal_reconstruction(msh, cl, degree, eta, bcs);
        auto S = hho_minimal_stabilization(msh, cl, degree, bcs);
        disk::dynamic_matrix<scalar_type> lhs = A+S;

        disk::dynamic_vector<scalar_type> rhs =
            hho_minimal_rhs(msh, cl,
                f,     // source
                gN,    // neumann
                gR,    // robin
                degree, eta, bcs);


        disk::dynamic_vector<scalar_type> gD_rhs =
            disk::dynamic_vector<scalar_type>::Zero(A.rows());
        auto fcs = faces(msh, cl);
        auto ofs = cbs;
        for (auto& fc : fcs) {
            auto bi = msh.boundary_info(fc);
            if (bi.is_boundary()){
                auto boundary_id = bi.id();
                if ( bcs[offset(msh, fc)] == bc::dirichlet ) {
                    auto fb = disk::make_scalar_monomial_basis(msh, fc, degree);
                    auto fqps = disk::integrate(msh, fc, 2*degree);
                    disk::dynamic_matrix<scalar_type> M =
                        disk::dynamic_matrix<scalar_type>::Zero(fb.size(), fb.size());
                    disk::dynamic_vector<scalar_type> f_gD =
                        disk::dynamic_vector<scalar_type>::Zero(fb.size());
                    for (const auto& qp : fqps) {
                        auto phi = fb.eval_functions(qp.point());
                        M += qp.weight() * phi * phi.transpose();
                        f_gD += qp.weight() * gD(qp.point()) * phi;
                    }
                    gD_rhs.segment(ofs, fbs) += M.ldlt().solve(f_gD);
                }
            }
            ofs += fbs;
        }

        rhs += -lhs*gD_rhs;

        lcs.push_back({lhs, rhs});
        
        auto cbs = disk::scalar_basis_size(degree+1, Mesh::dimension);
        auto [Lc, Rc] = disk::static_condensation(lhs, rhs, cbs);
    
        assm.assemble(msh, cl, Lc, Rc);
    }
    assm.finalize();

    std::cout << "************" << std::endl;
    //std::cout << " Assembly time: " << tc.toc() << std::endl;
    auto bfsize = msh.faces_size() - msh.boundary_faces_size();
    std::cout << " Internal faces:    " << bfsize << ", fbs = " << fbs;
    std::cout << ", intfaces*fbs = " << bfsize * fbs << std::endl;
    std::cout << " Unknowns: " << assm.LHS.rows() << " ";
    std::cout << " Nonzeros: " << assm.LHS.nonZeros() << std::endl;
    tc.tic();
    std::cout << "  Solver: " << std::flush;
    disk::dynamic_vector<scalar_type> sol;
    disk::solvers::sparse_lu(assm.LHS, assm.RHS, sol);
    //std::cout << " Solver time: " << tc.toc() << std::endl;
    
    std::vector<scalar_type> u_data;
    std::vector<scalar_type> uex_data;
    std::vector<scalar_type> ae_data;
    std::vector<scalar_type> conditioning;
    auto solfun = make_solution_function(msh);

    scalar_type L2error = 0.0;
    scalar_type Aerror = 0.0;
    auto u_sol = make_solution_function(msh);
    tc.tic();
    size_t cell_i = 0;
    for (auto& cl : msh)
    {
        const auto& [lhs, rhs] = lcs[cell_i++];
        auto locsolF = assm.take_local_solution(msh, cl, sol);
        auto cbs = disk::scalar_basis_size(degree+1, Mesh::dimension);
        disk::dynamic_vector<scalar_type> locsol =
            disk::static_decondensation(lhs, rhs, locsolF);
        u_data.push_back(locsol(0));
        uex_data.push_back( u(barycenter(msh,cl)) );

        disk::dynamic_vector<scalar_type> ana_sol =
            disk::project_function(msh, cl, degree+1, u);

        disk::dynamic_vector<scalar_type> diff = ana_sol - locsol.head(cbs);
        disk::dynamic_vector<scalar_type> Iu = disk::project_function(msh, cl, disk::hho_degree_info(di.cell, di.face), u);

        auto fcs = faces(msh, cl);
        auto ofs = cbs;
        for (auto& fc : fcs) {
            auto bi = msh.boundary_info(fc);
            if (bi.is_boundary()){
                auto boundary_id = bi.id();
                if ( bcs[offset(msh, fc)] == bc::dirichlet ) {
                    auto fb = disk::make_scalar_monomial_basis(msh, fc, degree);
                    auto fqps = disk::integrate(msh, fc, 2*degree);
                    disk::dynamic_matrix<scalar_type> M =
                        disk::dynamic_matrix<scalar_type>::Zero(fb.size(), fb.size());
                    disk::dynamic_vector<scalar_type> f_gD =
                        disk::dynamic_vector<scalar_type>::Zero(fb.size());
                    for (const auto& qp : fqps) {
                        auto phi = fb.eval_functions(qp.point());
                        M += qp.weight() * phi * phi.transpose();
                        f_gD += qp.weight() * gD(qp.point()) * phi;
                    }
                    locsol.segment(ofs, fbs) += M.ldlt().solve(f_gD);
                }
            }
            ofs += fbs;
        }

        auto cb = disk::make_scalar_monomial_basis(msh, cl, degree+1);
        disk::dynamic_matrix<scalar_type> mass = disk::make_mass_matrix(msh, cl, cb);

        if (compute_cond) {
            conditioning.push_back( cond(lhs, 1) );
        }

        L2error += diff.dot(mass*diff);
        auto ae = (Iu - locsol).dot(lhs*(Iu - locsol));
        ae_data.push_back(ae);
        Aerror += ae;
    }
    //std::cout << " Postpro time: " << tc.toc() << std::endl;
    //std::cout << " L2-norm error: " << std::sqrt(L2error) << std::endl;

    disk::silo_database silo;
    silo.create("nitsche.silo");
    silo.add_mesh(msh, "mesh");
    silo.add_variable("mesh", "u", u_data, disk::zonal_variable_t);
    silo.add_variable("mesh", "u_ex", uex_data, disk::zonal_variable_t);
    silo.add_variable("mesh", "a_err", ae_data, disk::zonal_variable_t);
    if (compute_cond) {
        silo.add_variable("mesh", "cond", conditioning, disk::zonal_variable_t);
    }

    return std::pair{std::sqrt(L2error), std::sqrt(Aerror)};
}

int main(void)
{
    using T = double;
    using mesh_type = disk::simplicial_mesh<T,2>;


    for (size_t k = 0; k < 3; k++) {
        mesh_type msh;
        auto mesher = make_simple_mesher(msh);
        disk::renumber_hypercube_boundaries(msh);
        
        auto prev_L2err = 0.0;
        auto prev_Aerr = 0.0;
        auto prev_h = 0.0;

        std::cout << "Minimal-HHO(k+1, k), k = " << k << std::endl;
        for (size_t i = 0; i < 5; i++) {
            mesher.refine();
            std::vector<bc> bcs;
            set_boundary(msh, bcs, bc::robin, 0);
            set_boundary(msh, bcs, bc::neumann, 1);
            set_boundary(msh, bcs, bc::dirichlet, 2);
            set_boundary(msh, bcs, bc::dirichlet, 3);
            auto [L2err, Aerr] = minimal_hho_solver(msh, k, bcs);
            auto h = disk::average_diameter(msh);

            if (i == 0) {
                std::cout << "  h = " << h << ", L2err = " << L2err << ", Aerr = " << Aerr << std::endl;
            }
            else {
                auto L2rate = std::log(prev_L2err/L2err)/std::log(prev_h/h);
                auto Arate = std::log(prev_Aerr/Aerr)/std::log(prev_h/h);
                std::cout << "  h = " << h << ", L2err = " << L2err << ", L2 rate = " << L2rate << ", Aerr = " << Aerr  << ", A rate = " << Arate << std::endl;
            }
            prev_h = h;
            prev_L2err = L2err;
            prev_Aerr = Aerr;
        }
    }

    return 0;
}