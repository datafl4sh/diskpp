#include <iostream>
#include <iomanip>
#include <regex>

#include <sstream>
#include <string>

#include <unistd.h>

#include "diskpp/loaders/loader.hpp"
#include "diskpp/output/silo.hpp"
#include "diskpp/solvers/direct_solvers.hpp"
#include "diskpp/bases/bases.hpp"
#include "diskpp/methods/hho"
#include "diskpp/methods/implementation_hho/curl.hpp"
#include "asm.hpp"
#include "minimal_hho.hpp"


template<typename Mesh>
struct poisson_data;

template<disk::mesh_3D Mesh>
struct poisson_data<Mesh>
{
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;

    T rhs(const point_type& pt, size_t tag) {
        return 0.0;
    }

    T dirichlet(const point_type& pt, size_t tag) {
        if (tag == 143)
            return 340.0;

        return 0.0;
    }

    T neumann(const point_type& pt, size_t tag) {
        return 0.0;
    }

    T robin(const point_type&, size_t tag) {
        return 0.0;
    }
};

template<typename Mesh>
struct solver_state
{
    Mesh                    msh;
    size_t                  degree;
    poisson_data<Mesh>      data;
    std::vector<bc>         bcs;
};

template<typename Mesh>
auto
solver(solver_state<Mesh>& state)
{
    using scalar_type = typename Mesh::coordinate_type;
    using mat = disk::dynamic_matrix<scalar_type>;
    using vec = disk::dynamic_vector<scalar_type>;

    disk::hho_degree_info hdi(state.degree+1, state.degree);

    const static size_t DIM = Mesh::dimension;
    auto cbs = disk::scalar_basis_size(state.degree+1, DIM);
    auto fbs = disk::scalar_basis_size(state.degree, DIM-1);

    std::vector<bool> dirfaces;
    dirfaces.resize( state.msh.faces_size() );

    auto df = [&](bc b) {
        return (b == bc::dirichlet) or (b == bc::neumann) or (b == bc::robin);
    };

    std::transform(state.bcs.begin(), state.bcs.end(), dirfaces.begin(), df);

    condensed_assembler assm(state.msh, fbs, dirfaces);

    //timecounter tc;
    //tc.tic();

    vec global_dirichlet_data = vec::Zero( fbs * state.msh.faces_size() );

    std::vector<std::pair<mat, vec>> lcs;

    std::cout << "ASM" << std::endl;
    for (auto& cl : state.msh) {
        auto cb = disk::make_scalar_monomial_basis(state.msh, cl, state.degree+1);

        auto [R, A] = hho_minimal_reconstruction(state.msh, cl,
            state.degree, 1.0, state.bcs
        );

        auto S = hho_minimal_stabilization(state.msh, cl,
            state.degree, state.bcs);
        
        mat lhs = 237.0*A+S;
        vec rhs = vec::Zero( lhs.rows() );

        vec gD_rhs = vec::Zero(A.rows());
        auto fcs = faces(state.msh, cl);
        auto ofs = cbs;
        for (auto& fc : fcs) {
            auto bi = state.msh.boundary_info(fc);
            if (bi.is_boundary()){
                auto boundary_id = bi.tag();
                auto gofs = offset(state.msh, fc);
         
                if ( state.bcs[gofs] == bc::dirichlet ) {
                    auto fb = disk::make_scalar_monomial_basis(
                        state.msh, fc, state.degree);
                    auto fqps = disk::integrate(
                        state.msh, fc, 2*state.degree);
                    mat M = mat::Zero(fb.size(), fb.size());
                    vec f_gD = vec::Zero(fb.size());
                    for (const auto& qp : fqps) {
                        auto phi = fb.eval_functions(qp.point());
                        M += qp.weight() * phi * phi.transpose();
                        f_gD +=
                            qp.weight() * state.data.dirichlet(
                                qp.point(), boundary_id) * phi;
                    }
                    gD_rhs.segment(ofs, fbs) = M.ldlt().solve(f_gD);
                    global_dirichlet_data.segment(gofs, fbs) =
                        gD_rhs.segment(ofs, fbs);
                }

                if (state.bcs[gofs] == bc::neumann) {
                    auto fqps = disk::integrate(
                        state.msh, fc, 2*state.degree);
                    for (const auto& qp : fqps) {
                        auto phi = cb.eval_functions(qp.point());
                        //auto gN_val = gN(qp.point());
                        /* (gN, w)_F */
                        rhs.head(cbs) += qp.weight() * (-100.0) * phi;
                    }
                }
            }
            ofs += fbs;
        }
        rhs += -lhs*gD_rhs;
        lcs.push_back({lhs, rhs});
        auto [Lc, Rc] = disk::static_condensation(lhs, rhs, cbs);
        assm.assemble(state.msh, cl, Lc, Rc);
    }

    assm.finalize();

    std::cout << "SOLVE" << std::endl;
    std::cout << " Unknowns: " << assm.LHS.rows() << " ";
    std::cout << " Nonzeros: " << assm.LHS.nonZeros() << std::endl;
    vec sol;
    disk::solvers::sparse_lu(assm.LHS, assm.RHS, sol);

    std::cout << "POSTPRO" << std::endl;
    std::vector<scalar_type> u_data;
    size_t cell_i = 0;
    for (auto& cl : state.msh)
    {
        
        const auto& [lhs, rhs] = lcs[cell_i++];
        auto locsolF = assm.take_local_solution(state.msh, cl, sol);
        
        auto fcs = faces(state.msh, cl);
        auto ofs = 0;
        for (auto& fc : fcs) {
            auto gofs = offset(state.msh, fc);
            //locsolF.segment(ofs, fbs) +=
            //    global_dirichlet_data.segment(gofs, fbs);
            ofs += fbs;
        }
        
        disk::dynamic_vector<scalar_type> locsol =
            disk::static_decondensation(lhs, rhs, locsolF);
        u_data.push_back(locsol(0));
    }

    disk::silo_database silo;
    silo.create("hs.silo");
    silo.add_mesh(state.msh, "mesh");
    silo.add_variable("mesh", "u", u_data, disk::zonal_variable_t);
}

template<disk::mesh_3D Mesh>
void init_problem(solver_state<Mesh>& state)
{
    using point_type = typename Mesh::point_type;
    state.msh.transform( [](const point_type& pt) {
            return pt * 0.001;
        }
    );

    state.degree = 0;
    state.bcs.resize( state.msh.faces_size(), bc::none );

    for (auto& fc : faces(state.msh)) {
        auto bi = state.msh.boundary_info(fc);
        if ( bi.is_boundary() ) {
            auto gofs = offset(state.msh, fc); 
            state.bcs[ gofs ] = bc::neumann;
            
            if ( (bi.tag() == 143) ) {
                auto bar = barycenter(state.msh, fc);
                if ( bar.x() < 0.005 && bar.x() > -0.005 && bar.y()<0.020 && bar.y() > 0.010)
                    state.bcs[ gofs ] = bc::dirichlet;
            }
        }
    }
}

int main(int argc, const char *argv[])
{
    using T = double;

    if (argc != 2)
    {
        std::cout << argv[0] << " <mesh_file>" << std::endl;
        return 1;
    }

    const char *mesh_filename = argv[1];

    #if 0
    if (std::regex_match(mesh_filename, std::regex(".*\\.geo2s$") ))
    {
        std::cout << "Guessed mesh format: GMSH 2D simplicials" << std::endl;
        using mesh_type = disk::triangular_mesh<T>;
        disk::gmsh_geometry_loader< mesh_type > loader;
        loader.read_mesh(mesh_filename);
        loader.populate_mesh(msh);
        return 0;
    }
    #endif

    if (std::regex_match(mesh_filename, std::regex(".*\\.geo3s$") ))
    {
        std::cout << "Guessed mesh format: GMSH 3D simplicials" << std::endl;
        using mesh_type = disk::tetrahedral_mesh<T>;
        solver_state<mesh_type> state;
        disk::gmsh_geometry_loader< mesh_type > loader;
        loader.read_mesh(mesh_filename);
        loader.populate_mesh(state.msh);
        init_problem(state);
        solver(state);
        return 0;
    }

    if (std::regex_match(mesh_filename, std::regex(".*\\.geo3g$") ))
    {
        std::cout << "Guessed mesh format: GMSH 3D generic" << std::endl;
        using mesh_type = disk::generic_mesh<T,3>;
        solver_state<mesh_type> state;
        disk::gmsh_geometry_loader< mesh_type > loader;
        loader.read_mesh(mesh_filename);
        loader.populate_mesh(state.msh);
        init_problem(state);
        solver(state);
        return 0;
    }

    std::cerr << "Didn't match any known mesh type\n";

    return 1;
}
