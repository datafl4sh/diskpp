#include <iostream>
#include <iomanip>
#include <regex>

#include <sstream>
#include <string>

#include <unistd.h>

#include "diskpp/loaders/loader.hpp"
#include "diskpp/output/silo.hpp"
#include "diskpp/solvers/direct_solvers.hpp"
#include "diskpp/solvers/iterative_solvers.hpp"
#include "diskpp/solvers/eigensolvers.hpp"
#include "diskpp/bases/bases.hpp"
#include "diskpp/methods/hho"
#include "diskpp/bases/bases_new.hpp"
#include "diskpp/methods/hho_assemblers.hpp"
#include "diskpp/methods/hho_slapl.hpp"
#include "diskpp/methods/implementation_hho/curl.hpp"
#include "diskpp/common/timecounter.hpp"
#include "asm.hpp"
#include "minimal_hho.hpp"

template<typename T>
struct robin_data {
    T   alpha;
    T   gR;
};

template<typename Mesh>
struct poisson_data;

template<disk::mesh_3D Mesh>
struct poisson_data<Mesh>
{
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;

    T rhs(const point_type& pt, size_t tag) {

        if (tag == 11011)
            return 10/(0.0015*0.01*0.007);
        
        return 0.0;
    }

    T kappa(const point_type& pt, size_t tag) {

        switch (tag) {
            /* Tab */
            case 11001:
                return 250.0;

            /* Silicon */
            case 11011:
                return 120.0;

            /* Package */
            case 11016:
                return 0.8;

            /* Pins */
            case 11017:
            case 11018:
            case 11019:
            case 11020:
            case 11021:
            case 11022:
                return 150.0;

            /* Screw */
            case 11012:
            case 11013:
            case 11015:
            case 11023:
            case 11024:
                return 20.0;

        }

        return 237.0;
    }

    T dirichlet(const point_type& pt, size_t tag) {
        //if (tag == 143)
        //    return 340.0;

        return 0.0;
    }

    T neumann(const point_type& pt, size_t tag) {
        return -600.0;
    }

    robin_data<T> robin(const point_type&, size_t tag) {
        return {
            .alpha = 15.0,
            .gR = 15.0 * 298.15
        };
    }
};

template<typename Mesh>
struct solver_state
{
    Mesh                    msh;
    size_t                  degree;
    poisson_data<Mesh>      data;
    std::vector<bc>         bcs;
    disk::silo_database     silo;
    bool                    mixed_order;
};


//#define OLD

template<typename Mesh>
auto
solver_minimal(solver_state<Mesh>& state)
{
    using namespace disk::basis;
    using namespace disk::hho::slapl;

    using mesh_type = Mesh;
    using point_type = typename mesh_type::point_type;
    using T = typename hho_space<Mesh>::scalar_type;

    using mat = disk::dynamic_matrix<T>;
    using vec = disk::dynamic_vector<T>;

    degree_info hdi(state.degree+1, state.degree);

    auto [cbs, fbs, rbs] = space_dimensions<hho_space<Mesh>>(hdi);

    const static size_t DIM = Mesh::dimension;

    std::vector<bool> dirfaces;
    dirfaces.resize( state.msh.faces_size() );

    auto df = [&](bc b) {
        return (b == bc::dirichlet) or (b == bc::neumann) or (b == bc::robin);
    };

    std::transform(state.bcs.begin(), state.bcs.end(), dirfaces.begin(), df);

    condensed_assembler assm(state.msh, fbs, dirfaces);

    timecounter tc;

    std::cout << "Assembly: " << std::flush;
    tc.tic();

    std::vector<std::pair<mat, vec>> lcs;
    lcs.reserve( state.msh.cells_size() );

    timecounter tcdetail;
    double time_reco = 0.0;
    double time_stab = 0.0;
    double time_rhs = 0.0;
    double time_schur = 0.0;
    double time_bnd = 0.0;

    for (auto& cl : state.msh) {
        #ifdef OLD
        auto cb = disk::make_scalar_monomial_basis(state.msh, cl, state.degree+1);
        #endif

        auto di = state.msh.domain_info(cl);
        auto kappa = state.data.kappa( barycenter(state.msh, cl), di.tag() );
        #ifdef OLD
        auto [R, A] = hho_minimal_reconstruction(state.msh, cl, state.degree, state.bcs);
        auto S = hho_minimal_stabilization(state.msh, cl, state.degree, state.bcs);
        #else
        tcdetail.tic();
        auto [R, A] = hho_minimal_reconstruction_n(state.msh, cl, state.degree, state.bcs);
        time_reco += tcdetail.toc();
        tcdetail.tic();
        auto S = hho_minimal_stabilization_n(state.msh, cl, state.degree, state.bcs);
        time_stab += tcdetail.toc();
        #endif
    
        auto phiT = hho_space<mesh_type>::cell_basis(state.msh, cl, hdi.cell);

        mat lhs = kappa*(A+S);
        vec rhs = vec::Zero( lhs.rows() );
        auto f = [&](const point_type& pt) {
            return state.data.rhs(pt, di.tag());
        };
        
        tcdetail.tic();
        #ifdef OLD
        auto qps = disk::integrate(state.msh, cl, 2*state.degree+2);
        for (auto& qp : qps) {
            auto phi = cb.eval_functions(qp.point());
            rhs.head(cbs) += qp.weight() * f(qp.point()) * phi;
        }
        #else
        rhs.head(cbs) = integrate(state.msh, cl, f, phiT);
        #endif
        time_rhs += tcdetail.toc();

        tcdetail.tic();
        vec gD_rhs = vec::Zero(A.rows());
        auto fcs = faces(state.msh, cl);
        auto ofs = cbs;
        for (auto& fc : fcs) {
            auto bi = state.msh.boundary_info(fc);
            if (bi.is_boundary() and not bi.is_internal()){
                auto boundary_id = bi.tag();
                auto gofs = offset(state.msh, fc);
         
                #if 0
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
                }

                if (state.bcs[gofs] == bc::neumann) {
                    auto fqps = disk::integrate(
                        state.msh, fc, 2*state.degree);
                    for (const auto& qp : fqps) {
                        auto phi = cb.eval_functions(qp.point());
                        auto gN = state.data.neumann(qp.point(), boundary_id);
                        rhs.head(cbs) += qp.weight() * gN * phi;
                    }
                }
                #endif
                if (state.bcs[gofs] == bc::robin) {
                    auto fqps = disk::integrate(state.msh, fc, 2*hdi.cell);
                    for (const auto& qp : fqps) {
                        #ifdef OLD
                        vec phi = cb.eval_functions( qp.point() );
                        #else
                        vec phi = phiT(qp.point());
                        #endif
                        /* (gR, w)_F */
                        auto [alpha, gR] = state.data.robin(qp.point(), boundary_id);
                        lhs.topLeftCorner(cbs, cbs) += alpha * qp.weight() * phi * phi.transpose();
                        rhs.head(cbs) += qp.weight() * gR * phi;
                    }
                }
            }
            ofs += fbs;
        }
        time_bnd += tcdetail.toc();

        //rhs += -lhs*gD_rhs;
        lcs.push_back({lhs, rhs});
        tcdetail.tic();
        auto [lhsc, rhsc] = disk::hho::schur(lhs, rhs, phiT);
        time_schur += tcdetail.toc();
        assm.assemble(state.msh, cl, lhsc, rhsc);
    }

    std::cout << "Reco: " << time_reco << ", stab: " << time_stab << ", rhs: " << time_rhs;
    std::cout << ", bnd: " << time_bnd << ", schur: " << time_schur << "\n";

    assm.finalize();

    std::cout << tc.toc() << " seconds\n";

    std::cout << "Solve: " << std::endl;
    std::cout << " Unknowns: " << assm.LHS.rows() << " ";
    std::cout << " Nonzeros: " << assm.LHS.nonZeros() << " (";
    std::cout << (100.0*assm.LHS.nonZeros())/(assm.LHS.rows()*assm.LHS.cols());
    std::cout << "%)\n";

    vec sol;
    tc.tic();
    disk::solvers::sparse_lu(assm.LHS, assm.RHS, sol,
        disk::solvers::direct_solver::mumps);

    //sol = vec::Zero(assm.LHS.rows());
    //disk::solvers::iterative_solver_params parms;
    //parms.max_iter = 2*assm.LHS.rows();
    //parms.verbose = true;
    //disk::solvers::inverse_operator_from_matrix precond(assm.PRE);
    //disk::solvers::cg(parms, assm.LHS, assm.RHS, sol, precond);
    
    std::cout << " " << tc.toc() << " seconds\n";

    //auto [evmax, emax] = disk::solvers::powiter(assm.LHS).value();
    //auto [evmin, emin] = disk::solvers::inv_powiter(assm.LHS, 0.0).value();
    //std::cout << "Max eigval: " << emax << ", min eigval: ";
    //std::cout << emin << ", cond: " << emax/emin << "\n";

    std::cout << "Postpro: " << std::flush;
    tc.tic();
    std::vector<T> u_data;
    size_t cell_i = 0;
    T flux = 0.0;
    for (auto& cl : state.msh)
    {
        #ifdef OLD
        auto cb = disk::make_scalar_monomial_basis(state.msh, cl, state.degree+1);
        #endif
        const auto& [lhs, rhs] = lcs[cell_i++];
        auto locsolF = assm.take_local_solution(state.msh, cl, sol);
        auto phiT = hho_space<mesh_type>::cell_basis(state.msh, cl, hdi.cell);   
        vec locsol = disk::hho::deschur(lhs, rhs, locsolF, phiT);
        u_data.push_back(locsol(0));

        auto di = state.msh.domain_info(cl);
        auto kappa = state.data.kappa( barycenter(state.msh, cl), di.tag() );

        auto fcs = faces(state.msh, cl);
        for (auto& fc : fcs) {
            auto bi = state.msh.boundary_info(fc);
            if (bi.is_boundary() and not bi.is_internal()) {
                auto n = normal(state.msh, cl, fc);
                auto fqps = disk::integrate(state.msh, fc, hdi.cell);
                for (auto& qp : fqps) {
                    #ifdef OLD
                    flux += kappa * qp.weight() * (cb.eval_gradients(qp.point()).transpose() * locsol.head(cbs)).dot(n);
                    #else
                    flux += kappa * qp.weight() * (grad(phiT)(qp.point()).transpose() * locsol.head(cbs)).dot(n);
                    #endif
                }
            }
        }
    }
    std::cout << tc.toc() << " seconds\n";

    std::cout << "Boundary flux: " << flux << "\n";

    state.silo.add_variable("mesh", "u_mcs", u_data, disk::zonal_variable_t);
}

template<typename Mesh>
void
solver_std(solver_state<Mesh>& state)
{
    using namespace disk::basis;
    using namespace disk::hho::slapl;

    using mesh_type = Mesh;
    using point_type = typename mesh_type::point_type;
    using T = typename hho_space<Mesh>::scalar_type;

    std::vector<bool> dirfaces;
    dirfaces.resize( state.msh.faces_size() );

    auto df = [&](bc b) {
        return (b == bc::dirichlet);
    };

    std::transform(state.bcs.begin(), state.bcs.end(), dirfaces.begin(), df);

    degree_info hdi;
    
    if (state.mixed_order)
        hdi = degree_info(state.degree+1, state.degree);
    else
        hdi = degree_info(state.degree);

    auto [cbs, fbs, rbs] = space_dimensions<hho_space<Mesh>>(hdi);

    condensed_assembler assm(state.msh, fbs, dirfaces);

    timecounter tcdetail;
    double time_reco = 0.0;
    double time_stab = 0.0;
    double time_rhs = 0.0;
    double time_schur = 0.0;
    double time_bnd = 0.0;

    timecounter tc;

    std::cout << "Assembly: " << std::flush;
    tc.tic();

    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    std::vector<std::pair<mat, vec>> lcs;

    for (auto& cl : state.msh)
    {
        auto di = state.msh.domain_info(cl);
        auto kappa = state.data.kappa( barycenter(state.msh, cl), di.tag() );
        tcdetail.tic();
        auto [R, A] = local_operator(state.msh, cl, hdi);
        time_reco += tcdetail.toc();
        tcdetail.tic();
        mat S;
        if (state.mixed_order)
            S = local_stabilization_hdg(state.msh, cl, hdi);
        else
            S = local_stabilization(state.msh, cl, hdi, R);

        time_stab += tcdetail.toc();
        mat lhs = kappa*(A+S);
        auto phiT = hho_space<mesh_type>::cell_basis(state.msh, cl, hdi.cell);
        tcdetail.tic();
        auto f = [&](const point_type& pt) {
            return state.data.rhs(pt, di.tag());
        };
        vec rhs = vec::Zero( lhs.rows() );
        rhs.head(phiT.size()) = integrate(state.msh, cl, f, phiT);
        time_rhs += tcdetail.toc();

        tcdetail.tic();
        vec gD_rhs = vec::Zero(A.rows());
        auto fcs = faces(state.msh, cl);
        auto ofs = cbs;
        for (auto& fc : fcs) {
            auto phiF = hho_space<mesh_type>::face_basis(state.msh, fc, hdi.face);
            auto bi = state.msh.boundary_info(fc);
            if (bi.is_boundary()){
                auto boundary_id = bi.tag();
                auto gofs = offset(state.msh, fc);
         
                #if 0
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
                }

                if (state.bcs[gofs] == bc::neumann) {
                    auto fqps = disk::integrate(
                        state.msh, fc, 2*state.degree);
                    for (const auto& qp : fqps) {
                        auto phi = cb.eval_functions(qp.point());
                        auto gN = state.data.neumann(qp.point(), boundary_id);
                        rhs.head(cbs) += qp.weight() * gN * phi;
                    }
                }
                #endif

                if (state.bcs[gofs] == bc::robin) {
                    auto fqps = disk::integrate(state.msh, fc, 2*hdi.face);
                    for (const auto& qp : fqps) {
                        vec phi = phiF(qp.point());
                        /* (gR, w)_F */
                        auto [alpha, gR] = state.data.robin(qp.point(), boundary_id);
                        lhs.block(ofs, ofs, fbs, fbs) += alpha * qp.weight() * phi * phi.transpose();
                        rhs.segment(ofs, fbs) += qp.weight() * gR * phi;
                    }
                }
            }
            ofs += fbs;
        }
        time_bnd += tcdetail.toc();
        //rhs += -lhs*gD_rhs;

        lcs.push_back({lhs, rhs});
        tcdetail.tic();
        auto [lhsc, rhsc] = disk::hho::schur(lhs, rhs, phiT);
        time_schur += tcdetail.toc();
        assm.assemble(state.msh, cl, lhsc, rhsc);
    }
    assm.finalize();
    std::cout << "Reco: " << time_reco << ", stab: " << time_stab << ", rhs: " << time_rhs;
    std::cout << ", bnd: " << time_bnd << ", schur: " << time_schur << "\n";
    std::cout << tc.toc() << " seconds\n";

    std::cout << "Solve: " << std::endl;
    std::cout << " Unknowns: " << assm.LHS.rows() << " ";
    std::cout << " Nonzeros: " << assm.LHS.nonZeros() << " (";
    std::cout << (100.0*assm.LHS.nonZeros())/(assm.LHS.rows()*assm.LHS.cols());
    std::cout << "%)\n";

    vec sol;
    tc.tic();
    disk::solvers::sparse_lu(assm.LHS, assm.RHS, sol,
        disk::solvers::direct_solver::mumps);
    //sol = vec::Zero(assm.LHS.rows());
    //disk::solvers::iterative_solver_params parms;
    //parms.max_iter = 2*assm.LHS.rows();
    //parms.verbose = true;
    //disk::solvers::cg(parms, assm.LHS, assm.RHS, sol);
    std::cout << " " << tc.toc() << " seconds\n";

    //auto [evmax, emax] = disk::solvers::powiter(assm.LHS).value();
    //auto [evmin, emin] = disk::solvers::inv_powiter(assm.LHS, 0.0).value();
    //std::cout << "Max eigval: " << emax << ", min eigval: ";
    //std::cout << emin << ", cond: " << emax/emin << "\n";

    std::cout << "Postpro: " << std::flush;
    tc.tic();
    std::vector<T> u_data;
    size_t cell_i = 0;
    T flux = 0.0;
    for (auto& cl : state.msh)
    {
        const auto& [lhs, rhs] = lcs[cell_i++];
        auto locsolF = assm.take_local_solution(state.msh, cl, sol);
        auto phiT = hho_space<mesh_type>::cell_basis(state.msh, cl, hdi.cell);   
        vec locsol = disk::hho::deschur(lhs, rhs, locsolF, phiT);
        u_data.push_back(locsol(0));

        auto di = state.msh.domain_info(cl);
        auto kappa = state.data.kappa( barycenter(state.msh, cl), di.tag() );

        auto fcs = faces(state.msh, cl);
        for (auto& fc : fcs) {
            auto bi = state.msh.boundary_info(fc);
            if (bi.is_boundary() and not bi.is_internal()) {
                auto n = normal(state.msh, cl, fc);
                auto fqps = disk::integrate(state.msh, fc, hdi.cell);
                for (auto& qp : fqps) {
                    flux += kappa * qp.weight() * (grad(phiT)(qp.point()).transpose() * locsol.head(cbs)).dot(n);
                }
            }
        }
    }
    std::cout << tc.toc() << " seconds\n";

    std::cout << "Boundary flux: " << flux << "\n";

    state.silo.add_variable("mesh", "u_std", u_data, disk::zonal_variable_t);

}

template<disk::mesh_3D Mesh>
void init_problem(solver_state<Mesh>& state)
{
    using point_type = typename Mesh::point_type;
    state.msh.transform( [](const point_type& pt) {
            return pt * 0.001;
        }
    );

    state.bcs.resize( state.msh.faces_size(), bc::none );

    size_t bndfcs = 0;

    auto surface = 0.0;
    auto volume = 0.0;

    for (auto& fc : faces(state.msh)) {
        auto bi = state.msh.boundary_info(fc);
        if ( bi.is_boundary() and not bi.is_internal() ) {
            auto gofs = offset(state.msh, fc); 
            state.bcs[ gofs ] = bc::robin;
            
            bndfcs++;
            surface += measure(state.msh, fc);
        }
    }

    for (auto& cl : state.msh) {
        volume += measure(state.msh, cl);
    }

    std::cout << "bnd: " << bndfcs << ", tot: " << state.msh.faces_size();
    std::cout << ", ratio: " << (100.0*bndfcs)/state.msh.faces_size();
    std::cout << "%" << std::endl;

    std::cout << "Surface: " << surface << ", volume: " << volume;
    std::cout << ", ratio (s/v) = " << surface/volume << std::endl;

    state.silo.create("hs.silo");
    state.silo.add_mesh(state.msh, "mesh");
}

int main(int argc, const char *argv[])
{
    using T = double;

    if (argc != 3)
    {
        std::cout << argv[0] << " <mesh_file> <degree>" << std::endl;
        return 1;
    }

    const char *mesh_filename = argv[1];
    size_t degree = std::stoull(argv[2]);

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
        state.mixed_order = false;
        state.degree = degree;
        std::cout << "### NO-BOUNDARY\n";
        solver_minimal(state);
        std::cout << "### STD EQUAL ORDER\n";
        solver_std(state);
        std::cout << "### STD MIXED ORDER\n";
        state.mixed_order = true;
        solver_std(state);
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
        state.degree = degree;
        solver_minimal(state);
        solver_std(state);
        return 0;
    }

    std::cerr << "Didn't match any known mesh type\n";

    return 1;
}
