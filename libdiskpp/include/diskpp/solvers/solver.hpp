/*
 *       /\        Matteo Cicuttin (C) 2016, 2017
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
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

#include "diskpp/common/eigen.hpp"
#include "sol/sol.hpp"

#ifdef HAVE_AGMG
#include "agmg.hpp"
#endif

#ifdef HAVE_INTEL_MKL
    //#include "feast.hpp"
#endif

#include "../contrib/solver_wrappers/iface_mumps/include/mumps.hpp"

namespace disk
{
namespace solvers
{

enum LinearSolverType {
    PARDISO_LU,
    PARDISO_LDLT,
    MUMPS_LU,
    MUMPS_LDLT,
    SPARSE_LU,
    CG,
    BICGSTAB,
};

void
init_lua(sol::state& lua)
{
    lua["solver"]       = lua.create_table();
    lua["solver"]["cg"] = lua.create_table();

#ifdef HAVE_INTEL_MKL
    lua["solver"]["pardiso"] = lua.create_table();
#endif
}

template<typename T>
struct conjugated_gradient_params
{
    T           rr_tol;
    T           rr_max;
    size_t      max_iter;
    bool        verbose;
    bool        save_iteration_history;
    bool        use_initial_guess;
    std::string history_filename;

    conjugated_gradient_params() :
      rr_tol(1e-8), rr_max(20), max_iter(100), verbose(false), save_iteration_history(false), use_initial_guess(false)
    {
    }
};

// TODO: return false and some kind of error in case of non convergence.
template<typename T>
bool
conjugated_gradient(const conjugated_gradient_params<T>&       cgp,
                    const Eigen::SparseMatrix<T>&              A,
                    const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
                    Eigen::Matrix<T, Eigen::Dynamic, 1>&       x)
{
    if (A.rows() != A.cols())
    {
        if (cgp.verbose)
            std::cout << "[CG solver] A square matrix is required" << std::endl;

        return false;
    }

    size_t N = A.cols();

    if (b.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of RHS vector" << std::endl;

        return false;
    }

    if (x.size() != N)
    {
        if (cgp.verbose)
            std::cout << "[CG solver] Wrong size of solution vector" << std::endl;

        return false;
    }

    if (!cgp.use_initial_guess)
        x = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(N);

    size_t iter = 0;
    T      nr, nr0;
    T      alpha, beta, rho;

    Eigen::Matrix<T, Eigen::Dynamic, 1> d(N), r(N), r0(N), y(N);

    r0 = d = r = b - A * x;
    nr = nr0 = r.norm();

    std::ofstream iter_hist_ofs;
    if (cgp.save_iteration_history)
        iter_hist_ofs.open(cgp.history_filename);

    auto max_iter = cgp.max_iter == 0 ? 2 * N : cgp.max_iter;

    while (nr / nr0 > cgp.rr_tol && iter < max_iter && nr / nr0 < cgp.rr_max)
    {
        if (cgp.verbose)
        {
            std::cout << "                                                 \r";
            std::cout << " -> Iteration " << iter << ", rr = ";
            std::cout << nr / nr0 << "\b\r";
            std::cout.flush();
        }

        if (cgp.save_iteration_history)
            iter_hist_ofs << nr / nr0 << std::endl;

        y     = A * d;
        rho   = r.dot(r);
        alpha = rho / d.dot(y);
        x     = x + alpha * d;
        r     = r - alpha * y;
        beta  = r.dot(r) / rho;
        d     = r + beta * d;

        nr = r.norm();
        iter++;
    }

    if (cgp.save_iteration_history)
    {
        iter_hist_ofs << nr / nr0 << std::endl;
        iter_hist_ofs.close();
    }

    if (cgp.verbose)
        std::cout << " -> Iteration " << iter << ", rr = " << nr / nr0 << std::endl;

    return true;
}

template<typename T>
bool
conjugated_gradient(sol::state&                                lua,
                    const Eigen::SparseMatrix<T>&              A,
                    const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
                    Eigen::Matrix<T, Eigen::Dynamic, 1>&       x)
{
    conjugated_gradient_params<T> cgp;

    cgp.max_iter = lua["solver"]["cg"]["max_iter"].get_or(1000);
    cgp.rr_tol   = lua["sover"]["cg"]["rr_tol"].get_or(1e-8);
    cgp.rr_max   = lua["solver"]["cg"]["rr_max"].get_or(20.0);
    cgp.verbose  = lua["solver"]["cg"]["verbose"].get_or(false);

    auto iter_hist_filename = lua["solver"]["cg"]["hist_file"];
    if (iter_hist_filename.valid())
    {
        cgp.save_iteration_history = true;
        cgp.history_filename       = iter_hist_filename;
    }

    return conjugated_gradient(cgp, A, b, x);
}

#ifdef HAVE_INTEL_MKL

#define PARDISO_IN_CORE 0
#define PARDISO_OUT_OF_CORE_IF_NEEDED 1
#define PARDISO_OUT_OF_CORE_ALWAYS 2

template<typename T>
struct pardiso_params
{
    bool report_factorization_Mflops;
    int  out_of_core; // 0: IC, 1: IC, OOC if limits passed, 2: OOC
    int  mflops;

    pardiso_params() : report_factorization_Mflops(false), out_of_core(0), mflops(0) {}
};

template<typename T>
bool
mkl_pardiso(pardiso_params<T>&                         params,
            const Eigen::SparseMatrix<T>&              A,
            const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
            Eigen::Matrix<T, Eigen::Dynamic, 1>&       x)
{
    Eigen::PardisoLU<Eigen::SparseMatrix<T>> solver;

    if (params.out_of_core >= 0 && params.out_of_core <= 2)
        solver.pardisoParameterArray()[59] = params.out_of_core;

    if (params.report_factorization_Mflops)
        solver.pardisoParameterArray()[18] = -1; // report flops

    solver.analyzePattern(A);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: analyzePattern failed" << std::endl;
        return false;
    }

    solver.factorize(A);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: Could not factorize the matrix" << std::endl;
        std::cerr << "Try to tweak MKL_PARDISO_OOC_MAX_CORE_SIZE" << std::endl;
        return false;
    }

    x = solver.solve(b);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: Could not solve the linear system" << std::endl;
        return false;
    }

    if (params.report_factorization_Mflops)
    {
        int mflops    = solver.pardisoParameterArray()[18];
        params.mflops = mflops;
        std::cout << "[PARDISO] Factorization Mflops: " << mflops << std::endl;
    }

    return true;
}

template<typename T>
bool
mkl_pardiso_ldlt(const pardiso_params<T>&                   params,
                 const Eigen::SparseMatrix<T>&              A,
                 const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
                 Eigen::Matrix<T, Eigen::Dynamic, 1>&       x)
{
    Eigen::PardisoLDLT<Eigen::SparseMatrix<T>> solver;

    if (params.out_of_core >= 0 && params.out_of_core <= 2)
        solver.pardisoParameterArray()[59] = params.out_of_core;

    if (params.report_factorization_Mflops)
        solver.pardisoParameterArray()[18] = -1; // report flops

    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: Could not factorize the matrix" << std::endl;
        return false;
    }

    x = solver.solve(b);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "ERROR: Could not solve the linear system" << std::endl;
        return false;
    }

    if (params.report_factorization_Mflops)
    {
        int mflops = solver.pardisoParameterArray()[18];
        std::cout << "[PARDISO] Factorization Mflops: " << mflops << std::endl;
    }

    return true;
}

template<typename T>
bool
mkl_pardiso(sol::state&                                lua,
            const Eigen::SparseMatrix<T>&              A,
            const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
            Eigen::Matrix<T, Eigen::Dynamic, 1>&       x)
{
    pardiso_params<T> pparams;

    pparams.report_factorization_Mflops = lua["solver"]["pardiso"]["print_mflops"].get_or(false);
    pparams.out_of_core                 = lua["solver"]["pardiso"]["out_of_core"].get_or(0);

    return mkl_pardiso(pparams, A, b, x);
}

#endif /* HAVE_INTEL_MKL */

template<typename T>
bool
linear_solver(sol::state&                          lua,
              Eigen::SparseMatrix<T>&              A,
              Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
              Eigen::Matrix<T, Eigen::Dynamic, 1>& x)
{
    std::string solver_type = lua["solver"]["solver_type"].get_or(std::string("cg"));

    if (solver_type == "cg"){
        return conjugated_gradient(lua, A, b, x);
    }
    else if (solver_type == "pardiso")
    {
#ifdef HAVE_INTEL_MKL
        return mkl_pardiso(lua, A, b, x);
#else
        throw std::runtime_error("Pardiso is not installed");
#endif /* HAVE_INTEL_MKL */
    }
    else if (solver_type == "agmg")
    {
#ifdef HAVE_AGMG
        return agmg_multigrid_solver(lua, A, b, x);
#else
        throw std::runtime_error("AGMG is not installed");
#endif /* HAVE_AGMG */
    }

    return false;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> linear_solver(const LinearSolverType &type,
                                                  Eigen::SparseMatrix<T> &A,
                                                  Eigen::Matrix<T, Eigen::Dynamic, 1> &b) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> x;
    bool sucess = false;

    switch (type) {
    case LinearSolverType::PARDISO_LU: {
        pardiso_params<T> pparams;
#ifdef HAVE_INTEL_MKL
        sucess = mkl_pardiso(pparams, A, b, x);
#else
        throw std::runtime_error("Pardiso is not installed");
#endif /* HAVE_INTEL_MKL */
        break;
    }
    case LinearSolverType::PARDISO_LDLT: {
        pardiso_params<T> pparams;
#ifdef HAVE_INTEL_MKL
        sucess = mkl_pardiso_ldlt(pparams, A, b, x);
#else
        throw std::runtime_error("Pardiso is not installed");
#endif /* HAVE_INTEL_MKL */
        break;
    }
    case LinearSolverType::MUMPS_LU: {
#ifdef HAVE_MUMPS
        x = mumps_lu(A, b);
        sucess = true;
#else
        throw std::runtime_error("Mumps is not installed");
#endif /* HAVE_INTEL_MKL */
        break;
    }
    case LinearSolverType::MUMPS_LDLT: {
#ifdef HAVE_MUMPS
        x = mumps_ldlt(A, b);
        sucess = true;
#else
        throw std::runtime_error("Mumps is not installed");
#endif /* HAVE_INTEL_MKL */
        break;
    }
    case LinearSolverType::SPARSE_LU: {
        Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        x = solver.solve(b);
        sucess = solver.info() == Eigen::Success;
        break;
    }
    case LinearSolverType::CG: {
        conjugated_gradient_params<T> cg_params;

        sucess = conjugated_gradient(cg_params, A, b, x);
        break;
    }
    case LinearSolverType::BICGSTAB: {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<T>> solver;
        solver.factorize(A);
        x = solver.solve(b);
        sucess = solver.info() == Eigen::Success;
        break;
    }
    default:
        throw std::runtime_error("Unexpected LinearSolver.");
        break;
    }

    if (!sucess) {
        throw std::runtime_error("Fail to solve the linear solver.");
    }

    return x;
}

template < typename T >
class LinearSolver {
    typedef typename Eigen::SparseMatrix< T > sparse_type;
    typedef typename Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > matrix_type;
    typedef typename Eigen::Matrix< T, Eigen::Dynamic, 1 > vector_type;

    // used to impose BC dirichlet
    std::shared_ptr< std::map< int, matrix_type > > m_lhs_loc;
    bool m_empty;
    LinearSolverType m_type;

    Eigen::PardisoLU< sparse_type > m_pardisoLU;
    Eigen::PardisoLDLT< sparse_type > m_pardisoLDLT;
    Eigen::SparseLU< sparse_type > m_sparseLU;
    Eigen::BiCGSTAB< sparse_type > m_bicg;

  public:
    LinearSolver() : m_empty( true ) { m_type = LinearSolverType::SPARSE_LU; }

    LinearSolver( const LinearSolverType &solv ) : m_empty( true ) {
        m_type = solv;
        switch ( solv ) {
        case LinearSolverType::PARDISO_LU:
        case LinearSolverType::PARDISO_LDLT: {
#ifndef HAVE_INTEL_MKL
            throw std::runtime_error( "Pardiso is not installed" );
#endif /* HAVE_INTEL_MKL */
            break;
        }
        case LinearSolverType::SPARSE_LU:
        case LinearSolverType::BICGSTAB: {
            break;
        }
        default:
            throw std::runtime_error( "Unexpected LinearSolver." );
            break;
        }
    }

    void analyze( const sparse_type &A ) {

        switch ( m_type ) {
        case LinearSolverType::PARDISO_LU: {
#ifdef HAVE_INTEL_MKL
            this->m_pardisoLU.analyzePattern( A );
#else
            throw std::runtime_error( "Pardiso is not installed" );
#endif /* HAVE_INTEL_MKL */
            break;
        }
        case LinearSolverType::PARDISO_LDLT: {
#ifdef HAVE_INTEL_MKL
            this->m_pardisoLDLT.analyzePattern( A );
#else
            throw std::runtime_error( "Pardiso is not installed" );
#endif /* HAVE_INTEL_MKL */
            break;
        }
        case LinearSolverType::SPARSE_LU: {
            this->m_sparseLU.analyzePattern( A );
            break;
        }
        case LinearSolverType::BICGSTAB: {
            this->m_bicg.analyzePattern( A );
            break;
        }
        default:
            throw std::runtime_error( "Unexpected LinearSolver." );
            break;
        }
    }

    bool factorize( const sparse_type &A, bool force_analysis = false ) {

        if ( force_analysis || this->m_empty ) {
            this->analyze( A );
        }

        this->m_empty = false;

        bool succes = false;

        switch ( m_type ) {
        case LinearSolverType::PARDISO_LU: {
#ifdef HAVE_INTEL_MKL
            this->m_pardisoLU.factorize( A );
            succes = this->m_pardisoLU.info() == Eigen::Success;
#else
            throw std::runtime_error( "Pardiso is not installed" );
#endif /* HAVE_INTEL_MKL */
            break;
        }
        case LinearSolverType::PARDISO_LDLT: {
#ifdef HAVE_INTEL_MKL
            this->m_pardisoLDLT.factorize( A );
            succes = this->m_pardisoLU.info() == Eigen::Success;
#else
            throw std::runtime_error( "Pardiso is not installed" );
#endif /* HAVE_INTEL_MKL */
            break;
        }
        case LinearSolverType::SPARSE_LU: {
            this->m_sparseLU.factorize( A );
            succes = this->m_pardisoLU.info() == Eigen::Success;
            break;
        }
        case LinearSolverType::BICGSTAB: {
            this->m_bicg.factorize( A );
            succes = this->m_pardisoLU.info() == Eigen::Success;
            break;
        }
        default:
            throw std::runtime_error( "Unexpected LinearSolver." );
            break;
        }

        return succes;
    }

    vector_type solve( const vector_type &b ) const {

        if ( this->m_empty ) {
            throw std::runtime_error( "Matrix is empty." );
        }

        bool succes = false;
        vector_type x;

        switch ( m_type ) {
        case LinearSolverType::PARDISO_LU: {
#ifdef HAVE_INTEL_MKL
            x = this->m_pardisoLU.solve( b );
            succes = this->m_pardisoLU.info() == Eigen::Success;
#else
            throw std::runtime_error( "Pardiso is not installed" );
#endif /* HAVE_INTEL_MKL */
            break;
        }
        case LinearSolverType::PARDISO_LDLT: {
#ifdef HAVE_INTEL_MKL
            x = this->m_pardisoLDLT.solve( b );
            succes = this->m_pardisoLU.info() == Eigen::Success;
#else
            throw std::runtime_error( "Pardiso is not installed" );
#endif /* HAVE_INTEL_MKL */
            break;
        }
        case LinearSolverType::SPARSE_LU: {
            x = this->m_sparseLU.solve( b );
            succes = this->m_pardisoLU.info() == Eigen::Success;
            break;
        }
        case LinearSolverType::BICGSTAB: {
            x = this->m_bicg.solve( b );
            succes = this->m_pardisoLU.info() == Eigen::Success;
            break;
        }
        default:
            throw std::runtime_error( "Unexpected LinearSolver." );
            break;
        }

        if ( !succes ) {
            throw std::runtime_error( "Fail to solve the linear system." );
        }

        return x;
    }

    vector_type solve( const sparse_type &A, const vector_type &b ) {
        const bool success = this->factorize( A );

        if ( !success ) {
            throw std::runtime_error( "Fail to factorize the linear system." );
        }

        return this->solve( b );
    }

    bool empty() const { return this->m_empty; }

    auto getLocalLhs() const { return this->m_lhs_loc; }

    void setLocalLhs( const std::map< int, matrix_type > &lhs_loc ) {
        m_lhs_loc = std::make_shared< std::map< int, matrix_type > >( lhs_loc );
    }

    void setLocalLhs( const std::shared_ptr< std::map< int, matrix_type > > lhs_loc ) {
        m_lhs_loc = lhs_loc;
    }
};

} // namespace solvers
} // namespace disk
