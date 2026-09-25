#include <array>

#include "diskpp/mesh/mesh.hpp"
#include "diskpp/mesh/meshgen.hpp"
#include "diskpp/methods/hho_assemblers.hpp"
#include "diskpp/methods/hho_slapl.hpp"
#include "diskpp/solvers/direct_solvers.hpp"
#include "diskpp/output/silo.hpp"
#include "diskpp/common/timecounter.hpp"


template<typename Mesh>
struct source_functor;

template<disk::mesh_1D Mesh>
struct source_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        return M_PI*M_PI*sx;
    }
};

template<disk::mesh_2D Mesh>
struct source_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        return 2.0*M_PI*M_PI*sx*sy;
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


template<typename Mesh>
struct solution_functor;

template<disk::mesh_1D Mesh>
struct solution_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        return sx;
    }
};

template<disk::mesh_2D Mesh>
struct solution_functor<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        return sx*sy;
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

template<typename T>
void
crouzeix_raviart_solver(const disk::simplicial_mesh<T,2>& msh,
    disk::silo_database& silo)
{
    std::cout << "Crouzeix-Raviart solver" << std::endl;

    using point_type = typename disk::simplicial_mesh<T,2>::point_type;
 
    disk::hho::slapl::degree_info di(0);

    auto f = make_rhs_function(msh);

    auto assm = make_assembler(msh, di);

    timecounter tc;

    tc.tic();
    for (auto& cl : msh)
    {
        auto pts = points(msh, cl);
        auto meas = measure(msh, cl);
        auto x0 = pts[0].x(); auto y0 = pts[0].y();
        auto x1 = pts[1].x(); auto y1 = pts[1].y();
        auto x2 = pts[2].x(); auto y2 = pts[2].y();
        std::array<point_type, 3> mids = {
            0.5*(pts[0]+pts[1]), 0.5*(pts[1]+pts[2]), 0.5*(pts[0]+pts[2])
        };
        std::array<T,3> b = { y0-y1, y1-y2, y2-y0 };
        std::array<T,3> c = { x1-x0, x2-x1, x0-x2 };
        disk::dynamic_matrix<T> K = disk::dynamic_matrix<T>::Zero(3,3);
        disk::dynamic_vector<T> rhs = disk::dynamic_vector<T>::Zero(3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                K(i,j) = (b[i]*b[j] + c[i]*c[j])/meas;
            }
            rhs(i) = meas*f(mids[i])/3.0;
        }

        assm.assemble(msh, cl, K, rhs);
    }
    assm.finalize();

    std::cout << " Assembly time: " << tc.toc() << std::endl;
    std::cout << " Unknowns: " << assm.LHS.rows() << " ";
    std::cout << " Nonzeros: " << assm.LHS.nonZeros() << std::endl;
    tc.tic();
    disk::dynamic_vector<T> sol;
    disk::solvers::sparse_lu(assm.LHS, assm.RHS, sol);    
    std::cout << " Solver time: " << tc.toc() << std::endl;

    std::vector<T> u_data;

    auto u_sol = make_solution_function(msh);
    tc.tic();
    size_t cell_i = 0;
    for (auto& cl : msh)
    {
        auto locsolF = assm.take_local_solution(msh, cl, sol);
        auto d = (locsolF(0) + locsolF(1) + locsolF(2))/3.0;
        u_data.push_back(d);
    }
    silo.add_variable("mesh", "u_cr", u_data, disk::zonal_variable_t);

    std::cout << " Postpro time: " << tc.toc() << std::endl;
}

int main(int argc, char **argv)
{
    using T = double;
    using mesh_type = disk::simplicial_mesh<T,2>;
    
    mesh_type msh;
    auto mesher = disk::make_simple_mesher(msh);

    mesher.refine();
    mesher.refine();
    mesher.refine();
    mesher.refine();
    mesher.refine();

    disk::silo_database db;
    db.create("poisson_cr.silo");
    db.add_mesh(msh, "mesh");

    crouzeix_raviart_solver(msh, db);

    db.close();
    return 0;
}