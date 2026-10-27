/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *
 * Matteo Cicuttin (C) 2026
 * matteo.cicuttin@polito.it
 *
 * Politecnico di Torino - DISMA
 * Dipartimento di Matematica
 */

#include <iostream>
#include "diskpp/mesh/mesh.hpp"
#include "diskpp/mesh/meshgen.hpp"
#include "diskpp/quadratures/quadratures.hpp"
#include "diskpp/solvers/direct_solvers.hpp"
#include "diskpp/solvers/eigensolvers.hpp"
#include "diskpp/output/silo.hpp"
#include "diskpp/methods/hho_slapl.hpp"
#include "sgr/sgr.hpp"

#define PIVOT_VERTEX_POS 1

template<typename Mesh>
struct source;

template<disk::mesh_2D Mesh>
struct source<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        return 2.0*M_PI*M_PI*sx*sy;
    }
};

template<typename Mesh>
struct solution;

template<disk::mesh_2D Mesh>
struct solution<Mesh> {
    using point_type = typename Mesh::point_type;
    auto operator()(const point_type& pt) const {
        auto sx = std::sin(M_PI*pt.x());
        auto sy = std::sin(M_PI*pt.y());
        return sx*sy;
    }
};


template<typename Mesh>
using rvec = disk::dynamic_vector<
        typename Mesh::coordinate_type
    >;

template<typename Mesh>
using rmat = disk::dynamic_matrix<
        typename Mesh::coordinate_type
    >; 

template<typename Mesh>
using rgrad = Eigen::Matrix<
        typename Mesh::coordinate_type,
        2,
        Eigen::Dynamic
    >; 

template<typename Mesh>
using cvec = disk::dynamic_vector<
    std::complex<
        typename Mesh::coordinate_type
        >
    >; 

template<typename Mesh>
using cmat = disk::dynamic_matrix<
    std::complex<
        typename Mesh::coordinate_type
        >
    >; 


template<typename It>
void rot(It begin, It end, size_t pivot_pos)
{
    if (pivot_pos <= PIVOT_VERTEX_POS) {
        std::rotate(begin, end - (PIVOT_VERTEX_POS-pivot_pos), end);
    } else {
        std::rotate(begin, begin + (pivot_pos-PIVOT_VERTEX_POS), end);
    }
}

template<typename It>
void unrot(It begin, It end, size_t pivot_pos)
{
    if (pivot_pos <= PIVOT_VERTEX_POS) {
        std::rotate(begin, begin + (PIVOT_VERTEX_POS-pivot_pos), end);
    } else {
        std::rotate(begin, end - (pivot_pos-PIVOT_VERTEX_POS), end);
    }
}

template<disk::mesh_2D Mesh>
struct cell_data
{
    using mesh_type = Mesh;
    using cell_type = typename mesh_type::cell_type;
    using point_type = typename mesh_type::point_type;
    using T = typename Mesh::coordinate_type;
    using complex = std::complex<T>;
    
    size_t      pivot_idx;
    rvec<Mesh>  cosines;
    cvec<Mesh>  complex_vertices;
    cvec<Mesh>  complex_tangents;
    cvec<Mesh>  complex_tangents_normalized;
    rvec<Mesh>  edge_lengths;
    cmat<Mesh>  alpha;

private:
    void compute_pivot_vertex(const mesh_type& msh, const cell_type& cl)
    {
        auto pts = points(msh, cl);
        cosines = rvec<mesh_type>::Zero(pts.size());
        for (size_t i = 0; i < pts.size(); i++)
        {
            size_t pp = (i == 0) ? pts.size()-1 : i-1;
            size_t pc = i;
            size_t pn = (i+1)%pts.size();
            auto va = (pts[pp] - pts[pc]).to_vector();
            auto vb = (pts[pn] - pts[pc]).to_vector();
            va = va/va.norm();
            vb = vb/vb.norm();
            cosines[i] = va.dot(vb);
        }

        pivot_idx = 0;
        for (size_t i = 1; i < cosines.size(); i++) {
            if ( std::abs(cosines[i]) < std::abs(cosines[pivot_idx]) ) {
                pivot_idx = i;
            }
        }
        
        rot(cosines.begin(), cosines.end(), pivot_idx);
    }

    void compute_complex_vertices(const mesh_type& msh, const cell_type& cl)
    {   
        auto pts = points(msh, cl);
        complex_vertices = cvec<mesh_type>::Zero(pts.size());
        std::transform(pts.begin(), pts.end(), complex_vertices.begin(),
            [](const point_type& pt) { return std::complex(pt.x(), pt.y()); }
        );
        
        rot(complex_vertices.begin(), complex_vertices.end(), pivot_idx);

        complex_tangents = cvec<mesh_type>::Zero(pts.size());
        for (size_t i = 0; i < pts.size(); i++) {
            auto m = i;
            auto n = (i+1)%pts.size();
            complex_tangents(m) = complex_vertices(n) - complex_vertices(m);
        }

        edge_lengths = rvec<mesh_type>::Zero(pts.size());
        std::transform( complex_tangents.begin(), complex_tangents.end(),
            edge_lengths.begin(), [](const auto& v) { return std::abs(v); }
        );
        
        complex_tangents_normalized =
            complex_tangents.array() / edge_lengths.array();
    }

    void compute_alpha()
    {
        const auto& cvtx = complex_vertices;
        alpha = cmat<Mesh>::Zero(cvtx.size(), cvtx.size());

        for (size_t i = 0; i < cvtx.size(); i++) {
            for (size_t j = 0; j < cvtx.size(); j++) {
                auto ipj = (i+j) % cvtx.size();
                alpha(i,j) = 1./(cvtx[i] - cvtx[ipj]); 
            }
        }
    }

public:

    cell_data(const Mesh& msh, const cell_type& cl)
    {
        compute_pivot_vertex(msh, cl);
        compute_complex_vertices(msh, cl);
        compute_alpha();
    }

    rvec<Mesh> solveL(const rvec<Mesh>& v) const
    {
        rvec<Mesh> ret = rvec<Mesh>::Zero(v.size());
        ret(0) = v(0);
        for (size_t i = 1; i < v.size(); i++) {
            ret(i) = v(i) - cosines(i)*ret(i-1);
        }
        return ret;
    }

    rvec<Mesh> solveU(const rvec<Mesh>& v) const
    {
        rvec<Mesh> cosprods = rvec<Mesh>::Zero(cosines.size());
        cosprods(0) = cosines(0);
        for (size_t i = 1; i < v.size(); i++) {
            cosprods(i) = -1.0*cosprods(i-1)*cosines(i);
        }
        cosprods(cosprods.size()-1) = 1.0 + cosprods(cosprods.size()-1);

        rvec<Mesh> ret = rvec<Mesh>::Zero(v.size());
        ret(v.size()-1) = v(v.size()-1)/cosprods(cosprods.size()-1);
        for (size_t r = 1; r < v.size(); r++) {
            size_t i = v.size() - r - 1;
            ret(i) = v(i) - cosprods(i)*ret(ret.size()-1);
        }
        return ret;
    }

    rvec<Mesh> applyH(const rvec<Mesh>& v) const
    {
        rvec<Mesh> ret = rvec<Mesh>::Zero(v.size());
        for (size_t i = 0; i < v.size(); i++) {
            ret(i) = v(i)*std::abs( complex_tangents(i) );
        }
        return ret;
    }

    auto eval_pi(const complex& z) const
    {   
        auto Ne = complex_vertices.size();

        cvec<Mesh> Pval = cvec<Mesh>::Zero(Ne);
        cvec<Mesh> pval = cvec<Mesh>::Zero(Ne);
        cvec<Mesh> pder = cvec<Mesh>::Zero(Ne);

        for (size_t i = 0; i < Ne; i++) {
            size_t idx = (i + Ne - 2) % Ne;
            complex diff_ = z - complex_vertices(idx);
            complex Pval_ = 1.0 + alpha(i, Ne-1)*diff_;
            complex pval_ = alpha(i, Ne-1);
            complex pder_ = 0;
            for (size_t l = Ne-2; l > 0; l--) {
                auto jdx = (i+l-1) % Ne;
                diff_ = z - complex_vertices(jdx);
                pder_ = alpha(i, l)*(2.0*pval_+diff_*pder_) ;
                pval_ = alpha(i, l)*(Pval_+diff_*pval_);
                Pval_ = 1.0 + alpha(i, l)*diff_*Pval_;
            }

            Pval(i) = Pval_; // must be lagrangian
            pval(i) = pval_;
            pder(i) = pder_; 
        }

        return std::tuple{Pval, pval, pder};
    }

    auto eval_pi(const point_type& pt) const
    {
        return eval_pi({pt.x(), pt.y()});
    }


    auto eval_btilde(const complex& z) const
    {
        auto Ne = complex_vertices.size();

        auto& normtan = complex_tangents_normalized;
        auto [Pval, pval, pder] = eval_pi(z);

        rvec<Mesh> bt  = - (pval.array() * normtan.array()).real();
        rvec<Mesh> bxt = - (pder.array() * normtan.array()).real();
        rvec<Mesh> byt =   (pder.array() * normtan.array()).imag();
    
        return std::tuple{bt, bxt, byt};
    }

    auto eval_b(const complex& z) const
    {
        auto [bt, bxt, byt] = eval_btilde(z);

        rvec<Mesh> b  = applyH( solveU( solveL( bt ) ) );
        rvec<Mesh> bx = applyH( solveU( solveL( bxt ) ) );
        rvec<Mesh> by = applyH( solveU( solveL( byt ) ) );

        return std::tuple{b, bx, by};
    }
};




template<disk::mesh_2D Mesh>
auto
eval_bases(const Mesh& msh, const typename Mesh::cell_type& cl,
    const cell_data<Mesh>& cdata,
    const std::complex<double>& z)
{
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;
    using vec2 = disk::static_vector<T,2>;

    auto Ne = cdata.complex_vertices.size();

    auto [b, bx, by] = cdata.eval_b(z);

    auto c2p = [](const std::complex<T>& z) {
        return point_type{z.real(), z.imag()};
    };

    point_type v1 = c2p(cdata.complex_vertices(0));
    point_type v2 = c2p(cdata.complex_vertices(1));
    point_type v3 = c2p(cdata.complex_vertices(2));

    auto invAreaT = 1./(det(v2-v1,v3-v1)/2.0);
    auto m1 = (v1+v3)/2.0;
    auto m2 = (v2+v1)/2.0;
    auto m3 = (v3+v2)/2.0;
    auto r1 = [&](const point_type& x) { return invAreaT*det(x-m1, v2-v1); };
    auto r2 = [&](const point_type& x) { return invAreaT*det(x-m2, v3-v2); };
    auto r3 = [&](const point_type& x) { return invAreaT*det(x-m3, v1-v3); };

    vec2 gr1z = invAreaT*vec2{v2.y() - v1.y(), v1.x() - v2.x()};
    vec2 gr2z = invAreaT*vec2{v3.y() - v2.y(), v2.x() - v3.x()};
    vec2 gr3z = invAreaT*vec2{v1.y() - v3.y(), v3.x() - v1.x()};

    Eigen::Matrix<T, 3, Eigen::Dynamic> tij =
        Eigen::Matrix<T, 3, Eigen::Dynamic>::Zero(3, Ne);

    auto fcs = faces(msh, cl);
    rot(fcs.begin(), fcs.end(), cdata.pivot_idx);

    for (size_t i = 0; i < Ne; i++) {
        const auto& fc = fcs[i];
        auto mf = barycenter(msh, fc);
        tij(0, i) = r1(mf);
        tij(1, i) = r2(mf);
        tij(2, i) = r3(mf);
    }

    size_t J = 0;
    for (size_t i = 1; i < Ne; i++) {
        if ( std::abs(tij(2,J)) < std::abs(tij(2,i)) ) {
            J = i;
        }
    }
    assert (J < Ne);

    auto r1z = r1( c2p(z) );
    auto r2z = r2( c2p(z) );
    auto r3z = r3( c2p(z) );

    T sub1 = 0.0, sub2 = 0.0, sub3 = 0.0;
    vec2 gsub1{0,0}, gsub2{0,0}, gsub3{0,0};

    rvec<Mesh> B = rvec<Mesh>::Zero(Ne);
    rgrad<Mesh> gB = rgrad<Mesh>::Zero(2, Ne);
    for (size_t i = 2; i < Ne; i++) {
        B(i) = b(i);
        gB(0, i) = bx(i);
        gB(1, i) = by(i);

        if (i != J) {
            sub3 += tij(2,i)*B(i);
            gsub3 += tij(2,i)*gB.col(i);
        }
    }
    B(J) = (r3z - sub3)/tij(2,J);
    gB.col(J) = (gr3z - gsub3)/tij(2,J);

    for (size_t i = 2; i < Ne; i++) {
        sub1 += tij(0,i)*B(i);
        sub2 += tij(1,i)*B(i);
        gsub1 += tij(0,i)*gB.col(i);
        gsub2 += tij(1,i)*gB.col(i);
    }
    
    B(0) = r1z - sub1;
    B(1) = r2z - sub2;
    
    gB.col(0) = gr1z - gsub1;
    gB.col(1) = gr2z - gsub2;
    
    
    std::vector<size_t> shift(Ne);
    std::iota(shift.begin(), shift.end(), 0);
    unrot(shift.begin(), shift.end(), cdata.pivot_idx);
    //std::cout << sgr::printc(shift.begin(), shift.end()) << std::endl;

    rvec<Mesh> Bret = rvec<Mesh>::Zero(Ne);
    rgrad<Mesh> gBret = rgrad<Mesh>::Zero(2, Ne);

    for (size_t i = 0; i < Ne; i++) {
        Bret(i) = B(shift[i]);
        gBret.col(i) = gB.col(shift[i]); 
    }

    return std::pair{Bret, gBret};
}

template<disk::mesh_2D Mesh>
auto
eval_bases(const Mesh& msh, const typename Mesh::cell_type& cl,
    const cell_data<Mesh>& cdata,
    const typename Mesh::point_type& pt)
{
    return eval_bases(msh, cl, cdata, {pt.x(), pt.y()});
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "Please specify degree.\n";
        return 1;
    }
    
    using T = double;
    using mesh_type = disk::generic_mesh<T,2>;
    using point_type = typename mesh_type::point_type;

    mesh_type msh;
    auto mesher = disk::make_fvca5_hex_mesher(msh);
    mesher.make_level(std::stoull(argv[1]));
    auto avgh = disk::average_diameter(msh);
    disk::shake(msh, avgh/10, 42);

    source<mesh_type> f;

    disk::silo_database silo;

    silo.create("bm.silo");
    silo.add_mesh(msh, "mesh");

    disk::hho::slapl::degree_info di(0);
    std::vector<bool> dirfaces( msh.faces_size(), false );
    auto assm = disk::hho::slapl::make_assembler(msh, di, dirfaces);
    auto assmM = disk::hho::slapl::make_assembler(msh, di, dirfaces);

    Eigen::Matrix<T, Eigen::Dynamic, 2> vecs;
    vecs.resize(msh.cells_size(), 2);

    for (size_t i = 0; i < msh.cells_size(); i++) {
        const auto& cl = msh.cell_at(i);
        auto bar = barycenter(msh, cl);
        auto pts = points(msh, cl);

        cell_data<mesh_type> cdata(msh, cl);

        auto Ne = cdata.complex_vertices.size();
        //std::cout << "############################# faces: " << Ne << std::endl;
        //std::cout << " -> cdata.pivot_idx = " << cdata.pivot_idx << "\n";
        auto fcs = faces(msh, cl);
        //rot(fcs.begin(), fcs.end(), cdata.pivot_idx);
        for (auto& fc : fcs) {
            rvec<mesh_type> v = rvec<mesh_type>::Zero(Ne);
            auto qps = integrate(msh, fc, Ne-2);
            for (auto& qp : qps) {
                auto [B, gB] = eval_bases(msh, cl, cdata, qp.point());
                v += qp.weight() * B;
            }
            //std::cout << "  " << v.transpose()/measure(msh, fc) << "\n";
        
            auto [B1, gB1] = eval_bases(msh, cl, cdata, barycenter(msh, fc));
            //std::cout << gB1.rowwise().sum() << std::endl;
            //std::cout << B1.sum() << std::endl;
        }

        rmat<mesh_type> lhs = rmat<mesh_type>::Zero(Ne, Ne);
        rmat<mesh_type> M = rmat<mesh_type>::Zero(Ne, Ne);
        rvec<mesh_type> rhs = rvec<mesh_type>::Zero(Ne);

        auto qps = integrate(msh, cl, 2*Ne);
        for (auto& qp : qps) {
            auto [B1, gB1] = eval_bases(msh, cl, cdata, qp.point());
            lhs += qp.weight() * gB1.transpose() * gB1;
            //M += qp.weight() * B1 * B1.transpose();
            rhs += qp.weight() * f(qp.point()) * B1;
        }

        for (auto& fc : fcs) {
            auto bi = msh.boundary_info(fc);
            if ( not bi.is_boundary() ) {
                continue;
            }

            auto fqps = integrate(msh, fc, 2*Ne);
            if (bi.tag() == 0) {
                /* robin */
                for (auto& qp : fqps) {
                    auto [B1, gB1] = eval_bases(msh, cl, cdata, qp.point());
                    lhs += qp.weight() * B1 * B1.transpose();
                }
            } else {
                /* steklov */
                for (auto& qp : fqps) {
                    auto [B1, gB1] = eval_bases(msh, cl, cdata, qp.point());
                    M += qp.weight() * B1 * B1.transpose();
                }
            }
        }

        assm.assemble(msh, cl, lhs, rhs);
        assmM.assemble(msh, cl, M, rhs);

#if 0
        std::cout << "*****\n";
        cmat<mesh_type> M = cmat<mesh_type>::Identity(pts.size(), pts.size());
        M(0, M.cols()-1) = cdata.cosines(0);
        for (size_t i = 1; i < M.rows(); i++) {
            M(i, i-1) = cdata.cosines(i);
        }
        std::cout << M.real() << "\n\n";

        std::cout << "*****\n";
        for (const auto& z : cdata.complex_vertices) {
            auto [Pval, pval, pder]  = cdata.eval_pi(z);
            std::cout << Pval.transpose().real() << std::endl;
            //std::cout << pval.transpose() << std::endl;
            //std::cout << pder.transpose() << std::endl;
        }

        cmat<mesh_type> M = cmat<mesh_type>::Identity(pts.size(), pts.size());
        M(0, M.cols()-1) = cdata.cosines(0);
        for (size_t i = 1; i < M.rows(); i++) {
            M(i, i-1) = cdata.cosines(i);
        }

        cmat<mesh_type> I = cmat<mesh_type>::Identity(pts.size(), pts.size());
        cmat<mesh_type> ret = cmat<mesh_type>::Zero(pts.size(), pts.size());

        for (size_t i = 0; i < pts.size(); i++) {
            ret.col(i) = cdata.solveU( cdata.solveL(M.col(i)) );
        }        

        std::cout << "******************\n";
        std::cout << ret.real() << "\n\n";
#endif

        //std::cout << cdata.alpha << "\n\n";
    }

    assm.finalize();
    assmM.finalize();

    disk::dynamic_vector<T> sol;
    disk::solvers::sparse_lu(assm.LHS, assm.RHS, sol);

    std::cout << "h = " << disk::average_diameter(msh) << std::endl;

    disk::solvers::feast_eigensolver_params<T> fep;
    fep.verbose = 1;
    fep.min_eigval = 0;
    fep.max_eigval = 5;
    fep.subspace_size = 20;
    fep.tolerance = 8;
    fep.max_iter = 10;
    fep.fis = disk::solvers::feast_inner_solver::mumps;

    disk::dynamic_matrix<T> eigvecs;
    disk::dynamic_vector<T> eigvals;

    disk::solvers::feast(fep, assm.LHS, assmM.LHS, eigvecs, eigvals);

    std::cout << eigvals.transpose() << std::endl;

    /*
    disk::dynamic_vector<T> eigvals_ref = disk::dynamic_vector<T>::Zero(eigvals.size());

    int k = 0;
    for (int m = 1; m < 4; m++) {
        for (int n = 1; n < 4; n++) {
            if (m+n > 4) {
                continue;
            }

            eigvals_ref(k++) = M_PI*M_PI*(m*m+n*n);
        }
    }

    std::sort(eigvals_ref.begin(), eigvals_ref.end());

    std::cout << eigvals_ref.transpose() << std::endl;

    
    std::cout << std::setprecision(15);
    std::cout << (eigvals - eigvals_ref).transpose() << std::endl;

    */

    std::vector<T> u_data;
    std::vector<T> pivots;

    T L2err = 0.0;

    auto u = solution<mesh_type>();

    size_t i = 0;
    for (auto& cl : msh) {
        cell_data<mesh_type> cdata(msh, cl);

        auto [B1, gB1] = eval_bases(msh, cl, cdata, barycenter(msh, cl));
    
        auto locsol = assm.take_local_solution(msh, cl, sol);
        u_data.push_back( locsol.dot(B1) );

        auto Ne = cdata.complex_vertices.size();
        auto qps = integrate(msh, cl, 2*Ne);
        for (auto& qp : qps) {
            auto [B2, gB2] = eval_bases(msh, cl, cdata, qp.point());
            auto uh = locsol.dot(B2);
            L2err += qp.weight() * (uh - u(qp.point())) * (uh - u(qp.point()));
        }
        pivots.push_back( cdata.pivot_idx );
    } 

    std::cout << "L2 error = " << std::sqrt(L2err) << std::endl;

    silo.add_variable("mesh", "ptrs", vecs, disk::zonal_variable_t);
    silo.add_variable("mesh", "u", u_data, disk::zonal_variable_t);
    silo.add_variable("mesh", "pivot_idx", pivots, disk::zonal_variable_t);
    

    return 0;
}