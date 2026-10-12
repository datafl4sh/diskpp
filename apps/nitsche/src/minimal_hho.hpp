#pragma once

/***************************************************************
 * Nitsche-HHO reconstruction operator
 */
template<typename Mesh>
auto hho_minimal_reconstruction(const Mesh& msh,
    const typename Mesh::cell_type& cl, size_t degree,
    const std::vector<bc>& bcs)
{
    using scalar_type = typename Mesh::coordinate_type;
    static const size_t DIM = Mesh::dimension;
    /* Reconstruction space basis */
    auto rb = disk::make_scalar_monomial_basis(msh, cl, degree+1);
    auto rbs = rb.size();
    /* Cell space basis: same as reconstruction space */ 
    //auto cb = disk::make_scalar_monomial_basis(msh, cl, degree+1);
    //auto cbs = cb.size();
    /* Face basis info */
    auto fcs = faces(msh, cl);
    auto fbs = disk::scalar_basis_size(degree, Mesh::dimension-1);
    auto n_allfacedofs = fcs.size() * fbs;

    using mat = disk::dynamic_matrix<scalar_type>;
    using vec = disk::dynamic_vector<scalar_type>;
    using grad = Eigen::Matrix<scalar_type, Eigen::Dynamic, DIM>;

    mat LHS = mat::Zero(rbs-1, rbs-1);

    /* Stiffness */
    mat K = mat::Zero(rbs, rbs);

    /* Local problem RHS */
    mat RHS = mat::Zero(rbs-1, rbs + n_allfacedofs);
    

    auto qps = disk::integrate(msh, cl, 2*degree);
    for (const auto& qp : qps) {
        /* (grad(v), grad(w))_T */
        auto cphi = rb.eval_functions(qp.point()); 
        auto dphi = rb.eval_gradients(qp.point());
        K += (qp.weight() * dphi) * dphi.transpose();
    }

    LHS.block(0,0,rbs-1,rbs-1) += K.block(1,1,rbs-1,rbs-1);
    RHS.block(0,0,rbs-1,rbs) += K.block(1,0,rbs-1,rbs);

    auto inv_hT = 1.0/diameter(msh, cl);
    for (size_t fcnum = 0; fcnum < fcs.size(); fcnum++) {
        const auto& fc = fcs[fcnum];
        auto fb = disk::make_scalar_monomial_basis(msh, fc, degree);
        auto bi = msh.boundary_info(fc);
        auto ofs = rbs + fbs*fcnum;
        auto fqps = disk::integrate(msh, fc, 2*degree+2);
        auto n = normal(msh, cl, fc);
        auto fcid = offset(msh, fc);

        if (bi.is_boundary() and not bi.is_internal()) { /* Do "minimal hho" if on a domain boundary */

            if (bcs[fcid] == bc::dirichlet) {
                for (const auto& qp : fqps) {
                    auto cphi = rb.eval_functions(qp.point());
                    auto fphi = fb.eval_functions(qp.point());
                    grad dphi = rb.eval_gradients(qp.point()).block(1,0,rbs-1,DIM);
                    RHS.block(0,   0, rbs-1, rbs) -=
                        qp.weight() * (dphi*n) * cphi.transpose();
                    RHS.block(0, ofs, rbs-1, fbs) +=
                        qp.weight() * (dphi*n) * fphi.transpose();
                }
            }

            /* nothing to do if on Neumann or Robin faces */

        } else { /* Do standard HHO if not on a domain boundary */
            for (const auto& qp : fqps) {
                auto cphi = rb.eval_functions(qp.point());
                auto fphi = fb.eval_functions(qp.point());
                grad dphi = rb.eval_gradients(qp.point()).block(1,0,rbs-1,DIM);
                RHS.block(0,   0, rbs-1, rbs) -=
                    qp.weight() * (dphi*n) * cphi.transpose();
                RHS.block(0, ofs, rbs-1, fbs) +=
                    qp.weight() * (dphi*n) * fphi.transpose();
            }
        }
    }

    mat oper = LHS.ldlt().solve(RHS);
    mat data = oper.transpose() * RHS;

    return std::pair{oper, data};
}

template<typename Mesh, typename Space = disk::hho::slapl::hho_space<Mesh>>
auto
hho_minimal_reconstruction_n(const Mesh& msh,
    const typename Mesh::cell_type& cl, size_t degree,
    const std::vector<bc>& bcs)
{
    using namespace disk::basis;
    using T = typename Space::scalar_type;

    disk::hho::slapl::degree_info di(degree+1, degree);

    auto phiR = Space::reco_basis(msh, cl, di.reco);
    auto phiT = Space::cell_basis(msh, cl, di.cell);

    auto [szT, szF, szR] = space_dimensions<Space>(di);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();
    assert(szR > 0);
    auto allcols = szT + num_faces*szF;

    using mat = disk::dynamic_matrix<T>;

    mat stiffness = integrate(msh, cl, grad(phiR), grad(phiR));
    mat lhs = stiffness.bottomRightCorner(szR-1, szR-1);
    mat rhs = mat::Zero(szR-1, allcols);
    rhs.block(0,0,szR-1,szT) = stiffness.bottomLeftCorner(szR-1, szT);

    size_t ofs = szT;
    for (const auto& fc : fcs)
    {
        auto n = normal(msh, cl, fc);
        auto phiF = Space::face_basis(msh, fc, di.face);
        auto fcid = offset(msh, fc);
        auto bi = msh.boundary_info(fc);

        if (bi.is_boundary() and not bi.is_internal()) {
            if (bcs[fcid] == bc::dirichlet) {

                rhs.block(0,ofs,szR-1,szF) += 
                    integrate(msh, fc, phiF, grad(phiR).dot(n)).block(1,0,szR-1,szF);

                rhs.block(0,0,szR-1,szT) -=
                    integrate(msh, fc, phiT, grad(phiR).dot(n)).block(1,0,szR-1,szT);
            }
        } else {
            rhs.block(0,ofs,szR-1,szF) += 
                integrate(msh, fc, phiF, grad(phiR).dot(n)).block(1,0,szR-1,szF);

            rhs.block(0,0,szR-1,szT) -=
                integrate(msh, fc, phiT, grad(phiR).dot(n)).block(1,0,szR-1,szT);
        }

        ofs += szF;
    }

    mat R = lhs.llt().solve(rhs);
    mat A = rhs.transpose()*R;
    return std::pair(R, A);
}

template<typename Mesh>
disk::dynamic_matrix<typename Mesh::coordinate_type>
hho_minimal_stabilization(const Mesh& msh,
    const typename Mesh::cell_type& cl, size_t degree, const std::vector<bc>& bcs)
{
    /* We use a standard Lehrenfeld-Schoeberl stabilization and we
     * need to stabilize only on the internal interfaces, not on
     * the domain boundary. */
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = degree+1;
    const auto cb = disk::make_scalar_monomial_basis(msh, cl, celdeg);
    const auto cbs = cb.size();

    const auto fcs = faces(msh, cl);
    const auto fbs = disk::scalar_basis_size(degree, Mesh::dimension-1);
    const auto num_faces_dofs = fbs*fcs.size();
    const auto total_dofs     = cbs + num_faces_dofs;

    matrix_type data = matrix_type::Zero(total_dofs, total_dofs);

    T hT = diameter(msh, cl);
    T stabparam = 1.0/hT;

    for (size_t i = 0; i < fcs.size(); i++) {
        size_t ofs = cbs+i*fbs;
        const auto fc = fcs[i];

        /* If the face is on the domain boundary, just skip to the next */
        auto bi = msh.boundary_info(fc);
        auto fcid = offset(msh, fc);
        if (bi.is_boundary() and (not bi.is_internal()) and (bcs[fcid] != bc::dirichlet)) {
            continue;
        }

        /* Compute standard L-S stabilization otherwise. */
        const auto facdeg = degree;
        const auto fb  = make_scalar_monomial_basis(msh, fc, facdeg);
        const auto fbs = disk::scalar_basis_size(facdeg, Mesh::dimension - 1);

        const matrix_type If    = matrix_type::Identity(fbs, fbs);
        matrix_type       oper  = matrix_type::Zero(fbs, total_dofs);
        matrix_type       tr    = matrix_type::Zero(fbs, total_dofs);
        matrix_type       mass  = make_mass_matrix(msh, fc, fb);
        matrix_type       trace = matrix_type::Zero(fbs, cbs);

        oper.block(0, ofs, fbs, fbs) = -If;

        const auto qps = integrate(msh, fc, facdeg + celdeg);
        for (auto& qp : qps)
        {
            const auto c_phi = cb.eval_functions(qp.point());
            const auto f_phi = fb.eval_functions(qp.point());

            assert(c_phi.rows() == cbs);
            assert(f_phi.rows() == fbs);
            assert(c_phi.cols() == f_phi.cols());

            trace += (qp.weight() * f_phi) * c_phi.transpose();
        }

        tr.block(0, ofs, fbs, fbs) = -mass;
        tr.block(0, 0, fbs, cbs)   = trace;

        oper.block(0, 0, fbs, cbs) = mass.ldlt().solve(trace);
        data += oper.transpose() * tr * stabparam;
    }

    return data;
}

template<typename Mesh, typename Space = disk::hho::slapl::hho_space<Mesh>>
disk::dynamic_matrix<typename Space::scalar_type>
hho_minimal_stabilization_n(const Mesh& msh,
    const typename Mesh::cell_type& cl, size_t degree,
    const std::vector<bc>& bcs)
{
    using namespace disk::basis;
    using T = typename Space::scalar_type;
    using mat = disk::dynamic_matrix<T>;

    disk::hho::slapl::degree_info di(degree+1, degree);

    auto [szT, szF, szR] = space_dimensions<Space>(di);
    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    auto sz_total = szT + num_faces*szF;
    mat S = mat::Zero(sz_total, sz_total);

    auto phiT = Space::cell_basis(msh, cl, di.cell);
    auto phiR = Space::reco_basis(msh, cl, di.reco);
    
    auto scale = 1./diameter(msh, cl);

    size_t ofs = szT;
    for (const auto& fc : fcs)
    {
        auto bi = msh.boundary_info(fc);
        auto fcid = offset(msh, fc);
        if (bi.is_boundary() and (not bi.is_internal()) and (bcs[fcid] != bc::dirichlet)) {
            continue;
        }

        auto phiF = Space::face_basis(msh, fc, di.face);
        mat MF = integrate(msh, fc, phiF, phiF);
        mat T2F = integrate(msh, fc, phiT, phiF);
        Eigen::LLT<mat> MF_llt(MF);
        
        mat rhs = mat::Zero(szF, szT+num_faces*szF);
        rhs.block(0,0,szF,szT) = -MF_llt.solve(T2F);
        rhs.block(0,ofs,szF,szF) = mat::Identity(szF, szF);

        S += scale * rhs.transpose() * MF * rhs;
        ofs += szF;
    }

    return S;
}

template<typename Mesh, typename SourceFun,
    typename NeumannFun, typename RobinFun>
disk::dynamic_vector<typename Mesh::coordinate_type>
hho_minimal_rhs(const Mesh& msh, const typename Mesh::cell_type& cl,
    SourceFun f, NeumannFun gN, RobinFun gR, size_t degree,
    typename Mesh::coordinate_type eta, const std::vector<bc>& bcs)
{
    using scalar_type = typename Mesh::coordinate_type;

    auto cb = disk::make_scalar_monomial_basis(msh, cl, degree+1);
    auto cbs = cb.size();

    auto fcs = faces(msh, cl);
    auto fbs = disk::scalar_basis_size(degree, Mesh::dimension-1);
    auto n_allfacedofs = fcs.size() * fbs;

    disk::dynamic_vector<scalar_type> ret =
        disk::dynamic_vector<scalar_type>::Zero(cbs + n_allfacedofs);
    
    auto qps = disk::integrate(msh, cl, 2*degree+2);
    for (auto& qp : qps) {
        auto phi = cb.eval_functions(qp.point());
        ret.head(cbs) += qp.weight() * f(qp.point()) * phi;
    }

    auto inv_hT = 1.0/diameter(msh, cl);
    for (size_t fcnum = 0; fcnum < fcs.size(); fcnum++) {
        const auto& fc = fcs[fcnum];
        auto bi = msh.boundary_info(fc);
        if ( not bi.is_boundary() ) {
            continue;
        }

        auto fb = disk::make_scalar_monomial_basis(msh, fc, degree);
        auto ofs = cbs + fbs*fcnum;
        auto fqps = disk::integrate(msh, fc, 2*degree+2);
        auto n = normal(msh, cl, fc);

        auto fcid = offset(msh, fc);

        if (bcs[fcid] == bc::dirichlet) {
            /* Nothing */
        }

        if (bcs[fcid] == bc::neumann) {
            for (const auto& qp : fqps) {
                auto phi = cb.eval_functions(qp.point());
                auto gN_val = gN(qp.point());
                /* (gN, w)_F */
                ret.head(cbs) += qp.weight() * gN_val * phi;
            }
        }

        if (bcs[fcid] == bc::robin) {
            for (const auto& qp : fqps) {
                auto phi = cb.eval_functions(qp.point());
                auto gR_val = gR(qp.point());
                /* (gN, w)_F */
                ret.head(cbs) += qp.weight() * gR_val * phi;
            }
        }
    }

    return ret;
}
