/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *  
 * Matteo Cicuttin (C) 2020, 2021, 2022
 * matteo.cicuttin@uliege.be
 *
 * University of Liège - Montefiore Institute
 * Applied and Computational Electromagnetics group  
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <regex>

#include <sstream>
#include <string>

#include <unistd.h>

#include "diskpp/methods/hho"
#include "diskpp/methods/implementation_hho/curl.hpp"
#include "diskpp/loaders/loader.hpp"
#include "diskpp/output/silo.hpp"

#include "mumps.hpp"

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
void
run_magneto_solver(Mesh<T, 3, Storage>& msh)
{
	auto Bdeg = 1;
	auto PsiTdeg = 1;
	auto PsiFdeg = 1;

	auto num_cells = msh.cells_size();
	std::vector<bool> dirichlet_faces(msh.faces_size(), false);

	std::vector<Eigen::Triplet<T>> triplets;

	for (size_t i = 0; i < msh.faces_size(); i++)
	{
		auto fc = *std::next(msh.faces_begin(), i);
		auto bi = msh.boundary_info(fc);
		if (bi.is_boundary())
			if (bi.tag() == 9 or bi.tag() == 11)
				dirichlet_faces[i] = true;
	}

	auto num_dirichlet_faces = std::count(dirichlet_faces.begin(), dirichlet_faces.end(), true);
	auto num_nondirichlet_faces = std::count(dirichlet_faces.begin(), dirichlet_faces.end(), false);

	std::cout << "Cells: " << num_cells << std::endl;
	std::cout << "Dirichlet faces: " << num_dirichlet_faces << std::endl;
	std::cout << "Nondirichlet faces: " << num_nondirichlet_faces << std::endl;

	std::vector<int>     compress_table;
    std::vector<int>     expand_table;

	compress_table.resize( msh.faces_size() );
    expand_table.resize( msh.internal_faces_size() );

	SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

	int face_i = 0;
	int compressed_ofs = 0;
	for (auto& fc : faces(msh))
	{
		if ( not dirichlet_faces[face_i] )
		{
			compress_table[face_i] = compressed_ofs;
			expand_table[compressed_ofs] = face_i;
			compressed_ofs++;
		}

		face_i++;
	}

	int cell_i = 0;
	for (auto& cl : msh)
	{
		auto B_basis = make_vector_monomial_basis(msh, cl, Bdeg);
		auto PsiT_basis = make_scalar_monomial_basis(msh, cl, PsiTdeg);
		auto B_sz = B_basis.size();
		auto PsiT_sz = PsiT_basis.size();
		auto PsiF_sz = disk::scalar_basis_size(PsiFdeg, 2);

		auto fcs = faces(msh, cl);
		auto num_faces = fcs.size();

		/* Init local matrix */
		auto Asz = B_sz + PsiT_sz + PsiF_sz*num_faces;
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Aloc(Asz, Asz);
		Aloc = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(Asz, Asz);

		std::vector<int> l2g(Asz);
		for (size_t i = 0; i < B_sz; i++)
			l2g[i] = cell_i*B_sz + i;
		for (size_t i = B_sz; i < B_sz+PsiT_sz; i++)
			l2g[i] = num_cells*B_sz + cell_i*PsiT_sz + i;
	
		/* B field mass term */
		auto qps_B = disk::integrate(msh, cl, 2*Bdeg);
		for (auto& qp : qps_B)
		{
			auto phi = B_basis.eval_functions(qp.point());
			Aloc.block(0, 0, B_sz, B_sz) += qp.weight() * phi * phi.transpose();
		}
	
		/* Cell gradient term */
		auto qps_PsiT = disk::integrate(msh, cl, 2*PsiTdeg);
		for (auto& qp : qps_PsiT)
		{
			auto trial = PsiT_basis.eval_functions(qp.point());
			auto test = B_basis.eval_divergences(qp.point());
			Aloc.block(0, B_sz, B_sz, PsiT_sz) += qp.weight() * test * trial.transpose();
		}
		Aloc.block(B_sz, 0, PsiT_sz, B_sz) = Aloc.block(0, B_sz, B_sz, PsiT_sz).transpose();

		/* Face gradient term */
		auto face_ofs = B_sz + PsiT_sz;
		for (auto& fc : fcs)
		{
			auto n = normal(msh, cl, fc);
			auto PsiF_basis = make_scalar_monomial_basis(msh, fc, PsiFdeg);
			auto qps_PsiF = disk::integrate(msh, cl, 2*PsiFdeg);
			for (auto& qp : qps_PsiF)
			{
				auto trial = PsiF_basis.eval_functions(qp.point());
				auto test = B_basis.eval_functions(qp.point());
				auto test_n = test*n;
				Aloc.block(0, face_ofs, B_sz, PsiF_sz) += qp.weight() * test_n * trial.transpose();
			}

			auto fnum = offset(msh, fc);

			if ( dirichlet_faces[fnum] )
			{
				for (size_t i = face_ofs; i < face_ofs+PsiF_sz; i++)
					l2g[i] = -1;
			}
			else 
			{
				//for (size_t i = face_ofs; i < face_ofs+PsiF_sz; i++)
				//	l2g[i] = num_cells*(B_sz + PsiT_sz) + compress_table[fnum]*PsiF_sz + i;
			}

			face_ofs += PsiF_sz;
		}
		Aloc.bottomLeftCorner(PsiF_sz*num_faces, B_sz) = Aloc.topRightCorner(B_sz, PsiF_sz*num_faces).transpose();
	
		for (size_t i = 0; i < Aloc.rows(); i++)
		{
			if (l2g[i] == -1)
				continue;
			
			for (size_t j = 0; j < Aloc.cols(); j++)
			{
				if (l2g[j] == -1)
					continue;

				triplets.push_back( Eigen::Triplet<T>(l2g[i], l2g[j], Aloc(i,j)) );
			}
		}

		cell_i++;
	}

	auto B_sz = disk::vector_basis_size(Bdeg, 3, 3);
	auto PsiT_sz = disk::scalar_basis_size(PsiTdeg, 3);
	auto PsiF_sz = disk::scalar_basis_size(PsiFdeg, 2);

	auto syssz = (B_sz + PsiT_sz) * msh.cells_size() + PsiF_sz * num_nondirichlet_faces;

	LHS = SparseMatrix<T>(syssz, syssz);

	LHS.setFromTriplets( triplets.begin(), triplets.end() );
	triplets.clear();

	disk::dump_sparse_matrix(LHS, "m.txt");
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cout << "params!" << std::endl;
		return 1;
	} 

	const char *mesh_filename = argv[1];

	using real_type = double;

	/* GMSH */
	if (std::regex_match(mesh_filename, std::regex(".*\\.geo3s$") ))
	{
		std::cout << "Guessed mesh format: GMSH simplicial" << std::endl;

		using mesh_type = disk::simplicial_mesh<real_type,3>;
		
		disk::gmsh_geometry_loader< mesh_type > loader;
		
		mesh_type msh;

		loader.read_mesh(mesh_filename);
		loader.populate_mesh(msh);
		run_magneto_solver(msh);
		return 0;
	}

	std::cout << "Unknown mesh format, can't proceed." << std::endl;
	return 1;
}