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

	for (auto& cl : msh)
	{
		auto B_basis = make_vector_monomial_basis(msh, cl, Bdeg);
		auto PsiT_basis = make_scalar_monomial_basis(msh, cl, PsiTdeg);
		auto B_sz = B_basis.size();
		auto PsiT_sz = PsiT_basis.size();
		auto PsiF_sz = disk::scalar_basis_size(PsiFdeg, 2);

		auto Asz = B_sz + PsiT_sz + PsiF_sz;
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Aloc;
		Aloc = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(Asz, Asz);


		auto qps_B = disk::integrate(msh, cl, 2*Bdeg);
		for (auto& qp : qps_B)
		{
			auto phi = B_basis.eval_functions(qp.point());
			Aloc.block(0, 0, B_sz, B_sz) += qp.weight() * phi * phi.transpose();
		}
	
		auto qps_PsiT = disk::integrate(msh, cl, 2*PsiTdeg);
		for (auto& qp : qps_PsiT)
		{
			auto trial = PsiT_basis.eval_functions(qp.point());
			auto test = B_basis.eval_divergences(qp.point());
			Aloc.block(0, B_sz, B_sz, PsiT_sz) += qp.weight() * test * trial.transpose();
		}
		Aloc.block(B_sz, 0, PsiT_sz, B_sz) = Aloc.block(0, B_sz, B_sz, PsiT_sz).transpose();
	}
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