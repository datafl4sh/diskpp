/*
 * DISK++, a template library for DIscontinuous SKeletal methods.
 *
 * Matteo Cicuttin (C) 2026
 * matteo.cicuttin@polito.it
 *
 * Politecnico di Torino - DISMA
 * Dipartimento di Matematica
 */

#include <tuple>
#include <vector>
#include <array>
#include <fstream>

#include "diskpp/mesh/mesh.hpp"
#include "diskpp/geometry/geometry.hpp"
#include "diskpp/loaders/mesh_loader.hpp"

namespace disk
{

template<typename T, size_t N>
class delynoi_mesh_loader
{
    static_assert(N == 2, "Delynoi is a 2D-only mesh format");
};

enum class delynoi_parser_state {
    success,
    failure,
};

template<typename T>
class delynoi_mesh_loader<T, 2> : public mesh_loader<generic_mesh<T, 2>>
{
    typedef generic_mesh<T, 2>               mesh_type;
    typedef typename mesh_type::point_type   point_type;
    typedef typename mesh_type::node_type    node_type;
    typedef typename mesh_type::edge_type    edge_type;
    typedef typename mesh_type::surface_type surface_type;

    std::ifstream ifs;
    size_t        current_line;

    std::vector<point_type>             m_points;
    std::vector<node_type>              m_nodes;
    std::vector<edge_type>              m_edges;              
    std::vector<std::vector<size_t>>    m_surfaces;

    bool
    failure(std::istringstream& ifs)
    {
        return ifs.rdstate() & std::ifstream::failbit;
    }

    delynoi_parser_state
    delynoi_parse_vertices()
    {
        size_t num_vertices;
        ifs >> num_vertices;

        if (this->verbose()) {
            std::cout << "Reading " << num_vertices << " vertices" << std::endl;
        }

        for (size_t i = 0; i < num_vertices; i++) {
            T x, y;
            ifs >> x >> y;

            m_points.push_back( point_type{x, y} );
            m_nodes.push_back( node_type(i) );
        }

        return delynoi_parser_state::success;
    }

    delynoi_parser_state
    delynoi_parse_edges()
    {
        size_t num_edges;
        ifs >> num_edges;

        if (this->verbose()) {
            std::cout << "Reading " << num_edges << " edges" << std::endl;
        }

        for (size_t i = 0; i < num_edges; i++) {
            size_t n0, n1;
            ifs >> n0 >> n1;
            auto ni0 = typename node_type::id_type(n0);
            auto ni1 = typename node_type::id_type(n1);
            m_edges.push_back( {ni0, ni1} );
        }

        return delynoi_parser_state::success;
    }
   
    delynoi_parser_state
    delynoi_parse_polygons()
    {
        size_t num_polys;
        ifs >> num_polys;

        if (this->verbose()) {
            std::cout << "Reading " << num_polys << " polygons" << std::endl;
        }

        for (size_t i = 0; i < num_polys; i++) {
            size_t num_vtx;
            ifs >> num_vtx;
            std::vector<size_t> nodes;
            for (size_t v = 0; v < num_vtx; v++) {
                size_t vtx;
                ifs >> vtx;
                nodes.push_back(vtx);
            }
            m_surfaces.push_back( std::move(nodes) );
        }

        return delynoi_parser_state::success;
    }

    bool
    parse(const std::string& filename)
    {
        ifs.open(filename);
        if (not ifs.is_open())
        {
            std::cout << "Can't open " << filename << std::endl;
            return false;
        }

        delynoi_parse_vertices();
        delynoi_parse_edges();
        delynoi_parse_polygons();

        return true;
    }

  public:
    static const char constexpr* expected_extension = "txt";

    delynoi_mesh_loader() : current_line(0) {}

    bool
    read_mesh(const std::string& filename)
    {
        if (this->verbose()) {
            std::cout << " *** READING DELYNOI MESH ***" << std::endl;
        }
        return parse(filename);
    }

    bool
    populate_mesh(mesh_type& msh)
    {
        if (this->verbose()) {
            std::cout << " *** POPULATING DELYNOI MESH ***" << std::endl;
        }
        auto storage = msh.backend_storage();

        /* Points */
        size_t nodes_size = m_points.size();
        storage->points   = std::move(m_points);

        /* Nodes */
        std::vector<node_type> nodes(nodes_size);
        for (size_t i = 0; i < nodes_size; i++)
            nodes[i] = node_type(disk::point_identifier<2>(i));

        storage->nodes = std::move(nodes);

        /* Edges */
        std::sort(m_edges.begin(), m_edges.end());
        m_edges.erase(std::unique(m_edges.begin(), m_edges.end()), m_edges.end());
        storage->edges = std::move(m_edges);

        /* Detect which ones are boundary edges */
        storage->boundary_info.resize(storage->edges.size());


        for (auto& s : m_surfaces)
        {
            std::vector<typename edge_type::id_type> surface_edges;
            for (size_t ni = 0; ni < s.size(); ni++)
            {
                auto n1 = typename node_type::id_type(s[ni]);
                auto n2 = typename node_type::id_type(s[(ni+1)%s.size()]);

                edge_type edge(n1, n2);
                auto  edge_id = find_element_id(storage->edges.begin(), storage->edges.end(), edge);
                if (!edge_id.first)
                {
                    std::cout << "Bad bug at " << __FILE__ << "(" << __LINE__ << ")" << std::endl;
                    return false;
                }

                surface_edges.push_back(edge_id.second);
            }

            auto surface = surface_type(surface_edges);
            surface.set_point_ids(s.begin(), s.end()); /* XXX: crap */
            storage->surfaces.push_back(surface);
        }

        storage->subdomain_info.resize(m_surfaces.size());
        
        mark_internal_faces(msh);

        if (this->verbose())
        {
            std::cout << "Nodes:    " << storage->nodes.size() << std::endl;
            std::cout << "Edges:    " << storage->edges.size() << std::endl;
            std::cout << "Faces:    " << storage->surfaces.size() << std::endl;
            std::cout << "BND info: " << storage->boundary_info.size() << std::endl;
            std::cout << "SD info:  " << storage->subdomain_info.size() << std::endl;
        }

        return true;
    }
};

}