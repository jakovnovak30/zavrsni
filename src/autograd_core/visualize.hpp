#include "expression.hpp"
#include <fstream>
#include <graphviz/cgraph.h>
#include <graphviz/gvcext.h>
#include <string>
#include <graphviz/gvc.h>

namespace autograd {
  template <typename T>
  void visualize(const Expression<T> &expr, const std::string &filePath) {
    std::fstream file;
    file.exceptions(std::fstream::badbit | std::fstream::failbit);
    GVC_t *context = gvContext();

    char *graph_name = (char *) "Graf test";
    Agraph_t *output_graph = agopen(graph_name, Agdirected, nullptr);
    
    Agnode_t *out_node = agnode(output_graph, (char *) "out_node", 1);
    expr.addSubgraph(output_graph, out_node);

    gvLayout(context, output_graph, "dot");
    gvRenderFilename(context, output_graph, "png", "./output.png");
  }
}
