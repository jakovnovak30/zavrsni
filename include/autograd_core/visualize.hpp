/**
 * @file
 * @brief definicija i implementacija funkcije autograd::visualize
 * @author Jakov Novak
 */

#include <cstdlib>
#include <string>

#include <graphviz/cgraph.h>
#include <graphviz/gvcext.h>
#include <graphviz/gvc.h>

#include "expression.hpp"

namespace autograd {
  template <typename T>
  void visualize(const Expression<T> &expr, const std::string &filePath, bool preview = false) {
    GVC_t *context = gvContext();

    char *graph_name = (char *) "Graf test";
    Agraph_t *output_graph = agopen(graph_name, Agdirected, nullptr);
    
    Agnode_t *out_node = agnode(output_graph, (char *) "out_node", 1);
    expr.addSubgraph(output_graph, out_node);

    gvLayout(context, output_graph, "dot");
    gvRenderFilename(context, output_graph, "png", filePath.c_str());

    if(preview) {
      system(("xdg-open " + filePath + " 2> /dev/null").c_str());
    }

    agfree(output_graph, nullptr);
    gvFreeLayout(context, output_graph);
    gvFreeContext(context);
  }
}
