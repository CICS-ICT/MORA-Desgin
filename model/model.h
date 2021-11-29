// Model structure 
// Created by blurSong on 2021/11/27.
//

#ifndef MORA_DESIGN_MODEL_H
#define MORA_DESIGN_MODEL_H

#include <memory>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>

#include "layer.h"

namespace boost {
    enum edge_flowtype_t {
        edge_flowtype
    };
    BOOST_INSTALL_PROPERTY(edge, flowtype);
}

namespace mora {
    namespace model {

        enum class ModelType { MLP, CNN, RNN, LSTM, ATTENTION };
        enum class FlowType { conv, linear, activate, downsample, batchnorm, skipcon, concat, softmax };
        struct ModelParam {
            int linear_layers;
            int nonlinear_layers;
            long param_nums;
        };
        typedef boost::property<boost::edge_flowtype_t, FlowType> EdgeProperty;

        class Model {
        protected:
            std::string model_name;
            ModelType model_type;
            ModelParam mode_param;
            boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeProperty> LayerDAG;

        public:
            Model(const std::string mn, const ModelType mt);
            virtual bool CheckModel();
            virtual ~Model();
        };

        class MLP : public Model {
        protected:

        }
    }
}


#endif //MORA_DESIGN_MODEL_H
