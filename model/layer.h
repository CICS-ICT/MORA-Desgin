// mora layer type classes
// Created by blurSong on 2021/11/27.
//

#ifndef MORA_DESIGN_LAYERS_H
#define MORA_DESIGN_LAYERS_H

#include <memory>
#include <string>

/*
 * mora_layer_param_dicts = {
    'IC': 'input_channel',
    'OC': 'output_channel',
    'FS': 'feature_size',
    'KS': 'kernel_size',
    'STR': 'stride',
    'TYP': 'layer_type',
    'RP': 'relu_or_relu&pooling',
    'APD': 'appending_index',
}
 * mora_layer_type_dicts = {
    0: "Linear", //MVM
    1: "CONV",
    2: "DWCONV",
    3: "Residual",
    4: "Batchnorm",
    5: "TRCONV",
    6: "NGCONV",
    7: "VDP",
    8: "VADD"
    9: "GEMM" //MMM
}
 * mora_non-linear_layer_type_dicts = {
    0: "relu",
    1: "tanh",
    2: "sigmoid",
    3: "pooling",
    4: "softmax1d",
    5: "softmax2d"
}
 */

namespace mora {
    namespace model {

        // enum class ModelType {MLP, CNN, RNN, LSTM, ATTENTION};
        enum class LinearLayerType {
            Linear, CONV, DWCONV, Residual, Batchnorm, TRCONV, NGCONV, VDP, VADD, GEMM
        };
        enum class NonlinearLayerType {
            Relu, Tanh, Sigmoid, Pooling, Softmax1D, Softmax2D
        };
        struct LayerFeature {
            int IC;
            int OC;
            int FS;
            int KS;
            int STR;
            int RP;
            int APD;
        };

        class Layer {
        protected:
            std::string layer_name;
            int layer_index;

        public:
            Layer(const std::string ln, const int li);
            virtual bool CheckFeatures();
            virtual ~Layer();
        };

        class LinearLayer : public Layer {
        protected:
            LinearLayerType layer_type;
            LayerFeature layer_feature;

        public:
            LinearLayer(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            virtual bool CheckFeatures();
            virtual ~LinearLayer();
        };

        class Linear : public LinearLayer {
        private:
            int x_dimension;
            int y_dimension;

        public:
            Linear(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            Linear(const int x, const int y, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~Linear();
        };

        class CONV : public LinearLayer {
        private:
            int input_channel;
            int output_channel;
            int feature_size;
            int kernel_size;
            int stride;

        public:
            CONV(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            CONV(const int inc, const int outc, const int fs, const int ks, const int str, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~CONV();
        };

        class DWCONV : public LinearLayer {
        private:
            int chennel;
            int feature_size;
            int kernel_size;
            int stride;

        public:
            DWCONV(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            DWCONV(const int ch, const int fs, const int ks, const int str, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~DWCONV();
        };

        class Residual : public LinearLayer {
        private:
            int chennel;
            int feature_size;

        public:
            Residual(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            Residual(const int ch, const int fs, const int  idx1, const int idx2, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~Residual();
        };

        class Batchnorm : public LinearLayer {
        private:
            int chennel;
            int feature_size;

        public:
            Batchnorm(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            Batchnorm(const int ch, const int fs, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~Batchnorm();
        };

        class TRCONV : public LinearLayer {}; //TODO

        class NGCONV : public LinearLayer {}; //TODO

        class VDP : public LinearLayer {
        private:
            int dimension;

        public:

            VDP(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            VDP(const int dim, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~VDP();
        };

        class VADD : public LinearLayer {
        private:
            int dimension;

        public:
            VADD(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            VADD(const int dim, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~VADD();
        };

        class GEMM : public LinearLayer { // MK mul KN
        private:
            int m_dimension;
            int k_dimension;
            int n_dimension;

        public:
            GEMM(const std::string ln, const int li, const LinearLayerType lt, const LayerFeature lf);
            GEMM(const int m, const int k, const int n, const std::string ln, const int li, const LinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~GEMM();
        };

        class NonlinearLayer : public Layer {
        protected:
            NonlinearLayerType layer_type;

        public:
            NonlinearLayer(const std::string ln, const int li, const NonlinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~NonlinearLayer();
        };

        class Pooling : public NonlinearLayer {
        private:
            int channel;
            int feature_size;
            int kernel_size;
            int stride;

        public:
            Pooling(const int ch, const int fs, const int ks, const int str, const std::string ln, const int li, const NonlinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~Pooling();
        };

        class Softmax1D : public NonlinearLayer {
        private:
            int dimension;

        public:
            Softmax1D(const int dim, const std::string ln, const int li, const NonlinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~Softmax1D();
        };

        class Softmax2D : public NonlinearLayer {
        private:
            int row_dimension;
            int col_dimension;

        public:
            Softmax2D(const int row, const int col, const std::string ln, const int li, const NonlinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~Softmax2D();
        };

        class ActivationLayer : public NonlinearLayer {
        private:
            int channel;
            int feature_size;

        public:
            ActivationLayer(const int ch, const int fs, const std::string ln, const int li, const NonlinearLayerType lt);
            virtual bool CheckFeatures();
            virtual ~ActivationLayer();
        };

        class Relu : public ActivationLayer {
            Relu(const int ch, const int fs, const std::string ln, const int li, const NonlinearLayerType lt);
            virtual ~Relu();
        };

        class Tanh : public ActivationLayer {
            Tanh(const int ch, const int fs, const std::string ln, const int li, const NonlinearLayerType lt);
            virtual ~Tanh();
        };

        class Sigmoid : public ActivationLayer {
            Sigmoid(const int ch, const int fs, const std::string ln, const int li, const NonlinearLayerType lt);
            virtual ~Sigmoid();
        };

    }
}

#endif //MORA_DESIGN_LAYERS_H
