// For Shogun
#include <shogun/lib/config.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/base/init.h>
#include <shogun/evaluation/GradientCriterion.h>  
#include <shogun/evaluation/GradientEvaluation.h>  
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>

// For nn-search
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <time.h>
#include <string>
#include <algorithm>

using namespace std;
using namespace shogun;

typedef vector<vector<double>> Matrix;
typedef vector<double> Vector;
typedef vector< int > VectorInt;
typedef KDTreeVectorOfVectorsAdaptor< Matrix, double > my_kd_tree_t;

struct kNeighborSoln {
	VectorInt neighbors;
	Vector dist;
};


class gp_transition_model {
    public:
        gp_transition_model(int, int, int = 2);

        // ~gp_transition_model();

        void optimize_hparam();

        void printSGMatrix(SGMatrix<float64_t>, int = 6);

        void predict(Vector, Vector);
        void predict(Vector);

        void kNeighbors(my_kd_tree_t&, Vector, kNeighborSoln&, size_t, bool = false);

        void getNN(Vector state_action, Matrix &X_data_nn, Matrix &Y_data_nn);

        void pred_path();

        Vector propagate(Vector, Vector, bool = false);

        Vector get_mu() {
            return mu_;
        }

        Vector get_sigma() {
            return sigma_;
        }

    // private:

        SGMatrix<float64_t> Matrix2SGMatrix(Matrix);
        SGVector<float64_t> Vector2SGVector(Vector);
        SGMatrix<float64_t> Vector2SGMatrix(Vector);

        Vector Max(Matrix M);
        Vector Min(Matrix M);
        void normz(Matrix &X, Vector xmax, Vector xmin);

        double norm(Vector, Vector);

        void load_data(string filename, Matrix &X, Matrix &Y, int = 1e8);

        int dim_, state_dim_, action_dim_, mode_;

        Matrix Xtraining_; // All current_state+action instances
        Matrix Ytraining_; // All next_states instances

        Vector xmax_X_, xmin_X_, xmax_Y_, xmin_Y_; // For normalization

        Vector get_column(Matrix, int);

        int K_ = 100; // Number of NN

        bool opt_H_param = false;

        Vector mu_, sigma_;

        my_kd_tree_t KDtree_;

        std::default_random_engine generator;

        bool print_opt_status = false;
};