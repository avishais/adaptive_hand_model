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

#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <time.h>
#include <string>

using namespace std;
using namespace shogun;

typedef vector<vector<double>> Matrix;
typedef vector<double> Vector;




class gp_transition_model {
    public:
        gp_transition_model(int, int, int = 2);

        // ~gp_transition_model();

        void regression();

        void printSGMatrix(SGMatrix<float64_t>, int = 6);

        void predict(Vector state, Vector action);


    // private:

        SGMatrix<float64_t> Matrix2SGMatrix(Matrix);
        SGVector<float64_t> Vector2SGVector(Vector);
        SGMatrix<float64_t> Vector2SGMatrix(Vector);

        void load_data();

        int dim_, state_dim_, action_dim_, mode_;

        Matrix Xtraining_; // All current_state+action instances
        Matrix Ytraining_; // All next_states instances

        Vector get_column(Matrix, int);

        int K = 100; // Number of NN

        bool opt_H_param = false;


};