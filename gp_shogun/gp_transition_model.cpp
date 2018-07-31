#include "gp_transition _model.h"

gp_transition_model::gp_transition_model(int mode, int state_dim, int action_dim ) {

    state_dim_ = state_dim;
    action_dim_ = action_dim;
    dim_ = 2*state_dim_ + action_dim_;
    mode_ = mode;

    load_data();

    // SGMatrix<float64_t> X(dim_vectors, n_train);
	// SGMatrix<float64_t> X_test(dim_vectors, n_test);
	// SGVector<float64_t> Y(n_train);

}

void gp_transition_model::load_data() {

    string fileName = "../data/data_25_" + to_string(mode_) + ".db";
    ifstream myfile;
    myfile.open(fileName);

    double t;
    int k = 0;
    while(k < 100) {//!myfile.eof()) {
        Vector x, y;
        for (int i = 0; i < state_dim_+action_dim_; i++) { // Get current state and action
            myfile >> t;
            x.push_back(t);
        }
        for (int i = 0; i < state_dim_; i++) { // Get next state
            myfile >> t;
            y.push_back(t);
        }
        Xtraining_.push_back(x);
        Ytraining_.push_back(y); 
        k++;
    }
}

void gp_transition_model::predict(Vector state, Vector action) {

    state.insert( state.end(), action.begin(), action.end() );

    SGMatrix<float64_t> Xtrain_nn(state_dim_+action_dim_, K);
    Xtrain_nn = Matrix2SGMatrix(Xtraining_);

    SGMatrix<float64_t> Xtest(state.size(), 2);
    Xtest = Vector2SGMatrix(state);
    // Matrix F;
    // F.push_back(state);
    // F.push_back(state);
    // Xtest = Matrix2SGMatrix(F);


    // specity GPR with exact inference 
    float64_t shogun_sigma=0.2;//sigma*sigma*2;
    CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
    CZeroMean* mean=new CZeroMean();
    CGaussianLikelihood* lik=new CGaussianLikelihood();


    for (int i = 0; i < state_dim_; i++) {

        SGVector<float64_t> y_train_nn(100);
        y_train_nn = Vector2SGVector(get_column(Ytraining_, i));

        // shogun representation 
        CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(Xtrain_nn);
        CRegressionLabels* label_train=new CRegressionLabels(y_train_nn);
        CDenseFeatures<float64_t>* feat_test1=new CDenseFeatures<float64_t>(Xtest);
        CDenseFeatures<float64_t>* feat_test2=new CDenseFeatures<float64_t>(Xtest);

        lik->set_sigma(1);
        CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train, mean, label_train, lik);
        CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

        if (opt_H_param) {
            CGradientCriterion* crit = new CGradientCriterion();
            CGradientEvaluation* grad=new CGradientEvaluation(gpr, feat_train, label_train, crit);
            grad->set_function(inf);
            // gpr->print_modsel_params();
            CGradientModelSelection* grad_search=new CGradientModelSelection(grad);
            CParameterCombination* best_combination=grad_search->select_model(true);
            best_combination->print_tree();
            best_combination->apply_to_machine(gpr);
        }

        // perform inference
        // CRegressionLabels* predictions=gpr->apply_regression(feat_test3);
        SGMatrix<float64_t> mu = gpr->get_mean_vector(feat_test1);
        // SGMatrix<float64_t> s2 = gpr->get_variance_vector(feat_test2);


    }

}


Vector gp_transition_model::get_column(Matrix M, int col) {

    Vector v;
    for (int i = 0; i < M.size(); i++) 
        v.push_back(M[i][col]);

    return v;
}



SGMatrix<float64_t> gp_transition_model::Matrix2SGMatrix(Matrix M) {

    int n = M.size();
    int m = M[0].size();

    SGMatrix<float64_t> X(m, n);

    for (int i = 0, j = 0; j < n; i+=dim_, j++) 
        for (int k = 0; k < m; k++) 
            X[i+k] = M[j][k];
  
    return X;
}


SGVector<float64_t> gp_transition_model::Vector2SGVector(Vector V) {

    int n = V.size();
    SGVector<float64_t> Y(n);

    for (int i = 0; i < n; i++)
        Y[i] = V[i];

    return Y;

}

SGMatrix<float64_t> gp_transition_model::Vector2SGMatrix(Vector V) {

    int n = V.size();
    SGMatrix<float64_t> Y(1, n);

    for (int i = 0; i < n; i++)
        Y[i] = V[i];

    return Y;

}

void gp_transition_model::printSGMatrix(SGMatrix<float64_t> M, int feature_dim) {

    // M.display_matrix("M");

    for (int i = 0, j = 0; j < M.size()/feature_dim; i+=dim_, j++) {
        for (int k = 0; k < feature_dim; k++) 
            cout << M[i+k] << " ";
        cout << endl;
    }

}


int main() {

    init_shogun_with_defaults();

    Matrix M = {{1,2},{4,5},{7,8},{10,11}};

    gp_transition_model gp(1, 2, 2);

    gp.predict({1,2},{3,4});

    // SGMatrix<float64_t> Y = gp.Matrix2SGMatrix(M);

    // gp.printSGMatrix(Y);

    exit_shogun();

    return 0;
}