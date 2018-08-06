#include "gp_transition _model.h"


void printMatrix(Matrix M) {
    cout << "[ ";
    for (int i = 0; i < M.size(); i++) {
        for (int j = 0; j < M[i].size(); j++)
            cout << M[i][j] << " ";
        cout << endl;
    }
    cout << " ]\n";
}

void printVector(Vector M) {
    cout << "[ ";
    for (int i = 0; i < M.size(); i++) {
            cout << M[i] << ", ";
    }
    cout << "]\n";
}

gp_transition_model::gp_transition_model(int mode, int state_dim, int action_dim ) : KDtree_(2, {{0,0},{1,1}}, 1) { // Dummy construction of the kd-tree class

    state_dim_ = state_dim;
    action_dim_ = action_dim;
    dim_ = 2*state_dim_ + action_dim_;
    mode_ = mode;

    string fileName = "../data/data_25_train_" + to_string(mode_) + ".db";
    // string fileName = "../data/toyData.db";
    load_data(fileName, Xtraining_, Ytraining_);

    // normalize data
    xmax_X_ = Max(Xtraining_);
    xmin_X_ = Min(Xtraining_);
    xmax_Y_ = Max(Ytraining_);
    xmin_Y_ = Min(Ytraining_);
    for (int i = 0; i < state_dim_; i++) {
        double tmp = std::max(xmax_X_[i], xmax_Y_[i]);
        xmax_X_[i] = xmax_Y_[i] = tmp;
        tmp = std::min(xmin_X_[i], xmin_Y_[i]);
        xmin_X_[i] = xmin_Y_[i] = tmp;
    }
    normz(Xtraining_, xmax_X_, xmin_X_);
    normz(Ytraining_, xmax_Y_, xmin_Y_);

    // Re-construction of the kd-tree with the real data set
	KDtree_.~KDTreeVectorOfVectorsAdaptor();
    apply_weights(apply_W);
	new(&KDtree_) my_kd_tree_t(state_dim_+action_dim_, Xtraining_nn_, 10);
	//KDtree_ = my_kd_tree_t(2, Obs, 10); // This is another option that did not work
	KDtree_.index->buildIndex();

    cout << KDtree_.kdtree_get_point_count() << " points were loaded to the kd-tree." << endl;

    Scales_.resize(state_dim_);
    Sigmas_.resize(state_dim_);
    Widths_.resize(state_dim_);  
    if (opt_H_param) 
        optimize_hparam(10000); // Run optimization on the current data
    else
        set_optimal_hparam(); // use pre-computed values based
}

void gp_transition_model::apply_weights(bool apply) {

    W_.resize(state_dim_+action_dim_);
    switch (mode_) {
        case 1: 
            W_ = {3,3,1,1};
            break;
        case 4: 
            W_ = {3,3,1,1,1,1};
            break;
        case 5: 
            W_ = {3,3,1,1,1,1};
            break;
        case 8: 
            W_ = {3,3,1,1,1,1,1,1};
            break;
    }

    Xtraining_nn_ = Xtraining_;
    if (apply) {
        for (int i = 0; i < Xtraining_nn_.size(); i++) {
            for(int j = 0; j < Xtraining_nn_[i].size(); j++)
                Xtraining_nn_[i][j] *= sqrt(W_[j]);  // Instead of doing weighted Euclidean distance 
        }
    }
}

void gp_transition_model::load_data(string fileName, Matrix &X, Matrix &Y, int max_data) {

    ifstream myfile;
    myfile.open(fileName);

    double t;
    int k = 0;
    while(!myfile.eof() && k < max_data) {
        Vector x, y;
        for (int i = 0; i < state_dim_+action_dim_; i++) { // Get current state and action
            myfile >> t;
            x.push_back(t);
        }
        for (int i = 0; i < state_dim_; i++) { // Get next state
            myfile >> t;
            y.push_back(t);
        }
        X.push_back(x);
        Y.push_back(y); 
        k++;
    }
    X.pop_back();
    Y.pop_back();

    cout << "Loaded data of " << X.size()+1 << " instances.\n";
}

void gp_transition_model::set_optimal_hparam() {

    switch (mode_) {
        case 1:
            // Based on optimization of random 1000 points
            Sigmas_ = {0.00011197, 0.000180151};
            Widths_ = {0.304855, 0.0993056};
            Scales_ = {0.698919, 0.380728};
            break;
        case 2:
            break;
        case 3:
            break;
        case 4:
            // Based on optimization of random 2000 points
            Sigmas_ = {9.51268e-05, 0.00017269, 0.000491737, 0.000605078};
            Widths_ = {0.280783, 0.53682, 0.493897, 0.494171};
            Scales_ = {0.383646, 0.365266, 0.412561, 0.474076};
            break;
        case 5:
            // Based on optimization of random 2000 points
            //Sigmas_ = {9.34607e-05, 0.000174265, 0.0106599, 0.0107837};
            Sigmas_ = {0.001, 0.001, 0.0211, 0.0169};
            Widths_ = {1.62301, 2.61356, 3.49915, 3.8618};
            Scales_ = {0.452675, 0.444676, 0.399524, 0.526306};
            // Widths_ = {0.9008, 0.8748, 0.7560, 1.3896}; // As reported in the opt. print.
            break;
        case 6:
            break;
        case 7:
            break;
        case 8:
            // Based on optimization of random 2000 points
            Sigmas_ = {9.31876e-05, 0.000173586, 0.000511495, 0.000635967, 0.0104902, 0.0107153};
            Widths_ = {2.85676, 3.28172, 16.1439, 15.7976, 3.17805, 3.65557};
            Scales_ = {0.436163, 0.398178, 0.696933, 0.769409, 0.411688, 0.51681};
            break;
        default:

            break;
    }

}

void gp_transition_model::optimize_hparam(int n) {
    
    Vector indices;
    for (int i = 0; i < Xtraining_.size(); ++i) indices.push_back(i);
    random_shuffle(indices.begin(), indices.end());
    Matrix XX, YY;
    for (int i = 0; i < n; i++) {
        XX.push_back( Xtraining_[i] );
        YY.push_back( Ytraining_[i] );
    }

    SGMatrix<float64_t> X(state_dim_+action_dim_, XX.size());
    X = Matrix2SGMatrix(XX);

    cout << "Initiated hyper-parameters optimization using " << n << " sampled points.\n";

    for (int i = 0; i < state_dim_; i++) {

        cout << "Optimizing for the " << i+1 << "th dimension..." << endl;

        SGVector<float64_t> y(YY.size());
        y = Vector2SGVector(get_column(YY, i));

        // shogun representation 
        CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
        CRegressionLabels* label_train=new CRegressionLabels(y);

        // specity GPR with exact inference 
        float64_t shogun_sigma=0.2; // <= width
        CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
        CZeroMean* mean=new CZeroMean();
        CGaussianLikelihood* lik=new CGaussianLikelihood();

        lik->set_sigma(1);
        CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train, mean, label_train, lik);
        CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

        CGradientCriterion* crit = new CGradientCriterion();
        CGradientEvaluation* grad=new CGradientEvaluation(gpr, feat_train, label_train, crit);
        grad->set_function(inf);
        // gpr->print_modsel_params();
        CGradientModelSelection* grad_search=new CGradientModelSelection(grad);
        CParameterCombination* best_combination=grad_search->select_model(print_opt_status);
        best_combination->print_tree();
        best_combination->apply_to_machine(gpr);

        Scales_[i] = inf->get_scale(); // set_scale
        Widths_[i] = kernel->get_width(); // set_width
        Sigmas_[i] = lik->get_sigma(); // set_sigma
    }

    cout << "Likelihood sigmas: "; printVector(Sigmas_);
    cout << "Kernel widths: "; printVector(Widths_);
    cout << "Inf. scales: "; printVector(Scales_); 
}

void gp_transition_model::predict(Vector state, Vector action) {

    state.insert( state.end(), action.begin(), action.end() );

    predict(state);
}

void gp_transition_model::predict(Vector state) {

    SGMatrix<float64_t> Xtrain_nn(state_dim_ + action_dim_, K_);
    Matrix X_data_nn, Y_data_nn;

    getNN(state, X_data_nn, Y_data_nn);
    Xtrain_nn = Matrix2SGMatrix(X_data_nn);

    SGMatrix<float64_t> Xtest(state_dim_+action_dim_, 1);
    Xtest = Vector2SGMatrix(state);

    mu_.clear();
    sigma_.clear();
    for (int i = 0; i < state_dim_; i++) { 

        SGVector<float64_t> y_train_nn(K_);
        y_train_nn = Vector2SGVector(get_column(Y_data_nn, i));

        // shogun representation 
        CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(Xtrain_nn);
        CRegressionLabels* label_train=new CRegressionLabels(y_train_nn);
        CDenseFeatures<float64_t>* feat_test1=new CDenseFeatures<float64_t>(Xtest);
        CDenseFeatures<float64_t>* feat_test2=new CDenseFeatures<float64_t>(Xtest);

        // specity GPR with exact inference 
        CGaussianKernel* kernel=new CGaussianKernel(10, Widths_[i]);
        CZeroMean* mean=new CZeroMean();
        CGaussianLikelihood* lik=new CGaussianLikelihood();

        lik->set_sigma(Sigmas_[i]);
        CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train, mean, label_train, lik);
        CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);
        inf->set_scale(Scales_[i]);

        // Optimization
        CGradientCriterion* crit = new CGradientCriterion();
        CGradientEvaluation* grad=new CGradientEvaluation(gpr, feat_train, label_train, crit);
        grad->set_function(inf);
        // gpr->print_modsel_params();
        CGradientModelSelection* grad_search=new CGradientModelSelection(grad);
        CParameterCombination* best_combination=grad_search->select_model(print_opt_status);
        best_combination->print_tree();
        best_combination->apply_to_machine(gpr);

        // perform inference
        // CRegressionLabels* predictions=gpr->apply_regression(feat_test3);
        SGMatrix<float64_t> mu = gpr->get_mean_vector(feat_test1);
        SGMatrix<float64_t> s2 = gpr->get_variance_vector(feat_test2);

        // mu.display_matrix("mu");
        // s2.display_matrix("s2");
        mu_.push_back(mu[0]);
        sigma_.push_back(sqrt(s2[0]));
    }

    cout << "Mean: "; printVector(mu_);
    cout << "std: ";  printVector(sigma_);

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

    for (int i = 0, j = 0; j < n; i+=m, j++) 
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
    SGMatrix<float64_t> Y(n, 1);

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

void gp_transition_model::normz(Matrix &X, Vector xmax, Vector xmin) {

    int d = X[0].size();

    for (int j = 0; j < d; j++)
        for (int i = 0; i < X.size(); i++)
            X[i][j] = (X[i][j] - xmin[j]) / (xmax[j]-xmin[j]);

}

Vector gp_transition_model::Max(Matrix M) { 

    int d = M[0].size();
    Vector v(d, -1e9);

    for (int j = 0; j < d; j++) 
        for (int i = 0; i < M.size(); i++)
            if (M[i][j] > v[j])
                v[j] = M[i][j];

    return v;
}

Vector gp_transition_model::Min(Matrix M) { 

    int d = M[0].size();
    Vector v(d, 1e9);

    for (int j = 0; j < d; j++) 
        for (int i = 0; i < M.size(); i++)
            if (M[i][j] < v[j])
                v[j] = M[i][j];

    return v;
}

void gp_transition_model::kNeighbors(my_kd_tree_t& mat_index, Vector query, kNeighborSoln& soln, size_t num_results, bool remove_1st_neighbor){
	// do a knn search 

	if (remove_1st_neighbor)
		num_results += 1;

	vector<size_t> ret_indexes(num_results);
	Vector out_dists_sqr(num_results);

	nanoflann::KNNResultSet<double> resultSet(num_results);
	resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

	Vector query_pnt = query;
	mat_index.index->findNeighbors(resultSet, &query_pnt[0], nanoflann::SearchParams(10));
    // mat_index.index->radiusSearch(&query_pnt[0], ..., ); // For radius search - See nanoflann.hpp, line 945

	VectorInt rtn_values(ret_indexes.begin(), ret_indexes.end());

	if (remove_1st_neighbor) {
		rtn_values.erase(rtn_values.begin()); // Remove first node that is itself.
		out_dists_sqr.erase(out_dists_sqr.begin()); // Remove first node that is itself.
	}

	Vector out_dists(out_dists_sqr.size());
	for (int i = 0; i < out_dists_sqr.size(); i++)
		out_dists[i] = sqrt(out_dists_sqr[i]);

	// If some error pops, try this
	//soln.neighbors.resize(rtn_values.size());
	//soln.dist.resize(rtn_values.size());

	soln.neighbors = rtn_values;
	soln.dist = out_dists;
}

void gp_transition_model::getNN(Vector state_action, Matrix &X_data_nn, Matrix &Y_data_nn) {
    
    // Apply weights for the weighted euclidean distance
    if (apply_W)
        for (int i = 0; i < state_action.size(); i++) 
            state_action[i] *= sqrt(W_[i]);

    kNeighborSoln soln;
	kNeighbors(KDtree_, state_action, soln, K_);

    X_data_nn.clear();
    Y_data_nn.clear();
    for (int i = 0; i < soln.neighbors.size(); i++) {
        X_data_nn.push_back(Xtraining_[soln.neighbors[i]]);
        Y_data_nn.push_back(Ytraining_[soln.neighbors[i]]);
    }  
}

Vector gp_transition_model::propagate(Vector s, Vector a, bool deterministic) {

    predict(s, a);
    Vector mu = get_mu();
    Vector sigma = get_sigma();

    Vector s_next(state_dim_);

    if (!deterministic) 
        for (int i = 0; i < state_dim_; i++) {
            std::normal_distribution<double> distribution(mu[i],sigma[i]);
            s_next[i] = distribution(generator);
        }
    else
        s_next = mu;

    return s_next;
}

// Run prediction on a testing set
void gp_transition_model::pred_path() {

    Matrix X, Y, Ypred;

    string fileName = "../data/data_25_test_" + to_string(mode_) + ".db";
    // string fileName = "../data/toyDataPath.db";
    load_data(fileName, X, Y);
    normz(X, xmax_X_, xmin_X_);
    normz(Y, xmax_Y_, xmin_Y_);

    Vector s(&X[0][0], &X[0][state_dim_]);

    Ypred.push_back(s);
    double loss = 0;
    for (int i = 0; i < X.size(); i++) {
        cout << "Prediction node " << i+1 << " of " <<  X.size()+1 << " ...\n";

        Vector a(&X[i][state_dim_], &X[i][state_dim_+action_dim_]);

        propagate(s, a);
        Vector s_next = get_mu();

        Ypred.push_back(s_next);
        s = s_next;

        loss += norm(Y[i], s_next);
    }

    cout << "MSE: " << loss/X.size() << endl;

    // Log the paths in file
    ofstream myfile;
    myfile.open ("./paths/path.txt");

	for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X[i].size(); j++)
            myfile << X[i][j] << " ";
        for (int j = 0; j < Ypred[i].size(); j++)
            myfile << Ypred[i][j] << " ";
        myfile << endl;
    }

    myfile.close();
}

double gp_transition_model::norm(Vector a, Vector b) {

    if (a.size() != b.size())
        return -1;

    double sum = 0;
    for (int i = 0; i < a.size(); i++)
        sum += (a[i]-b[i])*(a[i]-b[i]);

    return sqrt(sum);
}


int main() {

    init_shogun_with_defaults();

    if (1) {
        int mode = 5;
        Vector state_dims = {2, 6, 10, 4, 4, 12, 14, 6};

        gp_transition_model gp(mode, state_dims[mode-1], 2);

        // gp.predict({0.5406, 0.2978},{1, 1});
        gp.pred_path();
        }
    else {
        // for (int i = 1; i < 9; i++) 
        {
            int mode = 8;
            Vector state_dims = {2, 6, 10, 4, 4, 12, 14, 6};

            gp_transition_model gp(mode, state_dims[mode-1], 2);

            cout << "****** Optimizing for mode " << mode << " *******\n";
            gp.optimize_hparam(2000);
        }
    }
    

    exit_shogun();

    return 0;
}