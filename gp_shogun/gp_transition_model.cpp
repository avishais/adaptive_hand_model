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
            cout << M[i] << " ";
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
	new(&KDtree_) my_kd_tree_t(state_dim_+action_dim_, Xtraining_, 10);
	//KDtree_ = my_kd_tree_t(2, Obs, 10); // This is another option that did not work
	KDtree_.index->buildIndex();

    cout << KDtree_.kdtree_get_point_count() << " points were loaded to the kd-tree." << endl;
    
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

    cout << "Loaded data of " << X.size() << " instances.\n";
}

void gp_transition_model::predict(Vector state, Vector action) {

    state.insert( state.end(), action.begin(), action.end() );

    predict(state);
}

void gp_transition_model::optimize_hparam() {

    Matrix XX, YY;
    for (int i = 0; i < 100; i++) {
        XX.push_back(Xtraining_[i]);
        YY.push_back(Ytraining_[i]);
    }

    SGMatrix<float64_t> X(state_dim_+action_dim_, XX.size());
    X = Matrix2SGMatrix(XX);

    SGVector<float64_t> y(YY.size());
    y = Vector2SGVector(get_column(YY, 0));

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

    cout << inf->get_scale() << endl;
    cout << kernel->get_width() << endl;
    cout << lik->get_sigma() << endl;
}

void gp_transition_model::predict(Vector state) {

    SGMatrix<float64_t> Xtrain_nn(state_dim_ + action_dim_, K_);
    Matrix X_data_nn, Y_data_nn;

    getNN(state, X_data_nn, Y_data_nn);
    Xtrain_nn = Matrix2SGMatrix(X_data_nn);

    SGMatrix<float64_t> Xtest(state_dim_+action_dim_, 1);
    Xtest = Vector2SGMatrix(state);

    // specity GPR with exact inference 
    float64_t shogun_sigma=0.2;//sigma*sigma*2;
    CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
    CZeroMean* mean=new CZeroMean();
    CGaussianLikelihood* lik=new CGaussianLikelihood();

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

        lik->set_sigma(1);
        CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train, mean, label_train, lik);
        CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

        if (opt_H_param) {
            CGradientCriterion* crit = new CGradientCriterion();
            CGradientEvaluation* grad=new CGradientEvaluation(gpr, feat_train, label_train, crit);
            grad->set_function(inf);
            // gpr->print_modsel_params();
            CGradientModelSelection* grad_search=new CGradientModelSelection(grad);
            CParameterCombination* best_combination=grad_search->select_model(print_opt_status);
            best_combination->print_tree();
            best_combination->apply_to_machine(gpr);
        }

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

	// Vector out_dists(out_dists_sqr.size());
	// for (int i = 0; i < out_dists_sqr.size(); i++)
	// 	out_dists[i] = sqrt(out_dists_sqr[i]);

	// If some error pops, try this
	//soln.neighbors.resize(rtn_values.size());
	//soln.dist.resize(rtn_values.size());

	soln.neighbors = rtn_values;
	// soln.dist = out_dists;
}

void gp_transition_model::getNN(Vector state_action, Matrix &X_data_nn, Matrix &Y_data_nn) {
    
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

    if (deterministic) 
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
    load_data(fileName, X, Y, 100);
    normz(X, xmax_X_, xmin_X_);
    normz(Y, xmax_Y_, xmin_Y_);

    Vector s(&X[0][0], &X[0][state_dim_]);
    
    Ypred.push_back(s);
    double loss = 0;
    for (int i = 0; i < X.size(); i++) {
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

    int mode = 5;
    Vector state_dims = {2, 6, 10, 4, 4, 12, 14, 6};

    gp_transition_model gp(mode, state_dims[mode-1], 2);

    gp.optimize_hparam();

    // gp.predict({0.573672950220049,	0.155298637748662,	0.197238658777120,	0.843283582089552},{1, 1});
    // gp.pred_path();

    exit_shogun();

    return 0;
}