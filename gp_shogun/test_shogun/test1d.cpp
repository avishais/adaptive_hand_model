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

using namespace std;
using namespace shogun;

void log2file(index_t n_train, index_t n_test, SGMatrix<float64_t> X, SGMatrix<float64_t> Y, SGMatrix<float64_t> X_test, SGMatrix<float64_t> mu, SGMatrix<float64_t> s2) {
    ofstream myfile;
    myfile.open ("data.txt");

    for (index_t i = 0; i < n_test; i++ ) {
        if (i < n_train)
            myfile << X[i] << " " << Y[i] << " " << X_test[i] << " " << mu[i] << " " << s2[i] << endl;
        else
            myfile << 0 << " " << 0 << " " << X_test[i] << " " << mu[i] << " " << s2[i] << endl;
    }

    myfile.close();

}

void test()
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,0.1);

	/* create some easy regression data: 1d noisy sine wave */
	index_t n_train = 5;
    index_t n_test = 100;
	float64_t x_range=6;

	SGMatrix<float64_t> X(1, n_train);
	SGMatrix<float64_t> X_test(1, n_test);
	SGVector<float64_t> Y(n_train);

	for (index_t  i=0; i<n_train; ++i)
	{
		X[i]=CMath::random(0.0, x_range);
		Y[i] = std::sin(X[i]) + distribution(generator);
	}

    for (index_t  i=0; i<n_test; ++i)
        X_test[i]=(float64_t)i / n_test*(x_range+3);

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test1=new CDenseFeatures<float64_t>(X_test);
    CDenseFeatures<float64_t>* feat_test2=new CDenseFeatures<float64_t>(X_test);
    CDenseFeatures<float64_t>* feat_test3=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

    auto start = std::chrono::system_clock::now();

	/* specity GPR with exact inference */
	float64_t sigma=0.2;
	float64_t shogun_sigma=0.2;//sigma*sigma*2;
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
    CParameterCombination* best_combination=grad_search->select_model(true);
    // best_combination->print_tree();
    best_combination->apply_to_machine(gpr);

    // gpr->train();

	/* perform inference */
	CRegressionLabels* predictions=gpr->apply_regression(feat_test3);
    SGMatrix<float64_t> mu = gpr->get_mean_vector(feat_test1);
    SGMatrix<float64_t> s2 = gpr->get_variance_vector(feat_test2);

    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
    cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    log2file(n_train, n_test, X, Y, X_test, mu, s2);

    // auto eval = some<CMeanSquaredError>();
    // auto mse = eval->evaluate(labels_predict, labels_test);
    // auto marg_ll = inference_method->get_negative_log_marginal_likelihood();

	// SG_UNREF(predictions);
	SG_UNREF(gpr);
}

int main ()
{
init_shogun_with_defaults();

test();

exit_shogun();
return 0;
}