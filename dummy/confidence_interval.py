
import numpy as np
import scipy.stats as st
from scipy.special import logit, expit
import os,sys
import argparse
import matplotlib.pyplot as plt

def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI


def plot_global_results(glo_result_list, figure_result_dir, metric):

    xdata = range(len(glo_result_list)-1)
    ydata = glo_result_list[1:]


    # fig = plt.figure(figsize=(7.2, 4.2))
    fig = plt.figure(figsize=(4,3))

    plt.plot(xdata, ydata, 'go-', label = "FedAvg")
    plt.axhline(y= glo_result_list[0], color='r', linestyle='--' , label = "Before aggregation")
    plt.legend(loc = "center right", fontsize=11)
    plt.ylabel("Global %s" %metric, fontsize=15)
    plt.xlabel("Update iteration", fontsize=15)
    plt.xticks(xdata, range(1, len(glo_result_list)), fontsize=12, rotation=70 )
    plt.yticks(fontsize=12)
    # plt.title("Global %s across update iteration" %metric)
    # plt.ylim([0,500])

    fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s.png' %metric), dpi=500 ,bbox_inches='tight')
    fig.savefig(os.path.join(figure_result_dir, 'global_eval_%s.eps' %metric), dpi=500 ,bbox_inches='tight')


    plt.close()

    return

# get path of current directory
# current_path = Path(__file__).parent
current_dir = os.path.dirname(os.path.abspath(__file__))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_fold', default= 10, help='k of k-fold for ci', type=int)
    parser.add_argument('-n_train', default= 1310, help='k of k-fold for ci', type=int)
    parser.add_argument('-n_test', default= 164, help='k of k-fold for ci', type=int)
    args = parser.parse_args()
    # print("Arguments:",args)
    n_fold = args.n_fold
    
    N_train = args.n_train
    N_test = args.n_test

    ttest_dir = os.path.join(current_dir, "ttest_ci")

    global_ci_list = []
    local_ci_list = []

    for k in range(n_fold):
        global_k = np.load(os.path.join(ttest_dir, "global_ci_%s.npy" %k))
        local_k = np.load(os.path.join(ttest_dir, "local_ci_%s.npy" %k))
        print ("global_k", global_k)
        global_ci_list.append(global_k)
        local_ci_list.append(local_k)

    global_results_array = np.array(global_ci_list)
    local_results_array = np.array(local_ci_list)

    print ("global_results_array", global_results_array)

    glo_result_avg_list = []

    for u in range(global_results_array.shape[1]):
        result_u = global_results_array[:, u]
        print ("u", u)
        print ("result_u", result_u)

        result_u_avg = np.average(result_u)
        print ("result_u_avg", result_u_avg)
        glo_result_avg_list.append(result_u_avg)



        CI = compute_confidence(result_u, N_train, N_test, alpha=0.95)
        print ("CI", CI)

    plot_global_results(glo_result_avg_list, ttest_dir, "C-statistic")

if __name__ == '__main__':
    main()    

