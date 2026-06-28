import pickle
import os
import matplotlib.pyplot as plt
from models.baseline.GlobalPooling.vade_pooling.GlobalPool import GlobalPool


def draw_series(s):
    plt.plot(s, label='s')
    plt.xlabel('t')
    plt.ylabel('workload_minmax')
    # plt.title('trend clusters valuation')
    plt.show()

def build_gp(decomp_X, seasonal_cluster_res, args):
    cluster_num = args.nClusters
    gp = GlobalPool(cluster_num, cluster_num)
    gp.build_pool_seasonal(decomp_X, seasonal_cluster_res)

    pool_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pools'))
    os.makedirs(pool_dir, exist_ok=True)
    pool_path = os.path.join(pool_dir, 'global_pool_c{}_s{}_s{}.pkl'.format(cluster_num, args.series_len, args.step))
    pickle.dump(gp, open(pool_path, 'wb'))


if __name__ == "__main__":
    print('test')
