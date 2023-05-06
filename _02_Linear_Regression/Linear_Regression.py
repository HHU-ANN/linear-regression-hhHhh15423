# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y = read_data()
    weight = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y))
    return weight @ data
    
def lasso(data):
    mu = 1e-2;
opts = struct();
opts.method = 'grad_huber';
opts.verbose = 0;
opts.maxit = 4000;
opts.ftol = 1e-8;
opts.alpha0 = 1 / L;
[x, out] = LASSO_con(x0, A, b, mu, opts);
f_star = min(out.fvec);

opts.verbose = 0;
opts.maxit = 400;
if opts.verbose
    fprintf('\nmu=1e-2\n');
end
[x, out] = LASSO_con(x0, A, b, mu, opts);
data2 = (out.fvec - f_star)/f_star;
k2 = min(length(data2),400);
data2 = data2(1:k2);

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
