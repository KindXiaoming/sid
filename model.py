import autograd.numpy as np
from autograd import grad, jacobian
from scipy.optimize import minimize
import copy

# generate basis file
# In our case, we are interested in polynomial bases. We genearte the basis file with the following code.
# But of course one can manually write/edit the basis file (formulas obey numpy syntex)
def create_ploy_basis_file(orders, num_variable, path="./basis.txt"):

    def _create_poly_basis(order, num_variable):
        order = order
        num = num_variable
        content = []
        for i in range(order):
            content.append(np.arange(num))
        #content = [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]]
        basis = np.transpose(np.array(np.meshgrid(*content)).reshape(order, num**order))
        basis = np.sort(basis, axis=1)
        #basis
        basis = np.array(list(set(tuple(map(tuple, basis)))))
        index = np.sum(num**np.arange(order)[::-1][np.newaxis,:]*basis, axis=1)
        basis = basis[np.argsort(index)]
        return basis
    
    basis_string = []
    
    for order in orders:
        basis = _create_poly_basis(order, num_variable)
        for j in range(basis.shape[0]):
            string = ""
            for i in range(order):
                if i == 0:
                    string += "x[%d]"%(basis[j][i])
                else:
                    string += "*x[%d]"%(basis[j][i])
            basis_string.append(string)
                
    np.savetxt(path, np.array(basis_string), fmt="%s")
    
    
def find_cq(f, x, bases, tol_cq=1e-4, tol_dep=1e-4, seed=0, sparse_run=10, sparse_tol=1e-32, max_iter=100):

    results = {}
    np.random.seed(seed)
    # generate data (computing the basis)
    num_basis = bases.shape[0]
    num_points = x.shape[1]
    assert num_points >= num_basis
    basis_data = []
    grad_data = []

    #### Computing bases and gradients #####
    print("#### Computing bases and gradients #####")
    for i in range(num_basis):
        basis_data.append(eval(bases[i]))
        def bases_sum(x):
            return np.sum(eval(bases[i]))
        batch_grad = grad(bases_sum)
        grad_data.append(batch_grad(x))

    basis_data = np.transpose(np.array(basis_data))
    grad_data = np.transpose(np.array(grad_data))

    f_data = np.transpose(f(x))

    f_grad_prod = np.einsum('ij,ijk->ik', f_data, grad_data)
    
    #### Solving thetas ####
    print("#### Solving thetas ####")
    u, s, v = np.linalg.svd(f_grad_prod)
    s = s/np.sum(s)
    ### out: s
    results['s_cq'] = copy.deepcopy(s)

    num_cq = np.sum(s<tol_cq)
    solutions = v[-num_cq:]
    # np.einsum("ij,kj->ki", f_grad_prod, solutions) # check solutions
    ### out: solutions
    results['sol_cq'] = copy.deepcopy(solutions)


    #### Sparsifying thetas ####
    print("#### Sparsifying thetas ####")
    for i in range(100000):
        if i == num_cq:
            n = i
            break
            
    if n == 0:
        print("no conservation law")
        return
            
    def vector2orth(V):
        V = V.reshape(n, n)
        I = np.eye(n, n)
        A = (V-V.T)/2
        Q = np.matmul(I-A, np.linalg.inv(I+A))
        return Q

    def L1(V):
        Q = vector2orth(V)
        sp = np.matmul(Q, solutions)
        l1 = np.sum(np.abs(sp))
        return l1

    grad_L1 = grad(L1)


    sols = []
    sol_funcs = []
    # try different seeds
    num_seed = sparse_run
    for i in range(num_seed):
        print("{}/{}".format(i,num_seed))
        seed = i
        np.random.seed(seed)
        V = np.random.randn(n, n)
        sol = minimize(L1, V, method = "L-BFGS-B", jac=grad_L1, tol=1e-10, options={'maxiter':max_iter})
        sol_funcs.append(sol.fun)
        sols.append(sol.x)

    sol_funcs = np.array(sol_funcs)
    winner = np.argmin(sol_funcs)
    sol_func = sols[winner]
    solutions = np.matmul(vector2orth(sol_func), solutions)
    ### out: solutions
    results['sol_cq_sparse'] = copy.deepcopy(solutions)
    
    
    #### Independence ####
    print("#### Selecting independent CQs ####")
    A = np.einsum("ijk,lk->ijl", grad_data, solutions)
    U, S, V = np.linalg.svd(A)
    S = S/np.sum(S, axis=1)[:,np.newaxis]
    ### out: S
    results['s_cq_independent'] = copy.deepcopy(S)

    num_cq_is = np.sum(S>tol_dep, axis=1)
    counts = np.bincount(num_cq_is)
    num_cq_i = np.argmax(counts)
    count = np.sum(num_cq_is == num_cq_i)
    confidence = count/num_points
    print("CQ number={}".format(num_cq_i), ", confidence={}".format(confidence))
    results['number_cq'] = num_cq_i
    results['confidence'] = confidence

    # compute entropy as complexity
    sol2 = solutions**2
    entropy = np.sum(-np.log(sol2+1e-6)*sol2, axis=1)
    sort_id = np.argsort(entropy)
    A = A[:,:,sort_id]
    # rank according to complexity
    solutions = solutions[sort_id]

    sols = []
    ids = []
    num = 0
    for i in range(num_cq):
        if num == num_cq:
            break
        if i == 0:
            sols.append(solutions[i])
            ids.append(i)
            num += 1
        else:
            ids.append(i)
            A_ = A[:,:,ids]
            U, S, V = np.linalg.svd(A_)
            S = S/np.sum(S, axis=1)[:,np.newaxis]
            num_cq_ = np.round(np.mean(np.sum(S>tol_dep, axis=1))).astype(int)
            if num_cq_ == num + 1:
                sols.append(solutions[i])
                num += 1
            else:
                ids.pop()

    ### out:solutions
    results['sol_cq_independent'] = copy.deepcopy(sols)
    
    return results
    