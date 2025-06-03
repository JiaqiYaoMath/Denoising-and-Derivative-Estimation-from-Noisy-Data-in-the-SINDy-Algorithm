# -*- coding: utf-8 -*-


import numpy as np
from scipy.linalg import solve

from typing import Tuple

class TVR:

    def __init__(self, n: int, dx: float, target: str):
        """Differentiate with TVR.

        Args:
            n (int): Number of points in data.
            dx (float): Spacing of data.
            target (str): recover function or derivative
        """
        self.n = n
        self.dx = dx
        self.target = target
        
        if target=='derivative':
            self.a_mat = self.A()
            self.a_mat_t = np.transpose(self.a_mat)
        else:
            self.a_mat = self.I()
            self.a_mat_t = self.I()
            
        self.d_mat = self.D()

        
    ###########################################################
    def D(self) -> np.array:
        """Make differentiation matrix with central differences. NOTE: not efficient!

        Returns:
            np.array: n-1 x n
        """
        arr = np.zeros((self.n-1,self.n))
        for i in range(0,self.n-1):
            arr[i,i] = -1.0
            arr[i,i+1] = 1.0
        return arr / self.dx
    
    ###########################################################
    # TODO: improve these matrix constructors
    def A(self) -> np.array:
        """Make integration matrix with trapezoidal rule. NOTE: not efficient!

        Returns:
            np.array: n-1 x n
        """
        arr = np.zeros((self.n,self.n))
        for i in range(0,self.n):
            if i==0:
                continue
            for j in range(0,self.n):
                if j==0:
                    arr[i,j] = 0.5
                elif j<i:
                    arr[i,j] = 1.0
                elif i==j:
                    arr[i,j] = 0.5
        
        return arr * self.dx
    
    ###########################################################
    def I(self) -> np.array:
        """Identical matrix I

        Returns:
            np.array: n * n
        """
        
        return np.eye(self.n)
    
    ###########################################################
    def make_en_mat(self, solution_curr : np.array) -> np.array:
        """Diffusion matrix

        Args:
            solution_curr (np.array): Current solutin of length n

        Returns:
            np.array: n-1 x n-1 
        """
        eps = pow(10,-8)
        vec = 1.0/np.sqrt(pow(self.d_mat @ solution_curr,2) + pow(eps,2))
        return np.diag(vec)
    
    ###########################################################
    def make_ln_mat(self, en_mat : np.array) -> np.array:
        """Diffusivity term

        Args:
            en_mat (np.array): Result from make_en_mat

        Returns:
            np.array: n x n
        """
        return self.dx * np.transpose(self.d_mat) @ en_mat @ self.d_mat
    
    ###########################################################
    def make_gn_vec(self, solution_curr : np.array, data : np.array, alpha : float, ln_mat : np.array) -> np.array:
        """Negative right hand side of linear problem

        Args:
            solution_curr (np.array): Current solution of size n
            data (np.array): Data of size n
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat

        Returns:
            np.array: Vector of length n
        """
        return self.a_mat_t @ self.a_mat @ solution_curr - self.a_mat_t @ data + alpha * ln_mat @ solution_curr
    
    ###########################################################
    def make_hn_mat(self, alpha : float, ln_mat : np.array) -> np.array:
        """Matrix in linear problem

        Args:
            alpha (float): Regularization parameter
            ln_mat (np.array): Diffusivity term from make_ln_mat

        Returns:
            np.array: n x n
        """
        return self.a_mat_t @ self.a_mat + alpha * ln_mat
    
    ###########################################################
    def get_solution_update(self, data : np.array, solution_curr : np.array, alpha : float) -> np.array:
        """Get the TVR update

        Args:
            data (np.array): Data of size N
            solution_curr (np.array): Current solution of size n
            alpha (float): Regularization parameter

        Returns:
            np.array: Update vector of size n
        """

        n = len(data)
    
        en_mat = self.make_en_mat(
            solution_curr=solution_curr
            )

        ln_mat = self.make_ln_mat(
            en_mat=en_mat
            )

        hn_mat = self.make_hn_mat(
            alpha=alpha,
            ln_mat=ln_mat
            )

        gn_vec = self.make_gn_vec(
            solution_curr=solution_curr,
            data=data,
            alpha=alpha,
            ln_mat=ln_mat
            )

        return solve(hn_mat, -gn_vec)

    def get_solution(self, 
        data : np.array, 
        solution_guess : np.array, 
        alpha : float,
        no_opt_steps : int,
        visual : bool = False,
        return_progress : bool = False, 
        return_interval : int = 1
        ) -> Tuple[np.array,np.array]:
        """Get solution via TVR over optimization steps

        Args:
            data (np.array): Data of size N
            solution_guess (np.array): Guess for solution of size N+1
            alpha (float): Regularization parameter
            no_opt_steps (int): No. opt steps to run
            return_progress (bool, optional): True to return solution progress during optimization. Defaults to False.
            return_interval (int, optional): Interval at which to store solution if returning. Defaults to 1.

        Returns:
            Tuple[np.array,np.array]: First is the final solution of size n, second is the stored solutions if return_progress=True of size no_opt_steps+1 x n, else [].
        """

        solution_curr = solution_guess
        
        if self.target == 'derivative':
            data = data - data[0]
        
        if return_progress:
            solution_st = np.full((no_opt_steps+1, len(solution_guess)), 0)
        else:
            solution_st = np.array([])

        for opt_step in range(0,no_opt_steps):
            update = self.get_solution_update(
                data=data,
                solution_curr=solution_curr,
                alpha=alpha
                )

            solution_curr += update

            if return_progress:
                if opt_step % return_interval == 0:
                    solution_st[int(opt_step / return_interval)] = solution_curr
            
            if visual:
                print(opt_step)

        return (solution_curr, solution_st)
    
    
    
# import numpy as np
# import matplotlib.pyplot as plt



# if __name__ == "__main__":

#     # Data
#     dx = 0.01
    
#     data = []
#     for x in np.arange(0,2,dx):
#         data.append(abs(x-1))
#     data = np.array(data)

#     # True derivative
#     deriv_true = []
#     for x in np.arange(0,2,dx):
#         if x < 1:
#             deriv_true.append(-1)
#         else:
#             deriv_true.append(1)
#     deriv_true = np.array(deriv_true)

#     # Add noise
#     n = len(data)
#     data_noisy = data + np.random.normal(0,0.05,n)
    
#     # Plot true and noisy signal
#     fig1 = plt.figure()
#     plt.plot(data)
#     plt.plot(data_noisy)
#     plt.title("Signal")
#     plt.legend(["True","Noisy"])

#     # Derivative with TVR
#     diff_tvr = TVR(n,dx,target = 'derivative')
#     (deriv,_) = diff_tvr.get_solution(
#         data=data_noisy, 
#         solution_guess=np.full(n,0.0), 
#         alpha=1,
#         no_opt_steps=100
#         )

#     # Plot TVR derivative
#     fig2 = plt.figure()
#     plt.plot(deriv_true)
#     plt.plot(deriv)
#     plt.title("Derivative")
#     plt.legend(["True","TVR"])

#     fig1.savefig('signal.png')
#     fig2.savefig('derivative.png')

#     plt.show()