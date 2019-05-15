import unittest
from CPD.common_kernels import solve_sys
import tensors.synthetic_tensors as stsrs
from cpals import CP_ALS

class CPDALSTestCase(unittest.TestCase):
    """Tests for CPD_ALS."""

    def test_random_numpy_dense(self):
        import numpy_ext as tenpy

        [T,O] = stsrs.init_rand(tenpy,order=3,s=10,R=5,sp_frac=1.)
        A = []
        for i in range(T.ndim):
            A.append(tenpy.random((T.shape[i],5)))

        res = CP_ALS(tenpy,A,T,O,num_iter=30,sp_res=0,csv_writer=None,Regu=1e-6)
        self.assertTrue(res<=0.5)

    def test_random_numpy_sparse(self):
        import numpy_ext as tenpy
 
        [T,O] = stsrs.init_rand(tenpy,order=3,s=10,R=5,sp_frac=0.99)
        A = []
        for i in range(T.ndim):
            A.append(tenpy.random((T.shape[i],5)))

        res = CP_ALS(tenpy,A,T,O,num_iter=30,sp_res=0,csv_writer=None,Regu=1e-6)
        self.assertTrue(res<=0.5)

    def test_random_ctf_dense(self):
        import ctf_ext as tenpy

        [T,O] = stsrs.init_rand(tenpy,order=3,s=10,R=5,sp_frac=1.)
        A = []
        for i in range(T.ndim):
            A.append(tenpy.random((T.shape[i],5)))

        res = CP_ALS(tenpy,A,T,O,num_iter=30,sp_res=0,csv_writer=None,Regu=1e-6)
        self.assertTrue(res<=0.5)

    def test_random_ctf_sparse(self):
        import ctf_ext as tenpy
 
        [T,O] = stsrs.init_rand(tenpy,order=3,s=10,R=5,sp_frac=0.99)
        A = []
        for i in range(T.ndim):
            A.append(tenpy.random((T.shape[i],5)))

        res = CP_ALS(tenpy,A,T,O,num_iter=30,sp_res=0,csv_writer=None,Regu=1e-6)
        self.assertTrue(res<=0.5)

if __name__ == '__main__':
    unittest.main()

