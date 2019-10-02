import unittest
from CPD.common_kernels import solve_sys
import tensors.synthetic_tensors as stsrs
from cpals import CP_ALS
import ctf_ext
import numpy_ext


class CPDALSTestCase(unittest.TestCase):
    """Tests for CPD_ALS."""
    def test_random_tensor_dense(self):
        for tenpy in [numpy_ext, ctf_ext]:
            [T, O] = stsrs.init_rand(tenpy, order=3, s=10, R=5, sp_frac=1.)
            A = []
            for i in range(T.ndim):
                A.append(tenpy.random((T.shape[i], 5)))

            res = CP_ALS(tenpy,
                         A,
                         T,
                         O,
                         num_iter=30,
                         sp_res=0,
                         csv_writer=None,
                         Regu=1.e-6)
            self.assertTrue(res <= 0.5)

    def test_random_tensor_sparse(self):
        for tenpy in [numpy_ext, ctf_ext]:
            [T, O] = stsrs.init_rand(tenpy, order=3, s=10, R=5, sp_frac=0.9999)
            A = []
            for i in range(T.ndim):
                A.append(tenpy.random((T.shape[i], 5)))

            res = CP_ALS(tenpy,
                         A,
                         T,
                         O,
                         num_iter=30,
                         sp_res=0,
                         csv_writer=None,
                         Regu=1.e-6)
            self.assertTrue(res <= 0.5)


if __name__ == '__main__':
    unittest.main()
