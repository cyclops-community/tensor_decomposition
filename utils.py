import os


def save_decomposition_results(T, A, tenpy, folderpath):
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
    tenpy.save_tensor_to_file(T, folderpath + '/tensor')
    for i in range(T.ndim):
        tenpy.save_tensor_to_file(A[i], folderpath + '/mat' + str(i))
