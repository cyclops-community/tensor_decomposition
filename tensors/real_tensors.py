import numpy as np
import sys
import os
from os.path import dirname, join
import wget
from zipfile import ZipFile
from scipy.io import loadmat

def amino_acids(tenpy):
    """
    Data: 
        This data set consists of five simple laboratory-made samples. 
        Each sample contains different amounts of tyrosine, tryptophan and phenylalanine 
        dissolved in phosphate buffered water. 
        The samples were measured by fluorescence 
        (excitation 250-300 nm, emission 250-450 nm, 1 nm intervals) 
        on a PE LS50B spectrofluorometer with excitation slit-width of 2.5 nm, 
        an emission slit-width of 10 nm and a scan-speed of 1500 nm/s. 
        The array to be decomposed is hence 5 x 51 x 201. 
        Ideally these data should be describable with three PARAFAC components. 
        This is so because each individual amino acid gives a rank-one contribution to the data.
    References: 
        http://www.models.life.ku.dk/Amino_Acid_fluo
        Bro, R, PARAFAC: Tutorial and applications, 
        Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171

    """
    url = 'http://models.life.ku.dk/sites/default/files/Amino_Acid_fluo.zip'
    file_name = 'Amino_Acid_fluo.zip'
    tensor_name = 'amino.mat'

    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'data')
    try:
        os.stat(data_dir)
    except:
        os.mkdir(data_dir)  

    file_dir = join(data_dir, file_name)
    if not os.path.isfile(file_dir):
        wget.download(url, data_dir)
        zf = ZipFile(file_dir, 'r')
        zf.extractall(data_dir)
        zf.close()

    tensor = loadmat(join(data_dir, tensor_name))['X'].reshape((5,61,201))

    if tenpy.name() == 'ctf':
        return tenpy.from_nparray(tensor)
    return tensor

