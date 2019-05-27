import numpy as np
import os
from os.path import dirname, join
from scipy.io import loadmat
from .utils import download_unzip_data, load_images_from_folder

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
    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'data')

    urls = ['http://models.life.ku.dk/sites/default/files/Amino_Acid_fluo.zip']
    zip_names = ['Amino_Acid_fluo.zip']
    tensor_name = 'amino.mat'

    download_unzip_data(urls, zip_names, data_dir)
    tensor = loadmat(join(data_dir, tensor_name))['X'].reshape((5,61,201))

    if tenpy.name() == 'ctf':
        return tenpy.from_nparray(tensor)
    return tensor

def coil_100(tenpy):
    """
    Columbia University Image Library (COIL-100)
    References:
        http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
        Columbia Object Image Library (COIL-100), S. A. Nene, S. K. Nayar and H. Murase, 
        Technical Report CUCS-006-96, February 1996.

    """
    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'data')

    def create_bin():
        urls = ['http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip']
        zip_names = ['coil-100.zip']
        file_name = 'coil-100/'
        download_unzip_data(urls, zip_names, data_dir)

        coil_folder = join(data_dir, file_name)
        nonimage_names = ['convertGroupppm2png.pl','convertGroupppm2png.pl~']
        for file in nonimage_names:
            nonimage_path = join(coil_folder,file)
            if os.path.isfile(nonimage_path):
                os.remove(nonimage_path)

        pixel = load_images_from_folder(coil_folder)
        pixel_out = np.reshape(pixel,(7200,128,128,3)).astype(float)

        output_file = open(join(data_dir, 'coil-100.bin'), 'wb')
        print("Print out pixels ......")
        pixel_out.tofile(output_file)
        output_file.close()

    if not os.path.isfile(join(data_dir, 'coil-100.bin')):
        create_bin()
    pixels = np.fromfile(join(data_dir, 'coil-100.bin'), dtype=float).reshape((7200,128,128,3))

    if tenpy.name() == 'ctf':
        return tenpy.from_nparray(pixels)
    return pixels


