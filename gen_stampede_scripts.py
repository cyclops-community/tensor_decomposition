import os


for nodes in [1, 2, 4, 8, 16]:

    cores = nodes * 64
    #no = int(50*nodes**(1./4.))
    #nv = int(200*nodes**(1./4.))
    #nnz_frac = .005/(nv/200.)**2
    #zero_frac = 1.-nnz_frac

    script_file  = open("script_%s.sh" % nodes, "w")

    script_file.write("#!/bin/bash\n")
    script_file.write("#----------------------------------------------------\n")
    script_file.write("#SBATCH -J ctf_cp_lowr\n")
    script_file.write("#SBATCH -o cp_lowr.bench.ppn64.N%s.o%%j.out\n" % nodes)
    script_file.write("#SBATCH -e cp_lowr.bench.ppn64.N%s.o%%j.err\n" % nodes)
    script_file.write("#SBATCH -p normal\n")
    script_file.write("#SBATCH -N %s\n" % nodes)
    script_file.write("#SBATCH -n %s\n" % cores)
    script_file.write("#SBATCH -t 01:30:00\n")
    script_file.write("#SBATCH --mail-user=solomon2@illinois.edu\n")
    script_file.write("#SBATCH --mail-type=all    # Send email at begin and end of job\n")
    script_file.write("\n")
    script_file.write("module list\n")
    script_file.write("pwd\n")
    script_file.write("date\n")
    script_file.write("\n")
    script_file.write("source /home1/05572/esolomon/venv_py27/bin/activate\n")
    script_file.write("\n")
    script_file.write("LD_LIBRARY_PATH=\"/opt/apps/intel18/python2/2.7.15/lib:/opt/apps/libfabric/1.7.0/lib:/opt/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/lib:/opt/intel/debugger_2018/libipt/intel64/lib:/opt/intel/debugger_2018/iga/lib:/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64/gcc4.7:/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.2.199/linux/ipp/lib/intel64:/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/lib/intel64:/opt/apps/gcc/6.3.0/lib64:/opt/apps/gcc/6.3.0/lib:/opt/apps/xsede/gsi-openssh-7.5p1b/lib64:/opt/apps/xsede/gsi-openssh-7.5p1b/lib:::/home1/05572/esolomon/work/ctf/lib_shared:/home1/05572/esolomon/work/ctf/lib_python:/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64:/home1/05572/esolomon/work/ctf/hptt/lib\" PYTHONPATH=\"/opt/apps/intel18/impi18_0/python2/2.7.15/lib/python2.7/site-packages:/home1/05572/esolomon/work/ctf/lib_python:/home1/05572/esolomon/work/pyscf/pyscf/lib/build/\"\n")
    script_file.write("\n")
    script_file.write("\n")
    script_file.write("echo $PYTHONPATH\n")
    script_file.write("\n")
    script_file.write("export CTF_PPN=64\n")
    script_file.write("export OMP_NUM_THREADS=1\n")
    script_file.write("\n")
    script_file.write("\n")
    script_file.write("for s in 1000 2000 4000\n")
    script_file.write("do\n")
    script_file.write("  for R in 1000 2000 4000\n")
    script_file.write("  do\n")
    script_file.write("    for sp_frac in .0001 .001 .01\n")
    script_file.write("    do\n")
    script_file.write("      ibrun python ./als_lowr_msdt_order3.py $s $R 128 6 3 $sp_frac 1 1 0 1 0\n")
    script_file.write("    done\n")
    script_file.write("  done\n")
    script_file.write("done\n")
    script_file.write("ibrun python ./als_lowr_msdt_order3.py 256 2000 200 6 3 .5 1 1 0 1 1\n")
    script_file.write("ibrun python ./als_lowr_msdt_order3.py 1024 15000 200 6 3 .5 1 1 0 1 1\n")
    
