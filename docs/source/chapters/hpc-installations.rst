High-Performance-Computing Installations
----------------------------------------

On thist page, I'm documenting the different ways of installing packages for
clusters. Specifically the ones that need ``MPI`` support, such as
`h5py <https://docs.h5py.org/en/stable/>`_, 
`tables <https://www.pytables.org/index.html>`_, etc.

PyTables
++++++++

`PyTables <https://www.pytables.org/index.html>`_ is a great tool to
interact (write and read) with Pandas DataFrames. However, it can be finicky to 
install with your already parallel ``HDF5`` and ``h5py`` packages installed.

You will have to set two, three environment variables to make the installation
possible.

.. code:: bash

    # Set hdf5 library and mpi compiler paths
    export HDF5_DIR=/path/to/your/parallel_hdf5_installation
    export CC=$(which mpicc)

    pip install tables

For most clusters, you can make an mpi compiler available through
``module load <your_favorite_mpi_compiler>`` 
(I generally use ``openmpi/gcc``, which is not ideal, 
because cluster specific compilers usually give you better support/speed).
