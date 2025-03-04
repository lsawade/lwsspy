One-liners to Remember
++++++++++++++++++++++

Just one-liners that I decided to document for posterity.


Simple pattern syncing that retains directory structure
-------------------------------------------------------


This is going to be helpful if
1. All your files are located in a certain directory structure
2. you know exactly the file pattern that 
but need to get only this files from directories tress that you need to 
retain. 

.. code:: bash
    
rsync -av --prune-empty-dirs \
    --include="*/" \
    --include="<yourpattern>" \
    --exclude="*" \
    source_dir/ \
    destination_dir

Note that ``source_dir`` and/or ``destination_dir`` can be remote locations,
but it is important that source_dir is followed by a ``/``, otherwise 
``source_dir``'s content will be saved in ``destination_dir``.
An important flag is ``--prune-empty-dirs`` which will remove directories that
do not contain an files. Use the ``--dry-ru`` option to just see the files that
are goin to be copied.


Simple Function wrapper in Bash
-------------------------------

I had the issue that a cluster (very large, very national) didn't like 
launching jobs using mpirun or mpiexec (Noted in there User manual), 
but for debugging with an interactive allocation, that is nice. 
The alternative was writing ``jsrun -n <nproces> -a 1 -c 1 <yourscript>`` 
to avoid that I found a neat one-liner for bash that you can put in your 
``.bashrc``

.. code:: bash
    
    mpifunc() {​​​​​​​​ jsrun -n $1 -a 1 -c 1 ${​​​​​​​​@:2}​​​​​​​​ ; }​​​​​​​​


What I found nice about this is the simplicity of accessing the command line 
variables, where ``${​​​​​​​​@:2}​​​​​​​​`` means all arguments including and after argument 2. 
and ``$@`` is just all arguments.

Another example I found on stackoverflow:

1. open new file and edit it: r.sh:

.. code:: bash
    
    echo "params only 2    : ${​​​​​​​​@:2:1}​​​​​​​​"
    echo "params 2 and 3   : ${​​​​​​​​@:2:2}​​​​​​​​"
    echo "params all from 2: ${​​​​​​​​@:2:99}​​​​​​​​"
    echo "params all from 2: ${​​​​​​​​@:2}​​​​​​​​"


2. Make your script executable and execute it

.. code:: bash
    
    chmod u+x r.sh
    ./r.sh 1 2 3 4 5 6 7 8 9 10



3.  Checkout output:

.. code:: bash
    
    params only 2    : 2
    params 2 and 3   : 2 3
    params all from 2: 2 3 4 5 6 7 8 9 10
    params all from 2: 2 3 4 5 6 7 8 9 10



While this is super simple, I assume I haven't used bash enough to have had a need for using the${​​​​​​​​​@:2}​​​​​​​​ syntax! Here the stakcoverflow link: https://stackoverflow.com/questions/1537673/how-do-i-forward-parameters-to-other-command-in-bash-script


Here, more on accessing parameters etc. in BASH from IBM https://developer.ibm.com/tutorials/l-bash-parameters/#N10171


Create MP4 or GIF from images using ``ffmpeg``
----------------------------------------------

The one-liner

.. code:: bash

    ffmpeg -framerate 60 -r 30 -pattern_type glob -i '*.png' -vf scale=1920:-1 -pix_fmt yuv420p -vcodec libx264 hello.mp4

Decode the one-liner 

.. code:: bash

    ffmpeg \
        -framerate 60 \       # Define Framerate for Movie 
        -r 30 \               # Define Adjust frames to be displayed during s
        -pattern_type glob \  # Use glob pattern to get files
        -i '*.png' \          # Provide patter
        -vf scale=1920:-1 \   # rescale width of the video
        -pix_fmt yuv420p \    # Make it available for quicktime to play
        -vcodec libx264 \     #  --- == ---
        hello.mp4             # Provide file name



