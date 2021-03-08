One-liners to Remember
++++++++++++++++++++++

Just one-liners that I decided to document for posterity.

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

<https://teams.microsoft.com/l/message/19:b8c61f8b029b4aa8859776a073f2b12a@thread.tacv2/1615141061566?tenantId=2ff60116-7431-425d-b5af-077d7791bda4&amp;groupId=261db3c5-ac46-4990-a8f4-f4cd774e358f&amp;parentMessageId=1615141061566&amp;teamName=GuyotPhysics&amp;channelName=productivity&amp;createdTime=1615141061566>