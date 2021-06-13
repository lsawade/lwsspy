Limit the number of concurrent jobs in the terminal
---------------------------------------------------

This is a function I found on 
`StackOverflow <https://stackoverflow.com/questions/1537956/bash-limit-the-number-of-concurrent-jobs>`_,
 that seems to do the trick. It is ``user3769065``'s answer bulding on 
 ``tangens``'s answer. It defines a function that contains a ``while`` loop that
 will loop until less than a given concurrent jobs are executed. 
 So, if it is executed after every single background task is placed it will 
 simply start every job until a given limit is reached s


.. code:: bash

    job_limit () {
        # Test for single positive integer input
        if (( $# == 1 )) && [[ $1 =~ ^[1-9][0-9]*$ ]]
        then

            # Check number of running jobs
            joblist=($(jobs -rp))

            # Loop to heck whether jobs are still running
            while (( ${#joblist[*]} >= $1 ))
            do

                # Wait for any job to finish
                command='wait '${joblist[0]}
                for job in ${joblist[@]:1}
                do
                    command+=' || wait '$job
                done
                eval $command
                joblist=($(jobs -rp))

            done
        fi
    }


and the function is used as follows:

.. code:: bash

    while :
    do
        <some_task> &
        job_limit <nproc>
    done