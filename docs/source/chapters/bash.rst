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


.. note::

    It should be noted that we can combine both the job_limit and the status 
    check. See below


Save the exit status of background processes
--------------------------------------------

.. code:: bash

    # Some function that takes a long time to process
    longprocess() {
            # Sleep up to 14 seconds
            sleep $((RANDOM % 15))
            # Randomly exit with 0 or 1
            exit $((RANDOM % 2))
    }

    pids=""
    # Run five concurrent processes
    for i in {1..5}; do
            ( longprocess ) &
            # store PID of process
            pids+=" $!"
    done

    # Wait for all processes to finish, will take max 14s
    # as it waits in order of launch, not order of finishing
    for p in $pids; do
            if wait $p; then
                    echo "Process $p success"
            else
                    echo "Process $p fail"
            fi
    done


.. note::

    It should be noted that we can combine both the job_limit and the status 
    check. See below



Combining a joblimit and a status check
---------------------------------------

    .. code:: bash

        #!/bin/bash



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
        
        
        # Some function that takes a long time to process
        longprocess() {
                # Sleep up to 14 seconds
                sleep $((RANDOM % 15))
                # Randomly exit with 0 or 1
                exit $((RANDOM % 2))
        }
        
        indeces=()
        pids=()
        # Run five concurrent processes
        for i in {1..10}; do
                ( longprocess ) &
            
                # store PID of process
                pids+=("$!")
                indeces+=("$i")
        
                job_limit 6
        done
        
        # Wait for all processes to finish, will take max 14s
        # as it waits in order of launch, not order of finishing
        
        for i in ${!pids[@]}; do
                if wait ${pids[$i]}; then
                        echo "Process ${indeces[$i]}, ${pids[$i]} success"
                else
                        echo "Process ${indeces[$i]}, ${pids[$i]} fail"
                fi
        done
        
