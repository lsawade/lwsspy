from subprocess import Popen, PIPE


def run_cmds_parallel(cmd_list, cwdlist=None):
    """Takes in a list of shell commands:

    Parameters
    ----------
    cmd_list : list
        List of list of arguments

    Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
    """

    # Create list of processes that immediately start execution
    if cwdlist is None:
        cwdlist = len(cmd_list) * None
    process_list = [Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
                    for cmd, cwd in zip(cmd_list, cwdlist)]

    # Wait for the processes to finish
    for proc in process_list:
        proc.wait()

    # Print RETURNCODE, STDOUT and STDERR
    for proc in process_list:
        out, err = proc.communicate()
        if proc.returncode != 0:
            print(proc.returncode)
        if (out != b''):
            print(out.decode())
        if (err != b''):
            print(err.decode())
        if proc.returncode != 0:
            sys.exit()
