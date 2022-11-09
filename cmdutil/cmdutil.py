from subprocess import run

def run_command(cmd,
                verbose=True):
    proc_obj = run(cmd,
                   shell=True,
                   capture_output = True)
    return_status = proc_obj.returncode
    std_out = proc_obj.stdout.decode("utf8")
    std_err = proc_obj.stderr.decode("utf8")
    if return_status != 0:
        raise ValueError(f"Process: '{cmd}' returned a non-zero status, "
                         f"specifically; '{return_status}'; Details: '{std_err}'")
    if verbose == True:
        print(f"Successfully executed command: '{cmd}'.:\n {std_out}")
    return std_out
