# cmdutil
A python utility for (sanely an non-annoyingly) running shell commands within a python script. Simply pass a shell command as an argument. It will return the stdout as a string and will raise a ValueError if the process returns a non-zero status.

Use:

1. Clone the repo into the directory of your python script: $`cd [project directory]; git clone https://github.com/david-thrower/cmdutil.git`
2. In your script import it:
3. Pass a command as the first argument to run_command:

```python3

from cmdutil.cmdutil import run_command

cmd = "tar -czvf archive.tar.gz archive"
console_output = run_command(cmd)

```
Args:

cmd: str, a shell command to be executed
verbose: bool, default True, Whether to print annotations of the command being run and the outcome of the command.

Returns:

str: Standard output from command in UTF8 encoding.

Raises: ValueError, listing standard error and exit status if the process returned a non-zero status.

