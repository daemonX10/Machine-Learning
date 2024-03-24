# The line `import os , sys` is importing the `os` and `sys` modules in Python.
# The `os` module in Python provides a way to use operating system dependent functionality. It
# allows you to interact with the underlying operating system, such as accessing files and
# directories, manipulating paths, and executing system commands.
# The `sys` module in Python provides access to some variables used or maintained by the
# interpreter and to functions that interact with the interpreter. It allows you to
# manipulate the Python runtime environment, such as accessing command-line arguments,
# interacting with the standard input/output streams, and getting information about the
# Python version and the current platform.

import os , sys
from os.path import dirname , join ,abspath

sys.path.insert(0 , abspath(join(dirname(__file__),'..')))

from course_details import course_details

def payment():
    print ("This is my payment file")

course_details.course()


