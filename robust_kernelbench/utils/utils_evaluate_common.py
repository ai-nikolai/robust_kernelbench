import sys
import os
import io
import contextlib
import tempfile
import subprocess
import multiprocessing

# Global variables to hold original file descriptors (not ideal, but one way)
_original_stdout_fd = None
_original_stderr_fd = None


class OutputCaptureStandard:
    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        # self.combined = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def get_output(self):
        return self.stdout.getvalue()



class OutputCapture:
    """
    Context manager to capture both Python sys.stdout/stderr
    and output from subprocesses (like nvcc) by redirecting file descriptors.
    """
    def __init__(self):
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.original_stdout_fd = None
        self.original_stderr_fd = None
        self.temp_stdout_fd = None
        self.temp_stderr_fd = None

    def __enter__(self):
        # Save the original file descriptors (1 for stdout, 2 for stderr)
        self.original_stdout_fd = os.dup(1)
        self.original_stderr_fd = os.dup(2)

        # Create temporary files to redirect to
        # We'll read from these later
        temp_stdout = open(os.devnull, 'w+b') # Use a real file if you need to read content reliably
        temp_stderr = open(os.devnull, 'w+b')
        # For capturing content, StringIO works for Python, but for fd redirection,
        # we need actual file objects. Let's use a more robust approach with pipes or temporary files.
        # Actually, StringIO doesn't directly provide file descriptors suitable for os.dup2.
        # We need to use temporary files or pipes.

        # A more robust way using temporary files:
        self.temp_stdout_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        self.temp_stderr_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)

        # Duplicate the file descriptors
        os.dup2(self.temp_stdout_file.fileno(), 1) # Redirect fd 1 (stdout) to temp file
        os.dup2(self.temp_stderr_file.fileno(), 2) # Redirect fd 2 (stderr) to temp file

        # Also redirect Python's sys.stdout/sys.stderr to capture pure Python print statements
        # that might happen *after* fd redirection but *before* nvcc starts
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original file descriptors
        os.dup2(self.original_stdout_fd, 1)
        os.dup2(self.original_stderr_fd, 2)

        # Close the temporary files and the duplicated original ones
        self.temp_stdout_file.close()
        self.temp_stderr_file.close()
        os.close(self.original_stdout_fd)
        os.close(self.original_stderr_fd)

        # Restore Python's sys.stdout/sys.stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Read the content captured by the file redirection
        # Need to reopen the temporary files to read content
        with open(self.temp_stdout_file.name, 'r') as f:
            fd_stdout_content = f.read()
        with open(self.temp_stderr_file.name, 'r') as f:
            fd_stderr_content = f.read()

        # Clean up temporary files
        os.unlink(self.temp_stdout_file.name)
        os.unlink(self.temp_stderr_file.name)

        # Combine the content captured by Python redirection and file descriptor redirection
        # Pure Python print statements go to StringIO
        # Subprocess output (like nvcc) goes to the temp files
        self.combined_stdout = self.stdout_capture.getvalue() + fd_stdout_content
        self.combined_stderr = self.stderr_capture.getvalue() + fd_stderr_content

    def get_output(self):
        """Get combined stdout content."""
        return getattr(self, 'combined_stdout', '')

    def get_error(self):
        """Get combined stderr content."""
        return getattr(self, 'combined_stderr', '')


def compile_and_get_output(source_string, *args):
    """Compiles python code and returns stdout / stderr"""
    output = ""
    e = None
    comp_out = None
    try:
        with OutputCapture() as capture:
            try:
                comp_out = compile(source_string, *args)
                output = capture.get_output()
                # return output, None
            except Exception as e1:
                e = e1
                output = capture.get_output()

        return comp_out, output, e
    except Exception as e:
        print("THERE IS CAPTURE ERROR DURING COMPILE")
        print(e)
        return comp_out, output, e
    
def exec_and_get_output_standard(source_string, context):
    """Executes python code and returns stdout / stderr"""
    output = ""
    e = None
    exec_out = None
    try:
        with OutputCapture() as capture:
            try:
                exec_out =  exec(source_string, context)  # expose to current namespace
                output = capture.get_output()
                # return output, None
            except Exception as e2:
                e = e2
                output = capture.get_output()
        return exec_out, output, e

    except Exception as e1:
        print("THERE IS CAPTURE ERROR DURING EXEC")
        print(e)
        return exec_out, output, e1





@contextlib.contextmanager
def capture_subprocess_output():
    """
    Context manager to capture output from subprocesses by redirecting file descriptors.
    This redirects the actual OS file descriptors (1, 2) so subprocesses inherit them.
    """
    global _original_stdout_fd, _original_stderr_fd
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    temp_stdout_path = None
    temp_stderr_path = None

    try:
        # Save original file descriptors
        _original_stdout_fd = os.dup(1)
        _original_stderr_fd = os.dup(2)

        # Create temporary files to redirect to
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_stdout.log') as temp_stdout, \
             tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_stderr.log') as temp_stderr:

            temp_stdout_path = temp_stdout.name
            temp_stderr_path = temp_stderr.name

            # Redirect file descriptors 1 (stdout) and 2 (stderr) to temp files
            os.dup2(temp_stdout.fileno(), 1)
            os.dup2(temp_stderr.fileno(), 2)

            # Also redirect Python's sys.stdout/stderr to capture pure Python output
            # that might happen concurrently
            sys.stdout = temp_stdout
            sys.stderr = temp_stderr

            yield temp_stdout_path, temp_stderr_path

    finally:
        # Restore original file descriptors
        if _original_stdout_fd is not None:
            os.dup2(_original_stdout_fd, 1)
        if _original_stderr_fd is not None:
            os.dup2(_original_stderr_fd, 2)

        # Restore Python's sys.stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close and delete the temporary files
        if _original_stdout_fd is not None:
            os.close(_original_stdout_fd)
            _original_stdout_fd = None
        if _original_stderr_fd is not None:
            os.close(_original_stderr_fd)
            _original_stderr_fd = None

        # Ensure temp files are closed before deletion
        # The 'with' statement above should handle this, but explicitly check
        if temp_stdout_path and os.path.exists(temp_stdout_path):
            # File should be closed by 'with', now read and delete
            pass
        if temp_stderr_path and os.path.exists(temp_stderr_path):
            # File should be closed by 'with', now read and delete
            pass


def exec_and_get_output(source_string, context):
    """Executes python code and returns stdout / stderr captured via fd redirection."""
    exec_out = None
    captured_stdout = ""
    captured_stderr = ""
    e = None

    # Use the new context manager that redirects file descriptors
    with capture_subprocess_output() as (stdout_path, stderr_path):
        try:
            # Execute the code which may launch subprocesses (like nvcc)
            exec_out = exec(source_string, context)
        except Exception as e2:
            e = e2
        finally:
            # Flush any remaining output to the temporary files
            os.fsync(1) # Flush stdout fd
            os.fsync(2) # Flush stderr fd
            sys.stdout.flush()
            sys.stderr.flush()

    # Read the content from the temporary files after subprocesses finish
    try:
        with open(stdout_path, 'r') as f:
            captured_stdout = f.read()
    except FileNotFoundError:
        captured_stdout = "(stdout file not found)"

    try:
        with open(stderr_path, 'r') as f:
            captured_stderr = f.read()
    except FileNotFoundError:
        captured_stderr = "(stderr file not found)"

    # Clean up temporary files
    try:
        os.unlink(stdout_path)
    except FileNotFoundError:
        pass # Already deleted or never created properly
    try:
        os.unlink(stderr_path)
    except FileNotFoundError:
        pass # Already deleted or never created properly

    return exec_out, captured_stdout, e


def format_exception_string(exception_or_string, truncate=True, max_length=200):
    if type(exception_or_string) == str:
        return exception_or_string
    else:
        exception_str = str(exception_or_string)
        if truncate and len(exception_str) > max_length:
            exception_str = exception_str[: max_length - 3] + "..."
        exception_type = type(exception_or_string)
        return f"{exception_type}({exception_str})"


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
    ):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata, exception_str


# GPU MODE STUFF

# def worker(conn):
#     while True:
#         args = conn.recv()  # wait for data from parent

#         try:
#             result = benchmark(args)
#             error = None
#         except:
#             result = None
#             error = traceback.format_exc()

#         conn.send((result, error))  # send result to parent

# # in parent process
# class Worker:
#     def __init__(self):
#         mp_ctx = mp.get_context("spawn")
#         self.conn, conn = mp_ctx.Pipe()
#         self.worker = mp_ctx.Process(target=self.worker_process, args=(conn,), daemon=True)
#         self.worker.start()

#     def benchmark(self, args):
#         self.conn.send(args)  # send to child worker
#         result, error = self.conn.recv()  # obtain result

#         if error is not None:
#             # get full list of sticky CUDA errors here
#             # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
#             if "cudaErrorIllegalAddress" in error:
#                 self.worker.kill()
#                 # create pipe and subprocess again
#                 ...

#         return result
