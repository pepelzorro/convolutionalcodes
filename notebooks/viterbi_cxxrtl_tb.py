import subprocess
from threading import Thread
import numpy as np


def viterbi_cxxrtl_tb(coded_sequence, cxxrtl_tb_filename):
    """Python interface to cxxrtl executable

    coded_sequence
        Input data

    cxxrtl_tb_filename
        Executable name
    """

    coded_string = "".join([chr(c + 97) for c in coded_sequence])
    cxxrtl_bytes = bytearray(coded_string + ".", "ascii")
    decoded = []

    def decode_output(out, decoded):
        for line in iter(out.readline, b""):
            result = [1 if x == 49 else -1 for x in line if x == 48 or x == 49]
            decoded.extend(result)
        out.close()

    # Call out to cxxrtl process
    cxxrtl_exec = subprocess.Popen(
        cxxrtl_tb_filename, stdout=subprocess.PIPE, stdin=subprocess.PIPE
    )

    # Thread to decode the output
    t = Thread(target=decode_output, args=(cxxrtl_exec.stdout, decoded))
    t.daemon = True  # thread dies with the program
    t.start()

    # Chunked input
    size = 2 ** 14
    for pos in range(0, len(cxxrtl_bytes), size):
        cxxrtl_exec.stdin.write(cxxrtl_bytes[pos : pos + size])
        cxxrtl_exec.stdin.write(b"\n")

    # Close stdin
    try:
        cxxrtl_exec.stdin.close()
    except:
        pass

    # Wait for stdout to close
    while not cxxrtl_exec.stdout.closed:
        pass

    cxxrtl_exec.wait()

    return np.array(decoded)
