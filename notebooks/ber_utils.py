from multiprocessing import Pool
from scipy.signal import max_len_seq
import numpy as np


def awgn_channel(signal, eb_n0_dB=0):
    """
    Assume signal has a power of 1
    """

    n_dB = -eb_n0_dB
    n = 10 ** (n_dB / 10)

    noise = np.random.normal(0, np.sqrt(n), len(signal))

    return signal + noise


def sequence(length=1e6):
    """Deterministic Maximum Length Sequence

    Parameters
    ----------
    length: int
    """
    assert length < 2 ** 24
    length = int(length)

    sequence, state = max_len_seq(24, length=length)
    return np.array(sequence).astype(np.int8) * 2 - 1  # +1 and -1


def decision(s, q=1):
    """
    Quantise signal `s` into 2**q levels

    Input samples in the range [-1, 1]. Outputs in the range [0, (2**q) - 1]

    If q=1 (the default), then this is a hard decision function

    Parameters
    ----------
    s: numpy array
    """
    assert q > 0
    maximum = 2 ** (q - 1)

    s = np.array(s)

    return ((np.clip(s, -1, 1) + 0.999999999) * maximum).astype(np.uint8)


def map_symbols(s, *args, **kwargs):
    """Map quantised symbols onto the interval [-1, 1] from the interval [0, 1]

    Parameters
    s: numpy array
    """
    return np.clip((s - 0.5) * 2, -1, 1).astype(np.int8)


def bit_error_rate(s1: np.ndarray, s2: np.ndarray):
    """
    Calculate the Bit Error Rate (BER) by comparing two signals

    Parameters
    ----------
    s1, s2: numpy array
    """
    s2 = np.array(s2[: len(s1)])

    assert s1.shape == s2.shape

    matches = np.sum(s1 == s2)

    if matches < len(s1):
        return 1.0 - (matches / len(s1))
    else:
        return np.NaN


# Multiprocessing BER sweep
def ber_calculation(arg):
    coded = arg[0]
    signal = arg[1]
    eb_n0 = arg[2]
    q = arg[3]
    decode_fn = arg[4]

    channel = awgn_channel(coded, eb_n0_dB=eb_n0)
    bits = decision(channel, q=q)

    decoded = decode_fn(bits)
    error_rate = bit_error_rate(signal, decoded)

    print(".", end="")
    return (eb_n0, error_rate)


def ber_curve(
    decode_fn=None, encode_fn=None, q=1, length=1e6, start_dB=0, end_dB=10, step=1
):
    """Calculate Bit Error Rate (BER) curve in a AWGN channel

    Parameters
    ----------
    decode_fn: fn(bits)
        Function that takes quantised bits and returns the decoded bitstream.
        If None, defaults to uncoded for q=1

    encode_fn: fn(signal)
        Function that returns the encoded bits for transmission over the channel
        If None, bits are not encoded

    q : int
        Quantisation order for channel samples. Defaults to 1 (hard decision)
    """
    eb_n0_x = []
    ber = []

    if decode_fn == None:
        decode_fn = map_symbols

    # Encode Maximum Length Sequence
    signal = sequence(length)
    if encode_fn:
        coded = encode_fn(signal)
    else:
        coded = signal  # Uncoded
    print(".", end="")

    # Divide the calculations among a multiprocessing pool
    with Pool(processes=4) as pool:
        args = [
            [coded, signal, eb_n0, q, decode_fn]
            for eb_n0 in range(start_dB, end_dB + step, step)
        ]

        result = pool.map(ber_calculation, args)
        (eb_n0_x, ber) = zip(*result)

    return np.array(eb_n0_x), np.array(ber)
