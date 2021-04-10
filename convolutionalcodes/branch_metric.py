from nmigen import *
from nmigen.sim import Simulator, Active

from .util.test import *


class BMU(Elaboratable):
    """Branch Metric Unit (BMU) for a R = 1/2 Viterbi Decoder

    Generates low/high branch metrics using input codewords x0, x1

    Parameters
    ----------
    s_x0 : Signal(width), in
        Soft decision for x0

        Value in the range 0..(2^width - 1) inclusive, where 0 represents a
        logical 0 and (2^width - 1) represents a logical 1

    s_x1 : Signal(width), in
        Soft decision for x1

    bm_low : Signal(width + 1), out
        Branch Metric Output for the low branch

    bm_high : Signal(width + 1), out
        Branch Metric Output for the high branch

    n : int
        nth BMU

    x_max : int, optional
        Value of the largest input symbol value. If None, defaults to (2 **
        width) - 1

    k : int, optional
        Constraint length. Defaults to k=3

    g1 : int, optional
        Generator Polynomial 1. There are two generator polynomials for a R =
        1/2 code.

        Defaults to 0b111, which is a common g1 polynomial for a k=3 code

    g2 : int, optional
        Generator Polynomial 2. There are two generator polynomials for a R =
        1/2 code

        Defaults to 0b101, which is a common g2 polynomial for a k=3 code

    """

    def __init__(
        self,
        s_x0,
        s_x1,
        bm_low,
        bm_high,
        n,
        x_max=None,
        k=3,
        g1=0b111,
        g2=0b101,
    ):
        self.s_x0 = s_x0
        self.s_x1 = s_x1
        self.bm_low = bm_low
        self.bm_high = bm_high
        assert self.s_x0.width == self.s_x1.width, "x0, x1 must have the same width"
        assert self.bm_low.width == self.s_x0.width + 1

        self._width = s_x0.width
        if x_max == None:
            x_max = (2 ** s_x0.width) - 1
        self._max = x_max
        assert type(x_max) == int, "x_max must be an integer"

        self._k = k
        self._n = Const(n, k)
        self._g1 = Const(g1, k)
        self._g2 = Const(g2, k)

    def elaborate(self, platform):
        m = Module()

        s_x0 = self.s_x0
        s_x1 = self.s_x1
        bm_low = self.bm_low
        bm_high = self.bm_high

        k = self._k
        n = self._n
        g1 = self._g1
        g2 = self._g2

        x_low0 = Signal(self._width)
        x_high0 = Signal(self._width)
        x_low1 = Signal(self._width)
        x_high1 = Signal(self._width)

        # Branch constants
        c0 = 0
        c1 = 0
        for j in range(k - 1):
            c0 ^= n[j] & g1[j]
            c1 ^= n[j] & g2[j]

        # Distances for low branch
        m.d.comb += x_low0.eq(Mux(c0, self._max - s_x0, s_x0))
        m.d.comb += x_low1.eq(Mux(c1, self._max - s_x1, s_x1))

        # Distances for high branch
        m.d.comb += x_high0.eq(Mux(c0, s_x0, self._max - s_x0))
        m.d.comb += x_high1.eq(Mux(c1, s_x1, self._max - s_x1))

        # Use the Manhattan Distance as the branch metric.
        #
        # An optimum decoder can be constructed by minimising the Euclidean
        # distance, but the Manhattan distance is also acceptable as Viterbi
        # decoding is a linear process
        m.d.comb += bm_low.eq(x_low1 + x_low0)
        m.d.comb += bm_high.eq(x_high1 + x_high0)

        return m


class BMUTest(TestCase):
    def instantiate_dut(self):
        # Received codeword
        self.s_x0 = s_x0 = Signal()
        self.s_x1 = s_x1 = Signal()

        # Constraint length k = 3 => 8 branch metrics
        self.bm = bm = [Signal(2, name=f"bm{i}") for i in range(8)]

        m = Module()
        m.submodules.bmu0 = BMU(s_x0, s_x1, bm[0], bm[1], n=0)
        m.submodules.bmu1 = BMU(s_x0, s_x1, bm[2], bm[3], n=1)
        m.submodules.bmu2 = BMU(s_x0, s_x1, bm[4], bm[5], n=2)
        m.submodules.bmu3 = BMU(s_x0, s_x1, bm[6], bm[7], n=3)

        return m

    def set_s(self, s):
        assert s >= 0 and s < 4
        yield self.s_x0.eq(s & 1)
        yield self.s_x1.eq(s >> 1)

    @sync_test_case
    def test_unit_branch_metric_hard_decision(self):
        """Test Hard Decision functionality (width = 1)

        Using test polynomial (g1 = 0b111, g2 = 0b101), compare with Johnson
        2010 Figure 4.6 (b)
        """

        yield from self.set_s(0)
        yield
        assert (yield self.bm[0]) == 0  # this is going from 00 to 00
        assert (yield self.bm[1]) == 2  # this is going from 00 to 10
        assert (yield self.bm[2]) == 2  # this is going from 01 to 00
        assert (yield self.bm[3]) == 0  # this is going from 01 to 10
        assert (yield self.bm[4]) == 1  # this is going from 10 to 01
        assert (yield self.bm[5]) == 1  # this is going from 10 to 11
        assert (yield self.bm[6]) == 1  # this is going from 11 to 01
        assert (yield self.bm[7]) == 1  # this is going from 11 to 11

        yield from self.set_s(1)
        yield
        assert (yield self.bm[0]) == 1  # this is going from 00 to 00
        assert (yield self.bm[1]) == 1  # this is going from 00 to 10
        assert (yield self.bm[2]) == 1  # this is going from 01 to 00
        assert (yield self.bm[3]) == 1  # this is going from 01 to 10
        assert (yield self.bm[4]) == 0  # this is going from 10 to 01
        assert (yield self.bm[5]) == 2  # this is going from 10 to 11
        assert (yield self.bm[6]) == 2  # this is going from 11 to 01
        assert (yield self.bm[7]) == 0  # this is going from 11 to 11

        yield from self.set_s(2)
        yield
        assert (yield self.bm[0]) == 1  # this is going from 00 to 00
        assert (yield self.bm[1]) == 1  # this is going from 00 to 10
        assert (yield self.bm[2]) == 1  # this is going from 01 to 00
        assert (yield self.bm[3]) == 1  # this is going from 01 to 10
        assert (yield self.bm[4]) == 2  # this is going from 10 to 01
        assert (yield self.bm[5]) == 0  # this is going from 10 to 11
        assert (yield self.bm[6]) == 0  # this is going from 11 to 01
        assert (yield self.bm[7]) == 2  # this is going from 11 to 11

        yield from self.set_s(3)
        yield
        assert (yield self.bm[0]) == 2  # this is going from 00 to 00
        assert (yield self.bm[1]) == 0  # this is going from 00 to 10
        assert (yield self.bm[2]) == 0  # this is going from 01 to 00
        assert (yield self.bm[3]) == 2  # this is going from 01 to 10
        assert (yield self.bm[4]) == 1  # this is going from 10 to 01
        assert (yield self.bm[5]) == 1  # this is going from 10 to 11
        assert (yield self.bm[6]) == 1  # this is going from 11 to 01
        assert (yield self.bm[7]) == 1  # this is going from 11 to 11


if __name__ == "__main__":
    unittest.main()
