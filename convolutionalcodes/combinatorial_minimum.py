from nmigen import *
from nmigen.sim import Simulator, Active

from .util.test import *
from functools import partial


class CombinatorialMinimum(Elaboratable):
    """Combinatorial minimum of a set of metrics

    Includes the option to truncate bits from each metric. This reduces the
    amount of logic required, whilst still guaranteeing that the output is less
    than or equal to the minimum metric value.

    Parameters
    ----------
    pm : [Signal(width); N], in
        Input metrics. All metrics must have the same length. N must be a power
        of 2

    lsb : int, optional
        The index of the LSB to use in the comparison. All less significant bits
        will not be used in the comparison/multiplexing and will be zero in the
        `minimum` value.

    Attributes
    ----------

    minimum : Signal(width), out
        Less than or equal to the minimum value from `pm`

    """

    def __init__(self, pm, lsb=0):
        self.pm = pm
        assert type(pm) == list, "Must supply a list of metrics"
        assert len(pm) & (len(pm) - 1) == 0, "Number of metrics must be a power of 2"
        assert len(pm) >= 2, "Must have at least two metrics to compare"
        assert not pm[0].signed or lsb == 0, "Truncation valid only for unsigned values"

        self._width = width = pm[0].width
        self.minimum = Signal(unsigned(width), reset=0)

        self._lsb = lsb
        assert type(lsb) == int
        assert lsb >= 0 and lsb < width, "LSB must refer to a bit in the input metric"

    def elaborate(self, platform):
        m = Module()

        lsb = self._lsb
        width = self._width - lsb

        pm = [p[lsb:] for p in self.pm]  # Truncate 'lsb' bits

        def minimum(ia, ib, step, pm_out):
            if step == 1:  # Leaf
                m.d.comb += pm_out.eq(
                    Mux(
                        pm[ia] < pm[ib],
                        pm[ia],
                        pm[ib],
                    )
                )
            else:  # Stem
                pm_a = Signal(width)
                pm_b = Signal(width)
                step = int(step / 2)
                minimum(ia, ia + step, step, pm_a)
                minimum(ib, ib + step, step, pm_b)

                m.d.comb += pm_out.eq(Mux(pm_a < pm_b, pm_a, pm_b))

        # Trunk
        minimum(0, int(len(pm) / 2), int(len(pm) / 2), self.minimum[lsb:])

        return m


class CombinatorialMinimumTest(TestCase):
    def instantiate_dut(self, lsb=0):
        self.n = n = 8
        self.width = width = 4
        self.pm = [Signal(width, name=f"pm{i}") for i in range(n)]
        return CombinatorialMinimum(self.pm, lsb=lsb)

    @sync_test_case
    def test_unit_simple(self):
        """Test combinatorial minimum, without truncation"""

        for m in range(8):
            # Fill in n values, with magnitude at least m
            for i in range(self.n):
                yield self.pm[i].eq(m + abs(m - i))
            yield

            self.assertEqual((yield self.dut.minimum), m)

    sync_test_case_truncate = partial(sync_test_case, lsb=1)

    @sync_test_case_truncate
    def test_unit_truncation(self):
        """Test combinatorial minimum with truncation"""

        for m in range(8):
            # Fill in n values, with magnitude at least m
            for i in range(self.n):
                yield self.pm[i].eq(m + abs(m - i))
            yield

            self.assertLessEqual((yield self.dut.minimum), m)


if __name__ == "__main__":
    unittest.main()
