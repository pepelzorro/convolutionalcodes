from nmigen import *
from nmigen.sim import Simulator, Active

from .util.test import *


class ACSU(Elaboratable):
    """Add-Compare-Select Unit (ACSU) for a R = 1/2 Viterbi Decoder

    Generates path metric and survivor path selection. The survivor path is the
    path with the lowest metric.

    Parameters
    ----------
    pm_low : Signal(width), in
        Path metric for the low branch

    pm_high : Signal(width), in
        Path metric for the high branch

    pm_out : Signal(width), out
        Path metric for the output path

    survivor_path : Signal(), out
        Indicates if the survivor path is low (0) or high (1)

    """

    def __init__(self, pm_low, pm_high, pm_out, survivor_path):
        self.pm_low = pm_low
        self.pm_high = pm_high
        self.pm_out = pm_out
        self.survivor_path = survivor_path

        assert survivor_path.width == 1

    def elaborate(self, platform):
        m = Module()

        pm_low = self.pm_low
        pm_high = self.pm_high
        pm_out = self.pm_out
        survivor_path = self.survivor_path

        # Select survivor path
        m.d.comb += survivor_path.eq(pm_low > pm_high)
        # Mux output path
        m.d.comb += pm_out.eq(Mux(survivor_path, pm_high, pm_low))

        return m


class ACSUTest(TestCase):
    def instantiate_dut(self):
        self.pm_low = Signal(2)
        self.pm_high = Signal(2)
        self.pm_out = Signal(2)
        self.survivor_path = Signal()
        return ACSU(self.pm_low, self.pm_high, self.pm_out, self.survivor_path)

    @sync_test_case
    def test_unit_hard_path_metrics(self):
        """Test the ACSU with Hard Path Metrics"""

        # expected outputs (None = don't care)
        pm_out = [0, 0, 0, 1]
        survivor = [None, 1, 0, None]

        for i in range(4):
            yield self.pm_low.eq(i & 1)
            yield self.pm_high.eq((i & 2) >> 1)
            yield

            if pm_out[i] != None:
                self.assertEqual((yield self.pm_out), pm_out[i])
            if survivor[i] != None:
                self.assertEqual((yield self.survivor_path), survivor[i])

    @sync_test_case
    def test_unit_soft_path_metrics(self):
        """Test the ACSU with Soft Path Metrics"""

        # expected outputs (None = don't care)
        pm_out = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 0, 1, 2, 3]
        survivor = [None, 1, 1, 1, 0, None, 1, 1, 0, 0, None, 1, 0, 0, 0, None]

        for i in range(16):
            yield self.pm_low.eq(i & 3)
            yield self.pm_high.eq((i & 0xC) >> 2)
            yield

            if pm_out[i] != None:
                self.assertEqual((yield self.pm_out), pm_out[i])
            if survivor[i] != None:
                self.assertEqual((yield self.survivor_path), survivor[i])


if __name__ == "__main__":
    unittest.main()
