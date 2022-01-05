from amaranth import *
from amaranth.sim import Simulator, Active

from .add_compare_select import ACSU
from .combinatorial_minimum import CombinatorialMinimum

from .util.test import *


class PMU(Elaboratable):
    """Path Metric Unit (PMU) for a R = 1/2 Viterbi Decoder

    Generates Local Winners

    Parameters
    ----------
    bm : [Signal(width); 2^k], in

        2^k branch metrics. These can be either hard decisions (width = 1) or
        quantized soft decision symbols.

    bm_valid : Signal, optional
        Valid signal for branch metrics

    k : int, optional
        Constraint length. Default is 3

    normalise : bool/str, optional
        Perform path normalisation. Required for correct behavior with long
        sequences if soft decision symbols are used.

        If None, path normalisation is enabled for soft decisions (width >= 2)
        and disabled otherwise.

        Can also be set to the string "not_valid_tick_tock". In this case a
        special normalisation trick is used: the path metrics are normalised
        only on NOT(bm_valid). The bm_valid signal must be de-asserted regularly
        to perform path normalisation.

    pm_width: int, optional
        Register width used for path metric values.

        If None, defaults to 3 + width + int(k/2)

    minimum_lsb: int, optional
        If normalisation is enabled, this value specifies the lsb to use for
        path metric normalisation. All less significant bits are truncated and
        not used in the normalisation process. Must be less than the `pm_width`

        If normalisation is disabled then this value has no effect.

    verbose: bool, optional
        Print useful debugging information during instantiation and elaboration

    Attributes
    ----------

    local_winners : Signal(2^(k-1)), out
        Local winners

    """

    def __init__(
        self,
        bm,
        bm_valid=None,
        k=3,
        normalise=None,
        pm_width=None,
        minimum_lsb=4,
        verbose=True,
    ):
        self.bm = bm
        assert type(bm) == list, "Must supply a list of branch metrics"
        assert len(bm) == 2 ** k, f"There must be {2**k} branch metrics for k={k}"
        assert bm[0].signed == False, "Branch Metrics must be unsigned values"

        if bm_valid is None:
            bm_valid = Const(1)
        self.bm_valid = bm_valid

        if normalise == None:
            normalise = bm[0].width > 1
        if pm_width == None:
            pm_width = 3 + bm[0].width + int(k / 2)  # Estimate
            if verbose:
                print(f"Automatically setting Path Metric Width = {pm_width}")

        self._k = k
        self._normalise = normalise
        self._verbose = verbose

        self._minimum_lsb = minimum_lsb
        self._pm_width = pm_width
        self._pm_max = (2 ** pm_width) - 1
        assert (
            minimum_lsb < pm_width
        ), "Minimum LSB must be less than the PM width itself"

        self.local_winners = Signal(2 ** (k - 1))

    @staticmethod
    def van_der_corput_sequence(n):
        """Integer Van Der Corput sequence base 2, length n"""

        def vdc(n):
            vdc, denom = 0, 1
            while n:
                denom *= 2
                n, remainder = divmod(n, 2)
                vdc += remainder / denom
            return vdc

        return [int(n * vdc(i)) for i in range(n)]

    def elaborate(self, platform):
        m = Module()

        bm = self.bm
        bm_valid = self.bm_valid

        k = self._k
        n_metrics = 2 ** (k - 1)  # Number of path metrics to store

        pm_width = self._pm_width
        pm_max = self._pm_max

        # Storage for path metrics
        pm = [Signal(pm_width, name=f"pm{i}") for i in range(n_metrics)]
        pm_acs_out = [Signal(pm_width, name=f"pm_acs_out{i}") for i in range(n_metrics)]

        # Survivor paths
        survivor_paths = [Signal(name=f"survivor_path{i}") for i in range(n_metrics)]
        m.d.comb += self.local_winners.eq(Cat(survivor_paths))

        # Offset in butterfly structure
        butterfly = 2 ** (k - 2)

        # Van Der Corput sequence for arranging branch metrics in the butterfly
        van_der_corput = PMU.van_der_corput_sequence(butterfly)

        # Instantiate ACS blocks in butterfly structure
        for j in range(butterfly):
            low = 2 * j
            high = (2 * j) + 1
            b = 4 * van_der_corput[j]

            # Low ACSU
            acsu0 = ACSU(
                bm[b + 0] + pm[j],
                bm[b + 2] + pm[j + butterfly],
                pm_acs_out[low],
                survivor_paths[low],
            )
            # High ACSU
            acsu1 = ACSU(
                bm[b + 1] + pm[j],
                bm[b + 3] + pm[j + butterfly],
                pm_acs_out[high],
                survivor_paths[high],
            )

            m.submodules[f"acsu{low}"] = acsu0
            m.submodules[f"acsu{high}"] = acsu1

        if self._normalise:
            if self._normalise == "not_valid_tick_tock":
                # Special normalisation trick: normalise only on NOT(bm_valid)
                #
                # In this case, the bm_valid signal must be de-asserted
                # regularly to perform path normalisation
                if self._verbose:
                    print("PMU Normalisation Mode: not_valid_tick_tock")

                pm_reg = [Signal(pm_width, name=f"pm_reg{i}") for i in range(n_metrics)]
                pm_norm_reg = [
                    Signal(pm_width, name=f"pm_norm_reg{i}") for i in range(n_metrics)
                ]

                bm_valid_reg = Signal()
                m.d.sync += bm_valid_reg.eq(bm_valid)

                # Find minimum path metric: Registered PM
                m.submodules.comb_min = comb_min = CombinatorialMinimum(
                    pm_reg,
                    lsb=self._minimum_lsb,
                )

                # [] <- pm_acs_out
                m.d.sync += [pm_reg[i].eq(pm_acs_out[i]) for i in range(n_metrics)]
                # [] <- normalised
                m.d.sync += [
                    pm_norm_reg[i].eq(pm_reg[i] - comb_min.minimum)
                    for i in range(n_metrics)
                ]

                # Was valid: pm_acs_out -> [] ->
                # Not valid: normalised -> [] ->
                m.d.comb += [
                    pm[i].eq(Mux(bm_valid_reg, pm_reg[i], pm_norm_reg[i]))
                    for i in range(n_metrics)
                ]
            else:
                # Normalise on every cycle
                if self._verbose:
                    print("PMU Normalisation Mode: enabled")

                # Find minimum path metric: Butterfly output
                m.submodules.comb_min = comb_min = CombinatorialMinimum(
                    pm_acs_out,
                    lsb=self._minimum_lsb,
                )

                # Register normalised path metrics
                with m.If(bm_valid):
                    m.d.sync += [
                        pm[i].eq(pm_acs_out[i] - comb_min.minimum)
                        for i in range(n_metrics)
                    ]
        else:
            # Never normalise. In soft decision mode the path metrics may
            # overflow
            if self._verbose:
                print("PMU Normalisation Mode: disabled")

            # Register path metrics
            with m.If(bm_valid):
                m.d.sync += [pm[i].eq(pm_acs_out[i]) for i in range(n_metrics)]

        return m


class PMUTest(TestCase):
    def instantiate_dut(self, k=3):
        self._k = k
        self.bm = [Signal() for _ in range(2 ** k)]
        return PMU(self.bm, k=k, normalise=True)

    def set_bm(self, value):
        for i in range(len(self.bm)):
            yield self.bm[i].eq(value & 1)
            value >>= 1

    @sync_test_case_convolutional_k_3_4_5_7
    def test_unit_path_metrics(self):
        """Test PMU"""

        # Low branches
        low = int(
            "1010101010101010101010101010101010101010101010101010101010101010"
            "1010101010101010101010101010101010101010101010101010101010101010",
            2,
        )
        # High branches
        high = int(
            "0101010101010101010101010101010101010101010101010101010101010101"
            "0101010101010101010101010101010101010101010101010101010101010101",
            2,
        )
        # Expected result for high branches
        maximum_winners = 2 ** (2 ** (self._k - 1)) - 1

        print(f"test k={self._k}")

        # Stream of high branches
        yield from self.set_bm(high)
        for _ in range(self._k):
            yield
        self.assertEqual((yield self.dut.local_winners), maximum_winners)

        yield from self.set_bm(low)  # Stream of low branches
        for _ in range(self._k):
            yield
        self.assertEqual((yield self.dut.local_winners), 0)

        yield from self.set_bm(high)  # Stream of high branches
        for _ in range(self._k):
            yield
        self.assertEqual((yield self.dut.local_winners), maximum_winners)


if __name__ == "__main__":
    unittest.main()
