from nmigen import *
from nmigen.sim import Simulator, Active
from nmigen.back import cxxrtl

from .branch_metric import BMU
from .path_metric import PMU
from .traceback_buffer import Traceback

from .util.test import *
from .convolutional_encode import ConvolutionalCoderSoftware
from functools import partial


class ViterbiDecode(Elaboratable):
    """A R = 1/2 Viterbi Decoder

    Parameters
    ----------
    x0 : Signal(width), in
        Decision for x0 (c1)

    x1 : Signal(width), in
        Decision for x1 (c2)

    data: Signal
        Decoded data

    data_valid: Signal
        Output `data` is valid

    x_max : int, optional
        Value of the largest input decision value. If None, defaults to (2 **
        width) - 1

    k : int, optional
        Constraint length. Defaults to k=3

    g1 : int
        Generator Polynomial 1. There are two generator polynomials for a R =
        1/2 code.

        Defaults to 0b111, which is a common g1 polynomial for a k=3 code

    g2 : int
        Generator Polynomial 2. There are two generator polynomials for a R =
        1/2 code

        Defaults to 0b101, which is a common g2 polynomial for a k=3 code

    input_valid: Signal
        If given, this is used as a valid signal for x1, x0

    register_bm : bool, optional
        Toggles synthesis for registers for branch metrics. Adds an extra cycle
        delay. Enabled by default

    normalise : bool/str, optional
        Perform path normalisation. Required for correct behaviour with long
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

    tb_length : int, optional
        Traceback buffer length. The recommended length is at least 5*k

    tb_domain : str, optional
        Domain to use for the traceback output. Defaults to "sync_tb".

    verbose: bool, optional
        Print useful debugging information during instantiation and elaboration

    """

    def __init__(
        self,
        x0,
        x1,
        data,
        data_valid,
        x_max=None,
        k=3,
        g1=0b111,
        g2=0b101,
        input_valid=None,
        register_bm=True,
        normalise=None,
        pm_width=None,
        minimum_lsb=4,
        tb_length=64,
        tb_domain="sync_tb",
        verbose=True,
    ):
        self.x0 = x0
        self.x1 = x1
        self.data = data
        self.data_valid = data_valid

        if input_valid is None:
            input_valid = Const(1)
        self.input_valid = input_valid

        assert x0.width == x1.width, "Inputs must have the same width"
        self._width = x0.width

        # Our traceback decoder assumes that each codeword is sensitive to the
        # input data on that same state transition. Therefore we ensure that the
        # MSB of both polynomials is set
        assert g1 & 2 ** (k - 1), "Generator polynomial MSB must be set - see code"
        assert g2 & 2 ** (k - 1), "Generator polynomial MSB must be set - see code"

        self._x_max = x_max
        self._register_bm = register_bm
        self._normalise = normalise
        self._tb_domain = tb_domain
        self._verbose = verbose

        self._k = k
        self._g1 = g1
        self._g2 = g2

        self.bm_reg = [
            Signal(self._width + 1, name=f"bm_reg{i}") for i in range(2 ** k)
        ]
        self.input_valid_reg = Signal()

        # PMU and Traceback
        self.pmu = PMU(
            self.bm_reg,
            self.input_valid_reg,
            k=k,
            normalise=normalise,
            pm_width=pm_width,
            minimum_lsb=minimum_lsb,
            verbose=verbose,
        )
        self.traceback = Traceback(
            self.pmu.local_winners,
            self.input_valid_reg,
            data,
            data_valid,
            k=k,
            tb_length=tb_length,
            tb_domain=tb_domain,
            verbose=verbose,
        )

        assert k < 12  # 5k < 64 by common traceback length rule

    def elaborate(self, platform):
        m = Module()

        x0 = self.x0
        x1 = self.x1
        k = self._k

        # Branch Metrics
        bm = [Signal(self._width + 1, name=f"bm{i}") for i in range(2 ** k)]

        # Branch Metric Units
        for i in range(2 ** (k - 1)):
            bmu = BMU(
                x0,
                x1,
                bm[i * 2],
                bm[(i * 2) + 1],
                n=i,
                x_max=self._x_max,
                k=self._k,
                g1=self._g1,
                g2=self._g2,
            )

            m.submodules[f"bmu{i}"] = bmu

        # Register branch metrics
        if self._register_bm:
            with m.If(self.input_valid):
                m.d.sync += [self.bm_reg[i].eq(bm[i]) for i in range(2 ** k)]
            with m.Else():
                m.d.sync += [self.bm_reg[i].eq(0) for i in range(2 ** k)]

            m.d.sync += self.input_valid_reg.eq(self.input_valid)
        else:
            m.d.comb += [self.bm_reg[i].eq(bm[i]) for i in range(2 ** k)]
            m.d.comb += self.input_valid_reg.eq(self.input_valid)

        m.submodules.pmu = self.pmu
        m.submodules.traceback = self.traceback

        return m


class ViterbiTest(TestCase):
    domain = "sync_tb"

    def instantiate_dut(self, k=3, width=1):
        self._k = k
        self._width = width

        g1, g2 = ViterbiTest.polynomials(k)
        self.encoder = ConvolutionalCoderSoftware(k=k, g1=g1, g2=g2)

        self.x0 = Signal(width)
        self.x1 = Signal(width)
        self.x_valid = Signal(reset=1)
        self.data = Signal()
        self.data_valid = Signal()

        return ViterbiDecode(
            self.x0,
            self.x1,
            self.data,
            self.data_valid,
            input_valid=self.x_valid,
            k=k,
            g1=g1,
            g2=g2,
        )

    def add_clocks(self, **kwargs):
        self.sim.add_clock(1 / 1e6, domain="sync")
        self.sim.add_clock(1 / 2e6, domain="sync_tb")

    def set_codeword(self, codeword):
        assert codeword < 4
        c1 = codeword >> 1
        c2 = codeword & 1

        if self._width == 1:
            yield self.x0.eq(c1)  # C1
            yield self.x1.eq(c2)  # C2
        elif self._width == 3:
            # Just to generates some representative waves:
            #
            #            0   1   2   3   4   5   6   7
            #  hard zero |   |   |   |   |   |   |   | hard 1
            #            ^                   ^
            yield self.x0.eq(Mux(c1, 5, 0))  # C1
            yield self.x1.eq(Mux(c2, 5, 0))  # C2
        elif self._width == 4:
            yield self.x0.eq(Mux(c1, 14, 2))  # C1
            yield self.x1.eq(Mux(c2, 13, 1))  # C2
        else:
            assert 0  # Need to add symbols for this width

    @sync_test_case
    def test_unit_zeros(self):
        """Zero state-decoder test"""

        yield from self.set_codeword(0)

        for i in range(128):
            yield

    @staticmethod
    def polynomials(k):
        """Test Polynomials"""
        g1 = {3: 0b111, 4: 0b1101, 5: 0b10011, 7: 0b1111001}[k]
        g2 = {3: 0b101, 4: 0b1010, 5: 0b11101, 7: 0b1011011}[k]

        return g1, g2

    def software_model(self):
        """Test a Viterbi Decoder against a software model"""
        # Test Word is sent little endian
        word = 0x00C0C0C0AAAAAAFF00C0C0C0AAAAAAFF
        word_in = word
        output = []
        always_valid = True

        for i in range(480):
            d = word_in & 1

            if (i % 13 == 2) and not always_valid:
                yield self.x_valid.eq(0)
            else:
                word_in >>= 1

                yield from self.set_codeword(self.encoder.next(d))
                yield self.x_valid.eq(1)

            for _ in [0, 1]:
                if (yield self.data_valid) == 1:
                    output.append((yield self.data))
                yield

        # Discard first two traceback cycles, plus TODO
        start = 2 * self.dut.traceback.tb_length - 1

        bits = "".join([chr(x + 48) for x in output])
        output = int(bits[start : start + 128][::-1], 2)  # -> little endian

        self.assertEqual(f"{word:x}", f"{output:x}")

    sync_test_case_hard = partial(sync_test_case_convolutional_k_3_4_5_7, width=1)
    sync_test_case_q3 = partial(sync_test_case_convolutional_k_3_4_5_7, width=3)
    sync_test_case_q4 = partial(sync_test_case_convolutional_k_3_4_5_7, width=4)

    @sync_test_case_hard
    def test_software_model_hard(self):
        print("> hard")
        yield from self.software_model()

    @sync_test_case_q3
    def test_software_model_q3(self):
        print("> q = 3")
        yield from self.software_model()

    @sync_test_case_q4
    def test_software_model_q4(self):
        print("> q = 4")
        yield from self.software_model()

    def cxxrtl_model(self, width=1):
        """Build cxxrtl model for the Viterbi decoder"""
        import os
        import subprocess

        assert width > 0

        for k in [3, 4, 5, 7]:
            m = Module()

            g1, g2 = ViterbiTest.polynomials(k)

            # Received Codeword
            x0 = Signal(width)
            x1 = Signal(width)

            data = Signal()
            data_valid = Signal()

            m.submodules.decoder = ViterbiDecode(
                x0, x1, data, data_valid, k=k, g1=g1, g2=g2
            )
            output = cxxrtl.convert(m, ports=(x0, x1, data, data_valid))

            root = os.path.join("build")
            if not os.path.exists(root):
                os.mkdir("build")

            if width == 1:
                filename = os.path.join(root, f"viterbi_hard_k{k}.cpp")
                elfname = os.path.join(root, f"viterbi_hard_k{k}.elf")
            else:
                filename = os.path.join(root, f"viterbi_q{width}_k{k}.cpp")
                elfname = os.path.join(root, f"viterbi_q{width}_k{k}.elf")

            testbench = os.path.join(os.path.dirname(__file__), "viterbi_cxxrtl_tb.cpp")

            print(f"Writing CXXRTL model to {filename}..")

            with open(filename, "w") as f:
                f.write(output)
                f.write("\n")
                with open(testbench, "r") as tb:
                    f.write(tb.read())  # Copy testbench from separate file
                f.close()

            print(
                subprocess.check_call(
                    [
                        "clang++",
                        "-I",
                        "/usr/local/share/yosys/include",
                        "-O3",
                        "-std=c++11",
                        "-o",
                        elfname,
                        filename,
                    ]
                )
            )

    def test_cxxrtl_model_hard(self):
        print("> hard")
        self.cxxrtl_model()

    def test_cxxrtl_model_q3(self):
        print("> q = 3")
        self.cxxrtl_model(width=3)

    def test_cxxrtl_model_q4(self):
        print("> q = 4")
        self.cxxrtl_model(width=4)


if __name__ == "__main__":
    unittest.main()
