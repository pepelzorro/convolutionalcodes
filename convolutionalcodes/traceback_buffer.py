from nmigen import *
from nmigen.lib.cdc import *
from nmigen.sim import Simulator, Active

import numpy as np

from decimal import Decimal
from .util.test import *
from functools import partial


class Traceback(Elaboratable):
    """Traceback buffer for a R = 1/2 Viterbi Decoder

    Stores local winners from PMU.

    Reads out traced global winners.

    Domains
    -------

    Fixed Traceback Recursion Rate (TRR) = 2

    "sync" - n
    "sync_tb" - 2n

    phase = 1
    +--------+--------+--------+
    |          /\     _    _/  |
    |            \_  / \__/ \  |
    |              \/     \__  |
    +--------+--------+--------+
     -------> <----------------
      write   readout traceback

    phase = 2
    +--------+--------+--------+
    |    _/            /\     _|
    |\__/ \              \_  / |
    |   \__                \/  |
    +--------+--------+--------+
     -------- -------> <-------
    traceback  write   readout

    phase = 0
    +--------+--------+--------+
    | /\     _    _/           |
    |   \_  / \__/ \           |
    |     \/     \__           |
    +--------+--------+--------+
     <---------------- ------->
     readout traceback  write

    Parameters
    ----------
    local_winners : Signal(2^(k-1)), in
        Local winners

    lw_valid: Signal(), in
        Input valid signal

    data: Signal(), out
        Decoded data

    data_valid: Signal(), out

    k : int, optional
        Constraint length. Defaults to k=3

    tb_length : int, optional
        Length of the traceback buffer. Defaults to 5 * k, rounded up to the
        nearest power of 2.

        The fixed Traceback Recursion Rate (TRR) of 2 results in a total memory
        depth 3 * the length of the traceback buffer.

    in_domain : str, optional
        Domain to use for the input signals. Defaults to "sync".

    tb_domain : str, optional
        Domain to use for the traceback output. Defaults to "sync_tb".

    verbose: bool, optional
         Print useful debugging information during instantiation and elaboration

    Attributes
    ----------

    tb_length: int
        The trackback length actually being used. This will always be a power of
        2. Can be used to determine how many cycles of flushing are required to
        ensure all input data has been flushed from the output buffer.

    """

    def __init__(
        self,
        local_winners,
        lw_valid,
        data,
        data_valid,
        k=3,
        tb_length=None,
        in_domain="sync",
        tb_domain="sync_tb",
        verbose=True,
    ):
        self.local_winners = local_winners
        assert local_winners.width == 2 ** (k - 1)

        self.data = data
        self.data_valid = data_valid

        self._k = k
        self._in_domain = in_domain
        self._tb_domain = tb_domain
        self._verbose = verbose

        if tb_length == None:
            tb_length = 5 * k  # Common rule of thumb

        tb_order = 0 if tb_length & (tb_length - 1) else -1
        while tb_length:
            tb_order += 1
            tb_length >>= 1
        # tb_order is a power of 2

        if verbose:
            print(f"Traceback Order = {tb_order} Length = {2**tb_order}")

        self.tb_length = 2 ** tb_order
        self._tb_order = tb_order
        self._depth = 3 * (2 ** tb_order)  # Fixed TRR = 2
        # depth is _not_ a power of two

        # Memory ports
        #
        # The address_r reset value gives it a 'head start' in the first phase
        # (which is garbage anyhow). This ensures subsequent phases begins on
        # the phase signal transition
        self.address_w = Signal(self._tb_order + 2)
        self.address_r = Signal(self._tb_order + 2, reset=self._depth - 7)
        self.we = lw_valid
        self.mem = Memory(width=2 ** (k - 1), depth=self._depth)

    def elaborate(self, platform):
        m = Module()

        local_winners = self.local_winners
        data = self.data
        data_valid = self.data_valid
        k = self._k

        # Write port
        m.submodules.wrport = wrport = self.mem.write_port()
        m.d.comb += [
            wrport.addr.eq(self.address_w),
            wrport.data.eq(self.local_winners),
            wrport.en.eq(self.we),
        ]

        # Increment address
        with m.If(self.we):
            m.d.sync += self.address_w.eq(
                Mux(self.address_w == self._depth - 1, 0, self.address_w + 1)
            )

        # Traceback
        tb_rdata = Signal(2 ** (k - 1))

        # Read port
        m.submodules.rdport = rdport = self.mem.read_port(domain="sync_tb")
        m.d.comb += [
            rdport.addr.eq(self.address_r),
            tb_rdata.eq(rdport.data),
        ]

        # Three phase readout
        phase = Signal(2, reset=1)

        # Start of phase - determined by write address
        with m.If(self.we):
            # Launch phase signal one cycle early to account for synchronizer
            with m.If(
                self.address_w[: self._tb_order] == (Repl(1, self._tb_order) - 1)
            ):
                m.d.sync += phase.eq(Mux(phase == 2, 0, phase + 1))

        # ---- sync_tb domain ----
        # Clock Domain Crossing:
        #
        # The phase signal only has three states (00, 01, 10) such that only one
        # bit changes on each transition. Therefore it can safely cross clock
        # domains using a two-stage FF Synchronizer
        #
        # The timing between the read and write addresses holds for a 2:1 ratio
        # between the sync_tb and sync clocks. In the case of a slower sync_tb
        # clock, it is possible for the last readout word to be corrupted.
        #
        # It is also possible that the first readout word is corrupted. We
        # ignore this, because our traceback is long enough.

        phase_stb = Signal(2)
        m.submodules.phase_synchronizer = FFSynchronizer(
            phase,  # in
            phase_stb,  # out
            o_domain="sync_tb",
            stages=2,
        )

        traceback_active = Signal()
        phase_stb_reg = Signal(2)
        m.d.sync_tb += phase_stb_reg.eq(phase_stb)

        # Decrement traceback address, until we reach the end of phase
        with m.If(self.address_r != (phase_stb << self._tb_order)):
            m.d.sync_tb += self.address_r.eq(
                Mux(self.address_r == 0, self._depth - 1, self.address_r - 1)
            )
            m.d.sync_tb += traceback_active.eq(1)
        with m.Else():
            m.d.sync_tb += traceback_active.eq(0)

        # Traceback
        state = Signal(k - 1, reset=0)  # Start on fixed state
        m.d.sync_tb += state.eq(Cat(state[1:], tb_rdata.bit_select(state, 1)))

        # Tracebuffer Readout
        tb_readout_valid = Signal()
        m.d.comb += tb_readout_valid.eq(
            (self.address_r[-2:] == phase_stb_reg) & traceback_active
        )

        # Readout LIFO
        lifo_register = Signal(2 ** self._tb_order)
        lifo_level = Signal(self._tb_order + 1)

        with m.If(tb_readout_valid):
            m.d.sync_tb += lifo_register.eq(Cat(state[0], lifo_register))
            m.d.sync_tb += lifo_level.eq(2 ** self._tb_order)
        with m.Else():
            m.d.sync_tb += lifo_register.eq(lifo_register[1:])
            with m.If(lifo_level > 0):
                m.d.sync_tb += lifo_level.eq(lifo_level - 1)

        # Output
        m.d.comb += data_valid.eq(~tb_readout_valid & (lifo_level > 0))
        m.d.comb += data.eq(Mux(data_valid, lifo_register[0], 0))

        return DomainRenamer({"sync": self._in_domain, "sync_tb": self._tb_domain})(m)


class TracebackTest(TestCase):
    domain = "sync_tb"

    def instantiate_dut(self, k=3, sync_domain="sync"):
        self._k = k
        self.local_winners = Signal(2 ** (k - 1))
        self.lw_valid = Signal(reset=1)
        self.data = Signal()
        self.data_valid = Signal()

        return Traceback(
            self.local_winners,
            self.lw_valid,
            self.data,
            self.data_valid,
            k=k,
            in_domain=sync_domain,
        )

    def add_clocks(self):
        self.sim.add_clock(1 / 1e6, domain="sync")
        self.sim.add_clock(1 / 2e6, domain="sync_tb")

    def clock_out_data(self):
        if (yield self.data_valid):
            self.output.append((yield self.data))
            self.output_count += 1
        yield

    def lw_sequence(self, i, k=3):
        """Return a test sequence of lws that repeats every 13 symbols"""

        max_lw = (2 ** (2 ** (k - 1))) - 3
        return int(i % 13 * max_lw / 13)

    def expected_data(self, k=3):
        """Expected data given the sequence of lws generated by `lw_sequence`"""
        return {
            3: [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        }[k]

    @sync_test_case
    def test_unit_traceback_different_domains(self):
        """Test the traceback buffer with the input and traceback output in different
        domains"""

        N = 256

        self.output = []
        self.output_count = 0

        for i in range(N):
            # Supply local winners
            yield self.lw_valid.eq(1)
            yield self.local_winners.eq(self.lw_sequence(i, k=self._k))

            yield from self.clock_out_data()
            yield from self.clock_out_data()

        # Flush
        for i in range(self.dut.tb_length):
            yield from self.clock_out_data()

        # Check output length
        self.assertEqual(self.output_count, N)
        # Discard first two tracebacks
        output = self.output[2 * self.dut.tb_length - 1 :]

        # Check output
        np.testing.assert_array_equal(
            output,
            np.resize(self.expected_data(self._k), len(output)),
        )

    sync_test_case_same_domain = partial(sync_test_case, sync_domain="sync_tb")

    @sync_test_case_same_domain
    def test_unit_traceback_same_domain(self):
        """Test the traceback buffer with the input and traceback output in the same
        domain"""

        N = 256

        self.output = []
        self.output_count = 0

        for i in range(N):
            # Supply local winners at 50% duty cycle
            yield self.lw_valid.eq(1)
            yield self.local_winners.eq(self.lw_sequence(i, k=self._k))
            yield from self.clock_out_data()

            yield self.lw_valid.eq(0)
            yield from self.clock_out_data()

        # Flush
        for i in range(self.dut.tb_length):
            yield from self.clock_out_data()

        # Check output length
        self.assertEqual(self.output_count, N)
        # Discard first two tracebacks
        output = self.output[2 * self.dut.tb_length :]

        # Check output
        np.testing.assert_array_equal(
            output,
            np.resize(self.expected_data(self._k), len(output)),
        )


if __name__ == "__main__":
    unittest.main()
