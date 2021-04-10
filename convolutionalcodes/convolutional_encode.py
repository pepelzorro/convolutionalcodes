from nmigen import *
from nmigen.sim import Simulator, Active

from .util.test import *


class ConvolutionalCoder(Elaboratable):
    """A R = 1/2 Convolutional Encoder

    Parameters
    ----------
    data: Signal(), in
        Input data

    k : int
        Constraint length

    g1 : int / Const() / Signal()
        Generator Polynomial 1. There are two generator polynomials for a R =
        1/2 code

    g2 : int / Const() / Signal()
        Generator Polynomial 2. There are two generator polynomials for a R =
        1/2 code

    Attributes
    ----------

    c : Signal(2), out
        Codeword c1 c2

    """

    def __init__(self, data, k=3, g1=0b111, g2=0b101):
        self.data = data
        assert data.width == 1

        if type(g1) == int:
            g1 = Const(g1, k)
        if type(g2) == int:
            g2 = Const(g2, k)

        assert g1.width == k
        assert g2.width == k

        self._k = k
        self._g1 = g1
        self._g2 = g2

        self.c = Signal(2)

    def elaborate(self, platform):
        m = Module()

        k = self._k
        g1 = self._g1
        g2 = self._g2

        # Shift register
        reg = Signal(k - 1)
        state = Signal(k)

        m.d.comb += state.eq(Cat(reg, self.data))
        m.d.sync += reg.eq(state[1:])

        # G1 G2
        c1 = Signal()
        c2 = Signal()
        m.d.comb += c1.eq((state & g1).xor())
        m.d.comb += c2.eq((state & g2).xor())

        # Output
        m.d.comb += self.c.eq(Cat(c2, c1))  # LSB, MSB

        return m


class ConvolutionalCoderSoftware:
    """Software implementation of a R = 1/2 Convolutional Encoder

    Parameters
    ----------
    k : int
        Constraint length

    g1 : int
        Generator Polynomial 1. There are two generator polynomials for a R =
        1/2 code

    g2 : int
        Generator Polynomial 2. There are two generator polynomials for a R =
        1/2 code

    Methods
    -------
    next(int) : int
        Returns codeword

    """

    def __init__(self, k=3, g1=0b111, g2=0b101):
        self._k = k
        self._g1 = g1
        self._g2 = g2

        self.reg = 0

    @staticmethod
    def parity_kr(x):
        bit = 0
        parity = False
        while x:
            parity = not parity
            x = x & (x - 1)  # Clear least significant bit that is set

        return int(parity)

    def next(self, data):
        # Shift register
        state = (1 << (self._k - 1) if data == 1 else 0) + self.reg
        self.reg = state >> 1

        # G1 G2
        c1 = ConvolutionalCoderSoftware.parity_kr(state & self._g1)
        c2 = ConvolutionalCoderSoftware.parity_kr(state & self._g2)

        # Output
        return (c1 * 2) + c2


class ConvolutionalCoderTest(TestCase):
    def instantiate_dut(self, k=3):
        # test polynomials
        g1 = {3: 0b111, 4: 0b1101, 5: 0b10011, 7: 0b1111001}[k]
        g2 = {3: 0b101, 4: 0b1010, 5: 0b11101, 7: 0b1011011}[k]

        self.k = k
        self.model = ConvolutionalCoderSoftware(k=k, g1=g1, g2=g2)
        self.data = Signal()
        return ConvolutionalCoder(self.data, k=k, g1=g1, g2=g2)

    @sync_test_case
    def test_unit_encode_johnson_k3(self):
        """Test Convolutional Encoder against Johnson 2010 Example 4.4"""

        # Johnson 2010 Example 4.4
        # message u = [1 1 0]
        # codeword [c1 c2]  = [11 01 01 11 00]

        yield self.data.eq(1)
        yield
        assert (yield self.dut.c) == 0b11

        yield self.data.eq(1)
        yield
        assert (yield self.dut.c) == 0b01

        yield self.data.eq(0)
        yield
        assert (yield self.dut.c) == 0b01

        # Termination...
        yield
        assert (yield self.dut.c) == 0b11

        yield
        assert (yield self.dut.c) == 0b00

        for i in range(10):
            yield

    @sync_test_case_convolutional_k_3_4_5_7
    def test_unit_software_model(self):
        """Compare Convolutional Encoder against software model"""

        print(f"Testing software model for k={self.k}")
        # Test Word
        word = 0x485743BD3923DA93284920121328543BCB843984A0129

        while word:
            d = word & 1
            word >>= 1

            yield self.data.eq(d)
            yield

            assert (yield self.dut.c) == self.model.next(d)

        # Terminate..
        for _ in range(self.k - 1):
            yield self.data.eq(0)
            yield

            assert (yield self.dut.c) == self.model.next(0)


if __name__ == "__main__":
    unittest.main()
