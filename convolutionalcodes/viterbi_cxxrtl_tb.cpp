/**
 * CXXRTL testbench for viterbi_top
 */

#include <iostream>
#include <fstream>
#include <backends/cxxrtl/cxxrtl_vcd.h>

int viterbi_clock_and_output(cxxrtl_design::p_top *top, value<1> sync_clk) {

    unsigned int out_data, out_data_valid;

    top->p_clk = sync_clk;
    top->step();
    top->p_sync__tb__clk = value<1>{0u};
    top->step();
    top->p_sync__tb__clk = value<1>{1u};

    /* Output wire */
    out_data = top->p_data.get<unsigned int>();
    out_data_valid = top->p_data__valid.get<unsigned int>();

    if (out_data_valid == 1) {
        std::cout << out_data << std::endl;
    }

    return 0;
}

template <size_t Bits>
int viterbi_getc_value(value<Bits> *val) {

    int c;
    unsigned int value;

    while (1) {
        c = getchar();
        if (c >= 48 && c < 58) {
            /* numeric value */
            value = c - 48;
            val->set(value);
            return 0;

        } else if (c >= 97 && c < 123) {
            /* letters, use as numbers */
            value = c - 97;
            val->set(value);
            return 0;

        } else if (c <= 32) {
            /* whitespace, continue */
        } else {
          return -1; /* end of input */
        }
    }
}

int main(int argc, char *argv[]) {

    cxxrtl_design::p_top top;
    vcd_writer w;
    int input = 0;
    int capture_vcd = 0;
    int j = 0;

    debug_items items;
    top.debug_info(items);

    if (argc > 1) {
        capture_vcd = 1;
        w.add(items);
    }

    for (int i=0; i < 256; i++) {

        if (input == 0) {
            /* X0, C1 */
            input = viterbi_getc_value(&top.p_x0);

            if (input == 0) {
                /* X1, C2 */
                input = viterbi_getc_value(&top.p_x1);

                i = 0; /* Keep reading input */
            }
        }

        if (capture_vcd == 1) {
          w.sample(j);
          j += 1;
        }

        /* Clock tb domain */
        viterbi_clock_and_output(&top, value<1>{0u});

        if (capture_vcd == 1) {
          w.sample(j);
          j += 1;
        }

        /* Clock tb domain */
        viterbi_clock_and_output(&top, value<1>{1u});

        // if (capture_vcd == 1) {
        //   w.sample(j);
        //   j += 1;
        // }

        // /* Clock tb domain */
        // viterbi_clock_and_output(&top, value<1>{0u});

        // if (capture_vcd == 1) {
        //   w.sample(j);
        //   j += 1;
        // }

        // /* Clock tb domain */
        // viterbi_clock_and_output(&top, value<1>{1u});
    }

    if (capture_vcd == 1) {
      std::cout << "Writing " << argv[1] << std::endl;
      std::ofstream outfile (argv[1]);

      outfile << w.buffer;
    }

    return 0;
}
