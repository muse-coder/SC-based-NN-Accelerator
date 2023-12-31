`include "Dff.sv"

module Cmp #(
    parameter IWID = 10
) (
    input logic clk,
    input logic rst_n,
    input logic [IWID - 1 : 0] iData,
    input logic [IWID - 1 : 0] iRng,
    output logic oBit
);

    logic tBit;

    assign tBit = iData >= iRng;

    Dff U_Dff(
        .clk(clk),
        .rst_n(rst_n),
        .in(tBit),
        .out(oBit)
    );

endmodule