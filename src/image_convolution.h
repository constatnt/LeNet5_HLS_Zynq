#include "common.h"

void kernel_load(char *fileName, float * target);
void TransQ(float *src, short *dst, int size);
void PrintQ(short *src, int size);

// Convolution Layer 1
// Function by Batch_size(10)
// Input_feature_map[32x32],  Conv_kernel[6][25], Bias[6], Output_feature_map[6][28x28]
//#pragma SDS data access_pattern(input_feature:SEQUENTIAL, conv_kernel:SEQUENTIAL, conv_bias:SEQUENTIAL, output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature[0:INPUT_WH * INPUT_WH*2], conv_kernel[0:CONV_1_TYPE * CONV_1_WH * CONV_1_WH], conv_bias[0:CONV_1_TYPE], output_feature[0:CONV_1_TYPE * CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH*2])
void CONVOLUTION_LAYER_1(short input_feature[INPUT_WH * INPUT_WH * 2],
						short conv_kernel[CONV_1_TYPE * CONV_1_WH * CONV_1_WH],
						short conv_bias[CONV_1_TYPE],
						short output_feature[CONV_1_TYPE * CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH * 2]);


// Convolution Layer 2
// Function by Batch_size(10)
// Input_feature_map[6][14x14],  Conv_kernel[16][6][25], Bias[16], Output_feature_map[16][10x10]
#pragma SDS data access_pattern(input_feature:SEQUENTIAL, conv_kernel:SEQUENTIAL, conv_bias:SEQUENTIAL, output_feature:SEQUENTIAL)
#pragma SDS data zero_copy(input_feature[0:CONV_1_TYPE* CONV_2_INPUT_WH * CONV_2_INPUT_WH*10], conv_kernel[0:CONV_2_TYPE * CONV_1_TYPE * CONV_1_WH * CONV_1_WH], conv_bias[0:CONV_2_TYPE], output_feature[0:CONV_2_TYPE * CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH*10])
void CONVOLUTION_LAYER_2(short input_feature[CONV_1_TYPE * CONV_2_INPUT_WH *CONV_2_INPUT_WH*10],
						 short conv_kernel[CONV_2_TYPE * CONV_1_TYPE * CONV_2_WH * CONV_2_WH],
						 short conv_bias[CONV_2_TYPE],
						 short output_feature[CONV_2_TYPE * CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH*10]);


// Convolution Layer 3 (FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5],  Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]

void CONVOLUTION_LAYER_3(short input_feature[CONV_2_TYPE*CONV_3_INPUT_WH *CONV_3_INPUT_WH],
						 short conv_kernel[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH], short conv_bias[CONV_3_TYPE],
						 short output_feature[CONV_3_TYPE]);

#pragma SDS data access_pattern(input_feature:SEQUENTIAL, conv_kernel:SEQUENTIAL, conv_bias:SEQUENTIAL, output_feature:SEQUENTIAL)
//#pragma SDS data zero_copy(input_feature[0:CONV_2_TYPE*CONV_3_INPUT_WH *CONV_3_INPUT_WH], conv_kernel[0:CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3], output_feature[0:CONV_3_TYPE/3])
void CONVOLUTION_LAYER_3_Sub(short input_feature[CONV_2_TYPE*CONV_3_INPUT_WH *CONV_3_INPUT_WH],
						 short conv_kernel[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3], short conv_bias[CONV_3_TYPE/3],
						 short output_feature[CONV_3_TYPE/3]);
