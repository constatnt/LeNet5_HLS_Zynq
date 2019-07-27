#include "common.h"

#pragma SDS data access_pattern(src:SEQUENTIAL, pool_kernel:SEQUENTIAL, pool_bias:SEQUENTIAL, dst:SEQUENTIAL)
#pragma SDS data zero_copy(src[0:POOL_1_TYPE * 2 * POOL_1_INPUT_WH * POOL_1_INPUT_WH], pool_kernel[0:POOL_1_TYPE * POOL_1_SIZE], pool_bias[0:POOL_1_TYPE], dst[0:POOL_1_TYPE * 2 * POOL_1_OUTPUT_WH * POOL_1_OUTPUT_WH])
void POOLING_LAYER_1(short src[POOL_1_TYPE * 2 * POOL_1_INPUT_WH * POOL_1_INPUT_WH],
					short pool_kernel[POOL_1_TYPE * POOL_1_SIZE],
					short pool_bias[POOL_1_TYPE],
					short dst[POOL_1_TYPE * 2 * POOL_1_OUTPUT_WH * POOL_1_OUTPUT_WH]);


#pragma SDS data access_pattern(src:SEQUENTIAL, pool_kernel:SEQUENTIAL, pool_bias:SEQUENTIAL, dst:SEQUENTIAL)
#pragma SDS data zero_copy(src[0:POOL_2_TYPE * image_Batch*POOL_2_INPUT_WH * POOL_2_INPUT_WH], pool_kernel[0:POOL_2_TYPE*POOL_2_SIZE], pool_bias[0:POOL_2_TYPE], dst[0:POOL_2_TYPE * image_Batch*POOL_2_OUTPUT_WH * POOL_2_OUTPUT_WH])
void POOLING_LAYER_2(short src[POOL_2_TYPE * image_Batch*POOL_2_INPUT_WH * POOL_2_INPUT_WH],
					 short pool_kernel[POOL_2_TYPE*POOL_2_SIZE],
					 short pool_bias[POOL_2_TYPE],
					 short dst[POOL_2_TYPE * image_Batch*POOL_2_OUTPUT_WH * POOL_2_OUTPUT_WH]);

