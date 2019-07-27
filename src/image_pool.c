#include "image_pool.h"

void POOLING_LAYER_1(short src[POOL_1_TYPE * 2 * POOL_1_INPUT_WH * POOL_1_INPUT_WH],
					short pool_kernel[POOL_1_TYPE * POOL_1_SIZE],
					short pool_bias[POOL_1_TYPE],
					short dst[POOL_1_TYPE * 2 * POOL_1_OUTPUT_WH * POOL_1_OUTPUT_WH])
{
	int row, col, row_sub, col_sub, batch_cnt, depth;

	int value[4];
	#pragma HLS RESOURCE variable=value core=FMul_fulldsp

	short buff_pool_kernel[POOL_1_TYPE][POOL_1_SIZE];
	#pragma HLS ARRAY_PARTITION variable=buff_pool_kernel cyclic factor=6 dim=1
	short buff_pool_bias[POOL_1_TYPE];
	#pragma HLS ARRAY_PARTITION variable=buff_pool_bias cyclic factor=6 dim=1

	Buff:for (row = 0; row < POOL_1_TYPE; row++)
	{
		#pragma HLS PIPELINE II=1
		for (col = 0; col < POOL_1_SIZE; col++)
		{
			buff_pool_kernel[row][col] = pool_kernel[col + row * POOL_1_SIZE];
		}
		buff_pool_bias[row] = pool_bias[row];
	}

	short temp;

	Lv_1:for (batch_cnt = 0; batch_cnt < 2; batch_cnt++)
	{
		Lv_2:for (depth = 0; depth < POOL_1_TYPE; depth++)
		{
			Lv_3:for (row = 0; row < POOL_1_OUTPUT_WH; row++)
			{
				Lv_4:for (col = 0; col < POOL_1_OUTPUT_WH; col++)
				{
					#pragma HLS PIPELINE II=1
					// Computation of Pooling
					/*
					value[0] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_1_INPUT_WH + (col * 2)] * pool_kernel[depth*POOL_1_TYPE];
					value[1] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_1_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth*POOL_1_TYPE+1];
					value[2] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2)] * pool_kernel[depth*POOL_1_TYPE+2];
					value[3] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth*POOL_1_TYPE+3];
					*/
					value[0] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_1_INPUT_WH + (col * 2)] * buff_pool_kernel[depth][0];
					value[1] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_1_INPUT_WH + (col * 2 + 1)] * buff_pool_kernel[depth][1];
					value[2] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2)] * buff_pool_kernel[depth][2];
					value[3] = src[POOL_1_INPUT_SIZE * POOL_1_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_1_INPUT_WH + (col * 2 + 1)] * buff_pool_kernel[depth][3];

					temp = value[0] + value[1] + value[2] + value[3];
					// Activation function
					dst[batch_cnt * POOL_1_TYPE * POOL_1_OUTPUT_WH + depth * POOL_1_OUTPUT_WH + row * POOL_1_OUTPUT_WH + col] = (temp + pool_bias[depth]);
				}
			}
		}
	}
}


void POOLING_LAYER_2(short src[POOL_2_TYPE * image_Batch*POOL_2_INPUT_WH * POOL_2_INPUT_WH],
					 short pool_kernel[POOL_2_TYPE*POOL_2_SIZE],
					 short pool_bias[POOL_2_TYPE],
					 short dst[POOL_2_TYPE * image_Batch*POOL_2_OUTPUT_WH * POOL_2_OUTPUT_WH])
{
	int row, col, row_sub, col_sub, batch_cnt=0, depth;
	int value[4];
	short temp;

	short buff_pool_kernel[POOL_2_TYPE][POOL_2_SIZE];
	#pragma HLS ARRAY_PARTITION variable=buff_pool_kernel cyclic factor=16 dim=1
	short buff_pool_bias[POOL_2_TYPE];
	#pragma HLS ARRAY_PARTITION variable=buff_pool_bias cyclic factor=16 dim=1

	Buff:for (row = 0; row < POOL_2_TYPE; row++)
	{
		#pragma HLS PIPELINE II=1
		for (col = 0; col < POOL_2_SIZE; col++)
		{
			buff_pool_kernel[row][col] = pool_kernel[col + row * POOL_1_SIZE];
		}
		buff_pool_bias[row] = pool_bias[row];
	}


	Lv_1:for (batch_cnt = 0; batch_cnt < image_Batch; batch_cnt++)
	{
		Lv_2:for (depth = 0; depth < POOL_2_TYPE; depth++)
		{
			Lv_3:for (row = 0; row < POOL_2_OUTPUT_WH; row++)
			{
				Lv_4:for (col = 0; col < POOL_2_OUTPUT_WH; col++)
				{
					#pragma HLS PIPELINE II=1
					// Computation of Pooling
					/*
					value[0] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_2_INPUT_WH + (col * 2)] * pool_kernel[depth*POOL_2_TYPE];
					value[1] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_2_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth*POOL_1_TYPE+1];
					value[2] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2)] * pool_kernel[depth*POOL_1_TYPE+2];
					value[3] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2 + 1)] * pool_kernel[depth*POOL_1_TYPE+3];
					*/

					value[0] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_2_INPUT_WH + (col * 2)] * buff_pool_kernel[depth][0];
					value[1] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2) * POOL_2_INPUT_WH + (col * 2 + 1)] * buff_pool_kernel[depth][1];
					value[2] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2)] * buff_pool_kernel[depth][2];
					value[3] = src[POOL_2_INPUT_SIZE * POOL_2_TYPE * batch_cnt + depth * POOL_1_INPUT_SIZE + (row * 2 + 1) * POOL_2_INPUT_WH + (col * 2 + 1)] * buff_pool_kernel[depth][3];


					temp = value[0] + value[1] + value[2] + value[3];
					// Activation function
					dst[batch_cnt * POOL_2_TYPE * POOL_2_OUTPUT_WH + depth * POOL_2_OUTPUT_WH + row * POOL_2_OUTPUT_WH + col] = (temp + buff_pool_bias[depth]);
				}
			}
		}
	}
}
