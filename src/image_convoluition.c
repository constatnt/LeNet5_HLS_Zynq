#include "image_convolution.h"

void kernel_load(char *fileName, float * target)
{
	int read_i = 0;
	FILE* read_ptr;

	if (read_ptr = fopen(fileName, "rb"))
	{
		while (EOF != fscanf(read_ptr, "%f", &target[read_i]))
		{
			read_i++;
		}
	}
	fclose(read_ptr);

}

void TransQ(float *src, short *dst, int size)
{
	int i;

	for (i = 0; i < size; i++)
		dst[i] = TOFIX(src[i], QUANT_SIZE);
}

void PrintQ(short *src, int size)
{
	int i;
	for (i = 0; i < size; i++)
		printf("%.9f \n", TOFLT(src[i], QUANT_SIZE));
}


void kernel_write(char *fileName, short * source, short *target)
{
	int read_i = 0;
	FILE* read_ptr;

	if (read_ptr = fopen(fileName, "rb"))
	{
		while (EOF != fscanf(read_ptr, "%f", &target[read_i]))
		{
			read_i++;
		}
	}
	fclose(read_ptr);

}
/*
short FMUL(short a, short b)
{
	int temp = ((int)a * (int)b)>>QUANT_SIZE;
	return (short)temp;
}
*/


// Convolution Layer 1
// Function by Batch_size(10)
// Input_feature_map[32x32xN],  Conv_kernel[6][25], Bias[6], Output_feature_map[6*N][28x28]
void CONVOLUTION_LAYER_1(short input_feature[INPUT_WH * INPUT_WH * 2],
					 	 short conv_kernel[CONV_1_TYPE * CONV_1_WH * CONV_1_WH],
						 short conv_bias[CONV_1_TYPE],
						 short output_feature[CONV_1_TYPE * CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH * 2])
{

	short col, row, col_f, row_f;
	short depth_out, batch_cnt;
	
	short temp[5];
	short fvalue;
	int vec_0[5];

	short buff_input_feature[INPUT_WH*2][INPUT_WH];
	#pragma HLS ARRAY_PARTITION variable=buff_input_feature cyclic factor=16 dim=1

	short buff_conv_kernel[CONV_1_TYPE][CONV_1_WH*CONV_1_WH];
	#pragma HLS ARRAY_PARTITION variable=buff_conv_kernel cyclic factor=6 dim=1

	short buff_conv_bias[CONV_1_TYPE];
	#pragma HLS ARRAY_PARTITION variable=buff_conv_bias cyclic factor=6 dim=1

	Feature_Buff:
	for(depth_out=0;depth_out<2;depth_out++){
		for(row=0;row<INPUT_WH;row++){
		#pragma HLS PIPELINE II=1
			for(col=0;col<INPUT_WH;col++) {
				buff_input_feature[INPUT_WH*depth_out+row][col] = input_feature[depth_out*INPUT_WH*INPUT_WH + INPUT_WH * row + col];
			}
		}
	}

	Conv_Buff:
	for(depth_out=0;depth_out<CONV_1_TYPE;depth_out++) {
	#pragma HLS PIPELINE II=1
		for(row=0;row<CONV_1_WH*CONV_1_WH;row++)
		{
			buff_conv_kernel[depth_out][row] = conv_kernel[depth_out * 25 + (row/5) * CONV_1_WH + (row%5)];
		}
		buff_conv_bias[depth_out] =  conv_bias[depth_out];
	}

	// Convolution
	Lv_0:for(batch_cnt=0; batch_cnt<2;batch_cnt++) {
		Lv_1:for (depth_out = 0; depth_out < CONV_1_TYPE; depth_out++) {
			Lv_2:for (row = 0; row < CONV_1_OUTPUT_WH; row++) {
				Lv_3:for (col = 0; col < CONV_1_OUTPUT_WH; col++) {

					// Initialization of Reduction-Sum
					//fvalue=0;
#pragma HLS PIPELINE II=1
					// Multiplication by Convolution and Input feature map
					Lv_4:for (row_f = 0; row_f < CONV_1_WH; row_f++) {

						Lv_5:for (col_f = 0; col_f < CONV_1_WH; col_f++)
						{

							//vec_0[0] = (buff_input_feature[batch_cnt*INPUT_WH+row+row_f][col] * buff_conv_kernel[depth_out][row_f*CONV_1_WH])>>QUANT_SIZE;
							//vec_0[1] = (buff_input_feature[batch_cnt*INPUT_WH+row+row_f][col+1] * buff_conv_kernel[depth_out][row_f*CONV_1_WH+1])>>QUANT_SIZE;
							//vec_0[2] = (buff_input_feature[batch_cnt*INPUT_WH+row+row_f][col+2] * buff_conv_kernel[depth_out][row_f*CONV_1_WH+2])>>QUANT_SIZE;
							//vec_0[3] = (buff_input_feature[batch_cnt*INPUT_WH+row+row_f][col+3] * buff_conv_kernel[depth_out][row_f*CONV_1_WH+3])>>QUANT_SIZE;
							//vec_0[4] = (buff_input_feature[batch_cnt*INPUT_WH+row+row_f][col+4] * buff_conv_kernel[depth_out][row_f*CONV_1_WH+4])>>QUANT_SIZE;

							vec_0[col_f] = (buff_input_feature[batch_cnt*INPUT_WH+row+row_f][col+col_f] * buff_conv_kernel[depth_out][row_f*CONV_1_WH+col_f])>>QUANT_SIZE;;
						}
						temp[row_f] = vec_0[0] + vec_0[1] + vec_0[2] + vec_0[3] + vec_0[4];
					}
					fvalue = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
					output_feature[batch_cnt * CONV_1_OUTPUT_WH * CONV_1_TYPE + CONV_1_OUTPUT_WH + depth_out * CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH + CONV_1_OUTPUT_WH * row + col] = (fvalue+buff_conv_bias[depth_out]);
				}
			}
		}
	}
}

	/*
	// Convolution
	Lv_0:for(batch_cnt=0; batch_cnt<2;batch_cnt++) {
		Lv_1:for (depth_out = 0; depth_out < CONV_1_TYPE; depth_out++) {
			Lv_2:for (row = 0; row < CONV_1_OUTPUT_WH; row++) {
				Lv_3:for (col = 0; col < CONV_1_OUTPUT_WH; col++) {
					// Initialization of Reduction-Sum
					temp=0;

					// Multiplication by Convolution and Input feature map
					Lv_4:for (row_f = 0; row_f < CONV_1_WH; row_f++) {

						Lv_5:for (col_f = 0; col_f < CONV_1_WH; col_f++){
							temp += (input_feature[batch_cnt*32*32+INPUT_WH * (row + row_f) + col + col_f] * conv_kernel[depth_out * CONV_1_WH * CONV_1_WH + CONV_1_WH * row_f + col_f])>>QUANT_SIZE;
						}
						//temp[row_f] = vec_0[0] + vec_0[1] + vec_0[2] + vec_0[3] + vec_0[4];
					}
					//fvalue = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
					output_feature[batch_cnt * CONV_1_OUTPUT_WH * CONV_1_TYPE + CONV_1_OUTPUT_WH + depth_out * CONV_1_OUTPUT_WH * CONV_1_OUTPUT_WH + CONV_1_OUTPUT_WH * row + col] = (temp+conv_bias[depth_out]);
				}
			}
		}
	}
}
	*/


// Convolution Layer 2
// Function by Batch_size(10)
// Input_feature_map[6][14x14],  Conv_kernel[16][6][25], Bias[16], Output_feature_map[16][10x10]
void CONVOLUTION_LAYER_2(short input_feature[CONV_1_TYPE * CONV_2_INPUT_WH *CONV_2_INPUT_WH*10],
						 short conv_kernel[CONV_2_TYPE * CONV_1_TYPE * CONV_2_WH * CONV_2_WH],
						 short conv_bias[CONV_2_TYPE],
						 short output_feature[CONV_2_TYPE * CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH*10])
{
	// Connection Table for Dummy Operation
	/*
	3 Input feature map (6)
	----------------------------------------
	{ 1, 2, 3, 0, 0, 0 }, // 1,2 + 3 --> 2,3
	{ 0, 2, 3, 4, 0, 0 }, // 2,3 + 4 --> 3,4
	{ 0, 0, 3, 4, 5, 0 }, // 3,4 + 5 --> 4,5 V
	{ 0, 0, 0, 4, 5, 6 }, // 4,5 + 6 --> 5,6 V
	{ 1, 0, 0, 0, 5, 6 }, // 5,6 + 1 --> 6,1
	{ 1, 2, 0, 0, 0, 6 }, // 6,1 + 2

	4 Input feature map (9)
	----------------------------------------
	{ 1, 2, 3, 4, 0, 0 }, // 1,2,3 + 4
	{ 0, 2, 3, 4, 5, 0 }, // 2,3,4 + 5
	{ 0, 0, 3, 4, 5, 6 }, // 3,4,5 + 6
	{ 1, 0, 0, 4, 5, 6 }, // 4,5,6 + 1
	{ 1, 2, 0, 0, 5, 6 }, // 5,6,1 + 2
	{ 1, 2, 3, 0, 0, 6 }, // 6,1,2 + 3
	{ 1, 2, 0, 4, 5, 0 }, // 1,4 + 2,5
	{ 0, 2, 3, 0, 5, 6 }, // 2,5 + 3,6
	{ 1, 0, 3, 4, 0, 6 }, // 3,6 + 4,1

	6 Input feature map (1)
	----------------------------------------
	{ 1, 2, 3, 4, 5, 6 }  // 4,1 + 5,2
	*/

	short col, row;
	short col_f, row_f;
	short depth_in, depth_out, batch_cnt;

	short temp[5];
	short fvalue[16];
	short Final;

	int vec_0[5];

	short buff_input_feature[CONV_1_TYPE*10][CONV_2_INPUT_WH][CONV_2_INPUT_WH];
	//short buff_input_feature[CONV_1_TYPE*10][CONV_2_INPUT_WH][CONV_2_INPUT_WH];
	//#pragma HLS ARRAY_PARTITION variable=buff_input_feature cyclic factor=10 dim=1

	short buff_conv_kernel[CONV_2_TYPE][CONV_1_TYPE][CONV_2_WH*CONV_2_WH];
	//#pragma HLS ARRAY_PARTITION variable=buff_conv_kernel cyclic factor=16 dim=1

	short buff_conv_bias[CONV_2_TYPE];
	//#pragma HLS ARRAY_PARTITION variable=buff_conv_bias cyclic factor=16 dim=1

	Feature_Buff:	
	for(batch_cnt=0;batch_cnt<10;batch_cnt++) {
		for(depth_in=0;depth_in<CONV_1_TYPE;depth_in++) {
			#pragma HLS PIPELINE II=1
			for(row=0;row<CONV_2_INPUT_WH;row++) {
				for(col=0;col<CONV_2_INPUT_WH;col++)
				{
					buff_input_feature[batch_cnt*CONV_1_TYPE+depth_out][row][col] = input_feature[batch_cnt * CONV_2_INPUT_SIZE * CONV_1_TYPE + depth_in * CONV_2_INPUT_SIZE + row * CONV_1_WH + col];
					//buff_input_feature[batch_cnt*CONV_1_TYPE+depth_out][row*CONV_2_INPUT_WH+col] = input_feature[batch_cnt * CONV_2_INPUT_SIZE * CONV_1_TYPE + depth_in * CONV_2_INPUT_SIZE + row * CONV_1_WH + col];
				}
			}
		}	
	}
	
	/*
	Conv_Buff:	
	for(depth_out=0;depth_out<CONV_2_TYPE;depth_out++) {
		#pragma HLS PIPELINE II=1
		for(depth_in=0;depth_in<CONV_1_TYPE;depth_in++)
		{
			for(row=0;row<CONV_2_WH*CONV_2_WH;row++)
			{
				buff_conv_kernel[depth_out][depth_in][row] = conv_kernel[depth_out * 25 * 6 + depth_in * 25 + (row/5) * CONV_1_WH + (row%5)];
			}
		}
		buff_conv_bias[depth_out] = conv_bias[depth_out];
	}	
	*/
	
	// Convolution
	Lv_0:for(batch_cnt=0;batch_cnt<10;batch_cnt++) {
		Lv_1:for (depth_out = 0; depth_out < CONV_2_TYPE; depth_out++) {
			Lv_2:for (row = 0; row < CONV_2_OUTPUT_WH; row++) {
				Lv_3:for (col = 0; col < CONV_2_OUTPUT_WH; col++) {

					// Multiplication by Convolution and Input feature maps
					Lv_4:for (depth_in = 0; depth_in < CONV_1_TYPE; depth_in++) {
#pragma HLS PIPELINE II=1
						Lv_5:for (row_f = 0; row_f < CONV_2_WH; row_f++)
						{

							Lv_6:for (col_f = 0; col_f < CONV_2_WH; col_f++)
							{
								vec_0[col_f] = (buff_input_feature[batch_cnt*CONV_1_TYPE+depth_out][(row+row_f)][(col+col_f)] * conv_kernel[depth_out * 25 * 6 + depth_in * 25 + (row+row_f) * CONV_1_WH + (col + col_f)])>>QUANT_SIZE;
								//vec_0[col_f] = (buff_input_feature[batch_cnt*CONV_1_TYPE+depth_out][(row+row_f)][(col+col_f)] * buff_conv_kernel[depth_out][depth_in][row_f*5+col_f])>>QUANT_SIZE;
							}
							temp[row_f] = vec_0[0] + vec_0[1] + vec_0[2] + vec_0[3] + vec_0[4];
						}
						fvalue[depth_in] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
					}
					Final=fvalue[0]+fvalue[1]+fvalue[2]+fvalue[3]+fvalue[4]+fvalue[5]+fvalue[6]+fvalue[7]+fvalue[8]+
							fvalue[9]+fvalue[10]+fvalue[11]+fvalue[12]+fvalue[13]+fvalue[14]+fvalue[15];
					// Result of Convolution
					output_feature[batch_cnt * CONV_2_OUTPUT_WH * CONV_2_OUTPUT_WH * CONV_2_TYPE + depth_out*CONV_2_OUTPUT_WH + CONV_2_OUTPUT_WH * row + col] = (Final+conv_bias[depth_out]);
				}
			}
		}
	}
}

// Convolution Layer 3 (FC)
// Function by Batch_size(10)
// Input_feature_map[16][5x5],  Conv_kernel[120][16][5x5], Bias[120], Output_feature_map[120][1x1]
void CONVOLUTION_LAYER_3(short input_feature[CONV_2_TYPE*CONV_3_INPUT_WH *CONV_3_INPUT_WH],
						 short conv_kernel[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH], short conv_bias[CONV_3_TYPE],
						 short output_feature[CONV_3_TYPE])
{
	int col, row, col_f, row_f;
	int depth_in, batch_cnt=0, depth_out;

	short temp=0;

	for (depth_out = 0; depth_out < CONV_3_TYPE; depth_out++) {
		// Init
		temp = 0;

		// Multiplication by Convolution and Input feature maps
		for (depth_in = 0; depth_in < POOL_2_TYPE; depth_in++) {
			for (row_f = 0; row_f < CONV_3_WH; row_f++) {
				for (col_f = 0; col_f < CONV_3_WH; col_f++) {
					temp += (input_feature[depth_in * CONV_3_WH * CONV_3_WH + CONV_3_WH * row_f + col_f]
					  * conv_kernel[depth_out * CONV_3_WH * CONV_3_WH * POOL_2_TYPE+ depth_in * CONV_3_WH * CONV_3_WH + CONV_3_WH * row_f + col_f])>>QUANT_SIZE;;
				}
			}
		}
		// Result of Convolution
		output_feature[batch_cnt * CONV_3_TYPE + depth_out] = (temp+conv_bias[depth_out]);
	}
}

void CONVOLUTION_LAYER_3_Sub(short input_feature[CONV_2_TYPE*CONV_3_INPUT_WH *CONV_3_INPUT_WH],
							 short conv_kernel[CONV_3_TYPE*CONV_2_TYPE*CONV_3_WH * CONV_3_WH/3], short conv_bias[CONV_3_TYPE/3],
							 short output_feature[CONV_3_TYPE/3])
{
	int col, row, col_f, row_f;
	int depth_in, batch_cnt=0, depth_out;

	short temp[5];
	short fvalue;
	int vec_0[5];

	L5:for (depth_out = 0; depth_out < CONV_3_TYPE/3; depth_out++) {
		#pragma HLS PIPELINE II=1
		// Init
		fvalue = 0;

		// Multiplication by Convolution and Input feature maps
		L2:for (depth_in = 0; depth_in < POOL_2_TYPE; depth_in++) {
			L3:for (row_f = 0; row_f < CONV_3_WH; row_f++) {
				L4:for (col_f = 0; col_f < CONV_3_WH; col_f++) {
					//vec_0[col_f] = buff_input_feature[depth_in][row_f*5+col_f]
					  //* buff_conv_kernel[depth_out][depth_in][CONV_3_WH * row_f + col_f];
					
					vec_0[col_f] = (input_feature[depth_in * CONV_3_WH * CONV_3_WH + CONV_3_WH * row_f + col_f]
					  * conv_kernel[depth_out * CONV_3_WH * CONV_3_WH * POOL_2_TYPE + depth_in * CONV_3_WH * CONV_3_WH + CONV_3_WH * row_f + col_f])>>QUANT_SIZE;;
				}
				temp[row_f] = vec_0[0] + vec_0[1] + vec_0[2] + vec_0[3] + vec_0[4];
			}
		}
		fvalue += temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
		// Result of Convolution
		output_feature[batch_cnt * CONV_3_TYPE + depth_out] = (fvalue+conv_bias[depth_out]);
	}							 
}
