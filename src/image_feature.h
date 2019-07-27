#include "common.h"
#include "image_convolution.h"
#include "image_pool.h"

void FEATURE_EXTRACTION(short SRC[image_Batch * INPUT_SIZE],
						short CONV_1_FILTER[6*25], short CONV_1_BIAS[6],
						short POOL_1_FILTER[24], short POOL_1_BIAS[6],
						short CONV_2_FILTER[16*6*25], short CONV_2_BIAS[16],
						short POOL_2_FILTER[16*4], short POOL_2_BIAS[6],
						short CONV_3_FILTER[120*16*25], short CONV_3_BIAS[120],
						short DST[image_Batch * CONV_3_TYPE])
{
	int z, x, y;
	int batch_idx, sub_idx;
	unsigned long long inside_clock_start, inside_clock_end;

	//short *CONV_1_RESULT = malloc(image_Batch * CONV_1_TYPE * MNIST_SIZE * sizeof(short));
	//short *POOL_1_RESULT = malloc(image_Batch * CONV_1_TYPE * POOL_1_OUTPUT_SIZE * sizeof(short));
	//short *CONV_2_RESULT = malloc(image_Batch * CONV_2_TYPE * CONV_2_OUTPUT_SIZE * sizeof(short));
	//short *POOL_2_RESULT = malloc(image_Batch * CONV_2_TYPE * POOL_2_OUTPUT_SIZE * sizeof(short));
	//short *CONV_3_RESULT = malloc(image_Batch * CONV_3_TYPE * sizeof(short));
	short *CONV_1_RESULT = sds_alloc(image_Batch * CONV_1_TYPE * MNIST_SIZE * sizeof(short));
	short *POOL_1_RESULT = sds_alloc(image_Batch * CONV_1_TYPE * POOL_1_OUTPUT_SIZE * sizeof(short));
	short *CONV_2_RESULT = sds_alloc(image_Batch * CONV_2_TYPE * CONV_2_OUTPUT_SIZE * sizeof(short));
	short *POOL_2_RESULT = sds_alloc(image_Batch * CONV_2_TYPE * POOL_2_OUTPUT_SIZE * sizeof(short));
	short *CONV_3_RESULT = sds_alloc(image_Batch * CONV_3_TYPE * sizeof(short));
	
	inside_clock_start = sds_clock_counter();
	
	for(batch_idx=0;batch_idx<image_Batch/2;batch_idx++)
		CONVOLUTION_LAYER_1(&SRC[batch_idx*INPUT_SIZE*2], &CONV_1_FILTER[0], &CONV_1_BIAS[0], &CONV_1_RESULT[POOL_1_INPUT_SIZE*batch_idx*CONV_1_TYPE*2]);

	inside_clock_end = sds_clock_counter();
	
	printf("Convolution 1 Cycle is %llu \n", (inside_clock_end- inside_clock_start));
	#ifdef LOG_PRINT 
		for (z = 0; z < 6; z++)
		{
			for (x = 0; x < 28; x++)
			{
				for (y = 0; y < 28; y++)
				{
					fprintf(RESULT_CONV_1, "%.10f ", CONV_1_RESULT[z*28*28+28 * x + y]);
				}
				fprintf(RESULT_CONV_1, "\n");
			}
			fprintf(RESULT_CONV_1, "\n");
		}
	#endif 

	inside_clock_start = sds_clock_counter();

	for(batch_idx=0;batch_idx<image_Batch;batch_idx=batch_idx+2)
		//POOLING_LAYER_1(&CONV_1_RESULT[POOL_1_INPUT_SIZE*6*batch_idx], &POOL_1_FILTER[0], &POOL_1_BIAS[0], &POOL_1_RESULT[POOL_1_OUTPUT_SIZE*POOL_1_TYPE*batch_idx], 2);
		POOLING_LAYER_1(&CONV_1_RESULT[POOL_1_INPUT_SIZE*POOL_1_TYPE*batch_idx], POOL_1_FILTER, &POOL_1_BIAS[0], &POOL_1_RESULT[POOL_1_OUTPUT_SIZE*POOL_1_TYPE*batch_idx]);

	inside_clock_end = sds_clock_counter();
	
	printf("Pool 1 Cycle is %llu \n", (inside_clock_end- inside_clock_start));

	#ifdef LOG_PRINT 
		for (z = 0; z < 6; z++)
		{
			for (x = 0; x < 14; x++)
			{
				for (y = 0; y < 14; y++)
				{
					fprintf(RESULT_POOL_1, "%.9f ", POOL_1_RESULT[z*14*14+14 * x + y]);
				}
				fprintf(RESULT_POOL_1, "\n");
			}
			fprintf(RESULT_POOL_1, "\n");
		}
	#endif 

	inside_clock_start = sds_clock_counter();
	//for(batch_idx=0;batch_idx<image_Batch;batch_idx++)
	CONVOLUTION_LAYER_2(&POOL_1_RESULT[0], &CONV_2_FILTER[0], &CONV_2_BIAS[0], &CONV_2_RESULT[0]);

	inside_clock_end = sds_clock_counter();
	
	printf("Convolution 2 Cycle is %llu \n", (inside_clock_end- inside_clock_start));	
	
	#ifdef LOG_PRINT 
		for (z = 0; z < 16; z++)
		{
			for (x = 0; x < 10; x++)
			{
				for (y = 0; y < 10; y++)
				{
					fprintf(RESULT_CONV_2, "%.9f ", CONV_2_RESULT[z*10*10+10 * x + y]);
				}
				fprintf(RESULT_CONV_2, "\n");
			}
			fprintf(RESULT_CONV_2, "\n");
		}
	#endif 

	inside_clock_start = sds_clock_counter();
	
	//for(batch_idx=0;batch_idx<image_Batch/2;batch_idx=batch_idx+2)
		POOLING_LAYER_2(&CONV_2_RESULT[0], &POOL_2_FILTER[0], &POOL_2_BIAS[0], &POOL_2_RESULT[0]);

	inside_clock_end = sds_clock_counter();
	
	printf("Pool 2 Cycle is %llu \n", (inside_clock_end- inside_clock_start));

	#ifdef LOG_PRINT 
		for (z = 0; z < 16; z++)
		{
			for (x = 0; x < 5; x++)
			{
				for (y = 0; y < 5; y++)
				{
					fprintf(RESULT_POOL_2, "%.9f ", POOL_2_RESULT[z*25+5 * x + y]);
				}
				fprintf(RESULT_POOL_2, "\n");
			}
			fprintf(RESULT_POOL_2, "\n");
		}
	#endif 

	inside_clock_start = sds_clock_counter();	
	
	for(batch_idx=0;batch_idx<image_Batch;batch_idx++)
	{
		for(sub_idx=0;sub_idx<3;sub_idx++)
		{
			CONVOLUTION_LAYER_3_Sub(&POOL_2_RESULT[CONV_3_INPUT_SIZE*POOL_2_TYPE*batch_idx], &CONV_3_FILTER[40*sub_idx], &CONV_3_BIAS[40*sub_idx], &CONV_3_RESULT[sub_idx*40*batch_idx]);
		}
	}

	inside_clock_end = sds_clock_counter();
	
	printf("Convolution 3 Cycle is %llu \n", (inside_clock_end- inside_clock_start));
	
	#ifdef LOG_PRINT 
		for (z = 0; z < 1; z++)
		{
			for (x = 0; x < 120; x++)
			{
					fprintf(RESULT_CONV_3, "%.6f\n ", DST[z*120 + x]);
			}
			fprintf(RESULT_CONV_3, "\n");
			fprintf(RESULT_CONV_3, "\n");
		}
	#endif 

	//free(CONV_1_RESULT);
	//free(POOL_1_RESULT);
	//free(CONV_2_RESULT);
	//free(POOL_2_RESULT);
	//free(CONV_3_RESULT);

	sds_free(CONV_1_RESULT);
	sds_free(POOL_1_RESULT);
	sds_free(CONV_2_RESULT);
	sds_free(POOL_2_RESULT);
	sds_free(CONV_3_RESULT);	
}
