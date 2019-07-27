#include "Image_pre.h"
#include "image_feature.h"

int main()
{
	unsigned long long clock_start, clock_end;
	int loop_idx;

	printf("---------------------------------------------------------\n");
	printf("Acclerationg Lenet-5 (Base Version) for Default\n");
	printf("Implementation by Constant Park\n");
	printf("Hanyang university, ESoCLab [Version 1.1]\n");
	printf("---------------------------------------------------------\n");
	printf("Image Batch(%d), Total Image(%d)\n", image_Batch, image_Move);

	// Debug Log File Pointer Init
	// ----------------------------------------------------------------------
	#ifdef LOG_PRINT 
		MNIST_ORG = fopen("MNIST_ORG.txt", "wb");
		MNIST_PAD = fopen("MNIST_PAD.txt", "wb");

		RESULT_CONV_1 = fopen("RESULT_CONV_1.txt", "wb");
		RESULT_POOL_1 = fopen("RESULT_POOL_1.txt", "wb");
		RESULT_CONV_2 = fopen("RESULT_CONV_2.txt", "wb");
		RESULT_POOL_2 = fopen("RESULT_POOL_2.txt", "wb");
		RESULT_CONV_3 = fopen("RESULT_CONV_3.txt", "wb");

		RESULT_FC_1 = fopen("RESULT_FC_1.txt", "wb");
		RESULT_FC_2 = fopen("RESULT_FC_2.txt", "wb");
	#endif 	
	
	// MNIST Image and Label
	// ----------------------------------------------------------------------
	//unsigned char *IMAGE_BUFF = malloc(MNIST_SIZE * 10000 * sizeof(unsigned char));
	//unsigned char *IMAGE_LABEL = malloc(10000 * sizeof(unsigned char));
	unsigned char *IMAGE_BUFF = sds_alloc(MNIST_SIZE * image_Move * sizeof(unsigned char));
	//unsigned char *IMAGE_LABEL = sds_alloc(image_Move * sizeof(unsigned char));
	
	READ_MNIST("test/image.bin", IMAGE_BUFF);
	//READ_MNIST_LABEl("test/label.bin", IMAGE_LABEL);

	// Input(Feature extraction) and NN_Input(Classify)
	// ----------------------------------------------------------------------
	float *IMAGE = sds_alloc(INPUT_SIZE * image_Move * sizeof(float));
	short *Q_IMAGE = sds_alloc(INPUT_SIZE * image_Move* sizeof(short));
	short *NN_INPUT = sds_alloc(NN_INPUT_N * image_Batch * sizeof(short));

	// Boundary Padding
	// ----------------------------------------------------------------------
	for (int i = 0; i < INPUT_SIZE * image_Move; i++)
	{
		IMAGE[i] = 0;
	}

	// Feature extraction parameter
	// ----------------------------------------------------------------------
	// Convolution Layer
	float CONV_1_FILTER[6*25];
	float CONV_1_BIAS[6];

	float CONV_2_FILTER[16*6*25];
	float CONV_2_BIAS[16];

	float CONV_3_FILTER[120*16*25];
	float CONV_3_BIAS[120];

	// Pool(Sub-sampling) Layer
	float POOL_1_FILTER[6*4];
	float POOL_1_BIAS[6];

	float POOL_2_FILTER[16*4];
	float POOL_2_BIAS[16];


	// Classification parameter
	// ----------------------------------------------------------------------

	// Parameter load
	// ----------------------------------------------------------------------
	kernel_load("filter/LeNet-weights_Conv_1.txt", CONV_1_FILTER);
	//kernel_load("filter/LeNet-weights_Conv_2.txt", CONV_2_FILTER);

	kernel_load("filter/LeNet-weights_Conv_2_Dummy.txt", CONV_2_FILTER);
	kernel_load("filter/LeNet-weights_Conv_3.txt", CONV_3_FILTER);

	kernel_load("filter/LeNet-weights_Conv_1_Bias.txt", CONV_1_BIAS);
	kernel_load("filter/LeNet-weights_Conv_2_Bias.txt", CONV_2_BIAS);
	kernel_load("filter/LeNet-weights_Conv_3_Bias.txt", CONV_3_BIAS);

	kernel_load("filter/LeNet-weights_Pool_1.txt", POOL_1_FILTER);
	kernel_load("filter/LeNet-weights_Pool_2.txt", POOL_2_FILTER);

	kernel_load("filter/LeNet-weights_Pool_1_Bias.txt", POOL_1_BIAS);
	kernel_load("filter/LeNet-weights_Pool_2_Bias.txt", POOL_2_BIAS);

	// Convolution Layer Q
	short CONV_1_FILTER_Q[6*25];
	short CONV_1_BIAS_Q[6];

	short CONV_2_FILTER_Q[16*6*25];
	short CONV_2_BIAS_Q[16];

	short CONV_3_FILTER_Q[120*16*25];
	short CONV_3_BIAS_Q[120];

	// Pool(Sub-sampling) Layer
	short POOL_1_FILTER_Q[6*4];
	short POOL_1_BIAS_Q[6];

	short POOL_2_FILTER_Q[16*4];
	short POOL_2_BIAS_Q[16];


	// Quantization
	// ----------------------------------------------------------------------

	TransQ(CONV_1_FILTER, CONV_1_FILTER_Q, 150);
	//PrintQ(Q_CONV_1_FILTER, 150);
	TransQ(CONV_2_FILTER, CONV_3_FILTER_Q, 25*6*16);
	//PrintQ(Q_CONV_2_FILTER, 150);
	TransQ(CONV_3_FILTER, CONV_3_FILTER_Q, 25*16*120);
	//PrintQ(Q_CONV_3_FILTER, 150);

	TransQ(CONV_1_BIAS, CONV_1_BIAS_Q, 6);
	//PrintQ(Q_CONV_1_BIAS, 6);
	TransQ(CONV_2_BIAS, CONV_2_BIAS_Q, 16);
	//PrintQ(Q_CONV_2_BIAS, 16);
	TransQ(CONV_3_BIAS, CONV_3_BIAS_Q, 120);
	//PrintQ(Q_CONV_3_BIAS, 120);


	TransQ(POOL_1_FILTER, POOL_1_FILTER_Q, 24);
	//PrintQ(Q_CONV_1_FILTER, 24);
	TransQ(POOL_2_FILTER, POOL_2_FILTER_Q, 64);
	//PrintQ(Q_CONV_1_FILTER, 64);
	TransQ(POOL_1_BIAS, POOL_1_BIAS_Q, 6);
	//PrintQ(Q_CONV_1_FILTER, 6);
	TransQ(POOL_2_BIAS, POOL_2_BIAS_Q, 16);
	//PrintQ(Q_POOL_2_BIAS, 16);

	IMAGE_INIT(IMAGE_BUFF, IMAGE, 0);
	TransQ(IMAGE, Q_IMAGE, INPUT_SIZE*image_Move);



	// Program start
	// ----------------------------------------------------------------------

	clock_start = sds_clock_counter();

	//for (loop_idx = 0; loop_idx<10; loop_idx++)
	for (loop_idx = 0; loop_idx<image_Move/image_Batch; loop_idx++)
	{
		printf("Processing Count : %d \n", loop_idx);
		// Padding
		// Data Quantaty is decided by image batch
		//IMAGE_INIT(IMAGE_BUFF, IMAGE, loop_idx);

		// Feature extraction (Convolution + Sub-sampling)
		FEATURE_EXTRACTION(&Q_IMAGE[32*32*image_Batch*loop_idx], CONV_1_FILTER_Q, CONV_1_BIAS_Q, POOL_1_FILTER_Q, POOL_1_BIAS_Q,
			CONV_2_FILTER_Q, CONV_2_BIAS_Q, POOL_2_FILTER_Q, POOL_2_BIAS_Q, CONV_3_FILTER_Q,  CONV_3_BIAS_Q, NN_INPUT);

		// Fully connected layer
		//CLASSIFY(NN_INPUT, NN_WEIGHT_1, NN_WEIGHT_1_Bias, NN_WEIGHT_2, NN_WEIGHT_2_Bias, &IMAGE_LABEL[image_Batch * loop_idx]);
	}

	clock_end = sds_clock_counter();
	printf("=============================================\n");
	printf("Total Clock Cycle is %llu \n", (clock_end - clock_start) );
	printf("=============================================\n");

	#ifdef LOG_PRINT
		int z, x, y;

		for (z = 0; z < 6; z++)
		{
			for (x = 0; x < 32; x++)
			{
					for (y = 0; y < 32; y++)
					{
						fprintf(MNIST_PAD, "%.9f ",	IMAGE[z*32*32+32 * x + y]);
					}
					fprintf(MNIST_PAD, "\n");
			}
			fprintf(MNIST_PAD, "\n");
		}
	#endif

	// Parameters free
	// ----------------------------------------------------------------------
	//free(IMAGE_BUFF);
	//free(IMAGE_LABEL);
	
	//free(IMAGE);
	//free(NN_INPUT);
	sds_free(IMAGE_BUFF);
	//sds_free(IMAGE_LABEL);
	
	sds_free(IMAGE);
	sds_free(NN_INPUT);
	sds_free(Q_IMAGE);
	// Debug Log File Pointer Close
	// ----------------------------------------------------------------------
	#ifdef LOG_PRINT 
		fclose(MNIST_ORG);
		fclose(MNIST_PAD);

		fclose(RESULT_CONV_1);
		fclose(RESULT_POOL_1);
		fclose(RESULT_CONV_2);
		fclose(RESULT_POOL_2);
		fclose(RESULT_CONV_3);

		fclose(RESULT_FC_1);
		fclose(RESULT_FC_2);
	#endif

	return 0;
}
