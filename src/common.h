#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include "sds_lib.h"

// Environment Option
// ===============================================
// Debug_Log_Print Option
#define image_Move 10000
#define image_Batch 10

// 100(1310), 
#define MNIST_SIZE 28*28
#define MNIST_WH 28
#define INPUT_SIZE 32*32
#define INPUT_WH 32

#define CONV_1_OUTPUT_WH 28
#define CONV_1_TYPE 6
#define CONV_1_WH 5

#define POOL_1_INPUT_WH 28
#define POOL_1_INPUT_SIZE 28*28
#define POOL_1_OUTPUT_WH 14
#define POOL_1_OUTPUT_SIZE 14*14
#define POOL_1_TYPE 6
#define POOL_1_SIZE 4

#define CONV_2_OUTPUT_WH 10
#define CONV_2_OUTPUT_SIZE 10*10
#define CONV_2_INPUT_SIZE 14*14
#define CONV_2_INPUT_WH 14
#define CONV_2_TYPE 16
#define CONV_2_WH 5

#define POOL_2_INPUT_WH 10
#define POOL_2_OUTPUT_WH 5
#define POOL_2_TYPE 16
#define POOL_2_SIZE 4
#define POOL_2_OUTPUT_SIZE 5*5
#define POOL_2_INPUT_SIZE 10*10

#define CONV_3_OUTPUT_WH 1
#define CONV_3_INPUT_WH 5
#define CONV_3_TYPE 120
#define CONV_3_WH 5
#define CONV_3_INPUT_SIZE 5*5
#define CONV_3_OUTPUT_SIZE 1

#define NN_INPUT_N 120

#define INPUT_NN_1_SIZE 120
#define FILTER_NN_1_SIZE 120 * 84
#define BIAS_NN_1_SIZE 84

#define INPUT_NN_2_SIZE 84
#define FILTER_NN_2_SIZE 84 * 10
#define OUTPUT_NN_2_SIZE 10
#define BIAS_NN_2_SIZE 10

#define IMAGE_FILE "./train/image.txt"
#define LABEL_FILE "./train/label.txt"
int total_cnt;

#define SIZE 16
#define QUANT_SIZE 14

#define TOFIX(d,q) (short)(d*(1<<(q)))
#define TOFLT(a,q) (float)a/(1<<(q))
//#define LOG_PRINT

FILE *MNIST_ORG;
FILE *MNIST_PAD;

FILE * RESULT_CONV_1;
FILE * RESULT_POOL_1;
FILE * RESULT_CONV_2;
FILE * RESULT_POOL_2;
FILE * RESULT_CONV_3;
FILE * RESULT_FC_1;
FILE * RESULT_FC_2;

time_t start_time;
time_t end_time;

