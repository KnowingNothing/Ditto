// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "gemmini.h"
#include "conv_operator.h"

#define MAX_INPUT_H           250
#define MAX_INPUT_W  		  250
#define FIXED_INOUT_CHANNEL   16       //fpga vcs fixed pe bit
#define FIXED_INOUT_BATCH     16

#define MAX_FILTER_H          5
#define MAX_FILTER_W          5

#define read_csr(reg) ({ unsigned long __tmp; \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })

#define read_cycle() read_csr(mcycle)

#define SYSCTL_CLOCK_CPU 20000000
static int cycle2Ms(uint64_t cycle)
{
    return (cycle / (SYSCTL_CLOCK_CPU / 1000UL));
}

/****************************************/ 
/*Function: 开根号处理                 */ 
/*入口参数：被开方数，长整型            */ 
/*出口参数：开方结果，整型              */ 
/****************************************/ 
uint32_t  KitFastSqrt (uint32_t val) 
{ 
	uint32_t bak_val = val;
 	uint32_t n;
 	uint32_t i; 
    uint32_t tmp, ttp;      // 结果、循环计数 

    if (val == 0) return 0;       // 被开方数，开方结果也为0 
 	n = 0;
    tmp = (val >> 30);           // 获取最高位：B[m-1] 
    val <<= 2; 
    if (tmp > 1) {               // 最高位为1     
        n++;                   // 结果当前位为1，否则为默认的0 
        tmp -= n; 
    }
    for (i=15; i>0; i--) {       // 求剩余的15位     
        n <<= 1;                // 左移一位
        tmp <<= 2; 
        tmp += (val >> 30);      // 假设 
        ttp = n; 
        ttp = (ttp << 1) + 1;
        val <<= 2; 
        if (tmp >= ttp) {         // 假设成立        
            tmp -= ttp; 
            n++; 
        }
    }
    printf("sqrt bak_val %d is n %d\r\n",bak_val,n);
    return n; 
}

/* 从输出第H行获取输入对应H行 */
static inline int GetInHFromOutH(int outH, int stride, int pad)
{
	int inputH = outH * stride - pad;
	if(inputH < 0) inputH = 0;
	return inputH;
}

static int OffsetNHWC(Shape *shape, int i0, int i1, int i2, int i3)
{
	if (i0 >= shape->batch || i1 >= shape->height || i2 >= shape->width || i3 >= shape->channel)
	{
		printf("OffsetNHWC out of range %d,%d,%d,%d:%d,%d,%d,%d\r\n", i0, i1, i2, i3, shape->batch, shape->height, shape->width, shape->channel);
		return -1;
	}
	return ((i0 * shape->height + i1) * shape->width + i2) * shape->channel + i3;
}


static int16_t (*padding_input_data_buffer)[804][FIXED_INOUT_CHANNEL][FIXED_INOUT_BATCH] = (int16_t (*)[804][FIXED_INOUT_CHANNEL][FIXED_INOUT_BATCH])0x85000000;
static int16_t slice_out_data_buffer[MAX_INPUT_H][MAX_INPUT_W][FIXED_INOUT_CHANNEL][FIXED_INOUT_BATCH] row_align(1) = {0};
static int32_t (*filter_data_tranf_t)[MAX_FILTER_H * MAX_FILTER_W][FIXED_INOUT_CHANNEL][FIXED_INOUT_CHANNEL] = (int32_t (*)[MAX_FILTER_H * MAX_FILTER_W][FIXED_INOUT_CHANNEL][FIXED_INOUT_CHANNEL])0x9c000000;


void pre_convert_data(GemConvParams *params, const void* input_data, const void* filter_data, MulBitType bit_type, int slice_Ms, int slice_Cs)
{
  const int32_t input_height = params->input_height;
  const int32_t input_width = params->input_width;
  const int32_t input_depth = params->input_depth;
  const int32_t filter_height = params->filter_height;
  const int32_t filter_width = params->filter_width;
  const int32_t output_depth = params->output_depth;

  switch (bit_type)
  {
    case BIT4:
      for (size_t h = 0; h < input_height; ++h)
	    for(size_t input_slice_C = 0; input_slice_C < slice_Cs; input_slice_C++)
          for (size_t w = 0; w < input_width; ++w) {
            for (size_t c = 0; c < FIXED_INOUT_CHANNEL; ++c) {
			  int c1 = c * 4 + input_slice_C * FIXED_INOUT_CHANNEL * 4;
			  padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] = 0;
	    	  if(c1 < input_depth) {
	    	    padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] = *((uint8_t *)input_data+(h * input_width + w) * input_depth + c1) & 0x0F;
	    	  }
	    	  if(c1 + 1 < input_depth) {
	    	    padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] += (uint16_t)(*((uint8_t *)input_data+(h * input_width + w) * input_depth + c1 + 1) << 4 & 0x00F0);
	    	  }
	    	  if(c1 + 2 < input_depth) {
	    	    padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] += (uint16_t)(*((uint8_t *)input_data+(h * input_width + w) * input_depth + c1 + 2) << 8 & 0x0F00);
	    	  }
	    	  if(c1 + 3 < input_depth) {
	    	    padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] += (uint16_t)(*((uint8_t *)input_data+(h * input_width + w) * input_depth + c1 + 3) << 12 & 0xF000);
	    	  }
	        }
		  }
      
      for(int filter_slice_M = 0; filter_slice_M < slice_Ms; filter_slice_M++)
      {
	    for(int filter_slice_C = 0; filter_slice_C < slice_Cs; filter_slice_C++)
        {
          for (size_t h = 0; h < filter_height; ++h)
            for (size_t w = 0; w < filter_width; ++w)
              for (size_t c = 0; c < FIXED_INOUT_CHANNEL; ++c) {
                for (size_t m = 0; m < FIXED_INOUT_CHANNEL; ++m) {
	    		  int m1 = m * 2 + filter_slice_M * FIXED_INOUT_CHANNEL * 2;
	    		  int c1 = c * 4 + filter_slice_C * FIXED_INOUT_CHANNEL * 4;
				  filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] = 0;
	    	  	  if(m1 < output_depth && c1 < input_depth) {
	    			filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] = *((uint8_t *)filter_data+((m1 * filter_height + h) * filter_width + w) * input_depth + c1) & 0x0F;
	    	  	  }
	    	  	  if(m1 < output_depth && c1 + 1 < input_depth) {
	    			filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data + ((m1 * filter_height + h) * filter_width + w) * input_depth + c1 + 1) << 4 & 0x000000F0);
	    	  	  }
	    	  	  if(m1 < output_depth && c1 + 2 < input_depth) {
	    			filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data + ((m1 * filter_height + h) * filter_width + w) * input_depth + c1 + 2) << 8 & 0x00000F00);
	    	  	  }
	    	  	  if(m1 < output_depth && c1 + 3 < input_depth) {
	    			filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data + ((m1 * filter_height + h) * filter_width + w) * input_depth + c1 + 3) << 12 & 0x0000F000);
	    	  	  }														
	    		  if(m1 + 1 < output_depth && c1 < input_depth) {
	    		    filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data+(((m1 + 1) * filter_height + h) * filter_width + w) * input_depth + c1) << 16 & 0x000F0000);
	    		  }
	    		  if(m1 + 1 < output_depth && c1 + 1 < input_depth) {
	    		    filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data+(((m1 + 1) * filter_height + h) * filter_width + w) * input_depth + c1 + 1) << 20 & 0x00F00000);
	    		  }
	    		  if(m1 + 1 < output_depth && c1 + 2 < input_depth) {
	    		    filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data+(((m1 + 1) * filter_height + h) * filter_width + w) * input_depth + c1 + 2) << 24 & 0x0F000000);
	    		  }				  				  
	    		  if(m1 + 1 < output_depth && c1 + 3 < input_depth) {
	    		    filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data+(((m1 + 1) * filter_height + h) * filter_width + w) * input_depth + c1 + 3) << 28 & 0xF0000000);
	    		  }
                }
			  }
	    }
      }
      break;
    case BIT8: 
	  //printf("input data:\r\n{\r\n");
      for (size_t h = 0; h < input_height; ++h){
	    for(size_t input_slice_C = 0; input_slice_C < slice_Cs; input_slice_C++)
          for (size_t w = 0; w < input_width; ++w) {
            for (size_t c = 0; c < FIXED_INOUT_CHANNEL; ++c) {
	    	  int c1 = c * 2 + input_slice_C * FIXED_INOUT_CHANNEL * 2;
	    	  padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] = 0;
	    	  if(c1 < input_depth) {
	    	    padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] = *((uint8_t *)input_data+(h * input_width + w) * input_depth + c1);
	    	  }
	    	  if(c1 + 1 < input_depth) {
	    	    padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] += (uint16_t)(*((uint8_t *)input_data+(h * input_width + w) * input_depth + c1 + 1) << 8);
	    	  }
			  //printf("0x%x,",padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0]);		
	        }
			//printf("\r\n");
		  }
      }

	  //printf("}\r\n\r\n\r\nfilter data:\r\n{\r\n");
      for(int filter_slice_M = 0; filter_slice_M < slice_Ms; filter_slice_M++)
      {
	    for(int filter_slice_C = 0; filter_slice_C < slice_Cs; filter_slice_C++)
        {
          for (size_t h = 0; h < filter_height; ++h)
            for (size_t w = 0; w < filter_width; ++w)
              for (size_t c = 0; c < FIXED_INOUT_CHANNEL; ++c) {
                for (size_t m = 0; m < FIXED_INOUT_CHANNEL; ++m) {
	    		  int m1 = m * 2 + filter_slice_M * FIXED_INOUT_CHANNEL * 2;
	    		  int c1 = c * 2 + filter_slice_C * FIXED_INOUT_CHANNEL * 2;
	    		  filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] = 0;			  
	    	  	  if(m1 < output_depth && c1 < input_depth) {
	    			filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] = *((uint8_t *)filter_data+((m1 * filter_height + h) * filter_width + w) * input_depth + c1);
	    	  	  }
	    	  	  if(m1 < output_depth && c1 + 1 < input_depth) {
	    			filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data + ((m1 * filter_height + h) * filter_width + w) * input_depth + c1 + 1) << 8);
	    	  	  }				
	    		  if(m1 + 1 < output_depth && c1 < input_depth) {
	    		    filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data+(((m1 + 1) * filter_height + h) * filter_width + w) * input_depth + c1) << 16);
	    		  }
	    		  if(m1 + 1 < output_depth && c1 + 1 < input_depth) {
	    		    filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] += (uint32_t)(*((uint8_t *)filter_data+(((m1 + 1) * filter_height + h) * filter_width + w) * input_depth + c1 + 1) << 24);
	    		  }
				  //printf("0x%x,",filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m]);	 				 				 
                }
				//printf("\r\n");
			  }
	    }
      }
	  //printf("}\r\n\r\n\r\n");  	  
      break;
    case BIT16:
      for (size_t h = 0; h < input_height; ++h){
	    for(size_t input_slice_C = 0; input_slice_C < slice_Cs; input_slice_C++)
          for (size_t w = 0; w < input_width; ++w)
            for (size_t c = 0; c < FIXED_INOUT_CHANNEL; ++c)
	        {
	    		int c1 = c + input_slice_C * FIXED_INOUT_CHANNEL;
				padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] = 0;
	    		if(c1 < input_depth) {
	    			padding_input_data_buffer[h * slice_Cs + input_slice_C][w][c][0] = *((uint16_t *)input_data + (h * input_width + w) * input_depth + c1);
	    		}				
	        }
	  }

      for(int filter_slice_M = 0; filter_slice_M < slice_Ms; filter_slice_M++) {
	    for(int filter_slice_C = 0; filter_slice_C < slice_Cs; filter_slice_C++) {
          for (size_t h = 0; h < filter_height; ++h)
            for (size_t w = 0; w < filter_width; ++w)
              for (size_t c = 0; c < FIXED_INOUT_CHANNEL; ++c)
                for (size_t m = 0; m < FIXED_INOUT_CHANNEL; ++m) {
	    		 	int m1 = m + filter_slice_M * FIXED_INOUT_CHANNEL;
	    		 	int c1 = c + filter_slice_C * FIXED_INOUT_CHANNEL;
					filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] = 0;
	    			if(m1 < output_depth && c1 < input_depth) {				
	    		    	filter_data_tranf_t[filter_slice_M * slice_Cs + filter_slice_C][h*filter_width+w][c][m] = *((uint16_t *)filter_data + ((m1 * filter_height + h) * filter_width + w) * input_depth + c1);
	    			}			
	    		}
        }
      }
      break;
  }  
}

void post_convert_data(GemConvParams *params, int out_heights, int out_start_height, int out_start_channel, MulBitType bit_type, void* output_data, int out_start_width, int width)
{
  int32_t output_depth = params->output_depth;
  Shape   *outputShape = &params->outputShape;
  
  switch (bit_type)
  {
    case BIT4:
    case BIT8: 
	  for (int h=0; h<out_heights; h++)
	  {
	    for(int w=0;w<width;w++)
	    {
	      for(int c=0;c<FIXED_INOUT_CHANNEL;c++)
	      {	
			int out_channel = (c + out_start_channel) * 2;
			if(out_channel < output_depth)
				*((int8_t*)output_data + OffsetNHWC(outputShape, 0, h + out_start_height, w + out_start_width, out_channel)) = (int8_t)slice_out_data_buffer[h][w][c][0];
			out_channel += 1;
			if(out_channel < output_depth)
     			*((int8_t*)output_data + OffsetNHWC(outputShape, 0, h + out_start_height, w + out_start_width, out_channel)) = (int8_t)(slice_out_data_buffer[h][w][c][0] >> 8);	  
	      }
	    }		  
	  }
      break;
    case BIT16:
	  for (int h=0; h<out_heights; h++)
  	  {
        for(int w=0;w<width;w++)
        {
          for(int c=0;c<FIXED_INOUT_CHANNEL;c++)
          {	
		    int out_channel = c + out_start_channel;
			if(out_channel < output_depth)
		    	*((int16_t*)output_data + OffsetNHWC(outputShape, 0, h + out_start_height, w + out_start_width, out_channel)) = slice_out_data_buffer[h][w][c][0];
		  }					
        }
      }
      break;
  }  
}
#define SET_SNICE //定义后会切割拼接
#define SET_DMA //定义后会DMA操作

void Conv_Xbit(GemConvParams *params, const void* input_data, const void* filter_data, void* output_data, MulBitType bit_type)
{ 
  uint64_t cycle_start = read_cycle();
  const int32_t pad_width = params->pad_width;
  const int32_t pad_height = params->pad_height;
  const int16_t stride_width = params->stride_width;
  const int16_t stride_height = params->stride_height;
  const int32_t input_offset = params->input_offset;
  const int32_t output_offset = params->output_offset;
  const int64_t* bias_data = params->bias_data;
  const int32_t* output_multiplier = params->output_multiplier;
  const int32_t output_multiplier2 = params->output_multiplier2;
  const int32_t input_height = params->input_height;
  const int32_t input_width = params->input_width;
  const int32_t input_depth = params->input_depth;
  const int32_t filter_height = params->filter_height;
  const int32_t filter_width = params->filter_width;
  const int32_t output_height = params->output_height;
  const int32_t output_width = params->output_width;
  const int32_t output_depth = params->output_depth;
    
  int ctrl = 1; //default 16bit
  int fold_mul_outputC = 1; //default 16bit
  int fold_mul_inputC = 1; //default 16bit
  switch (bit_type)
  {
    case BIT4: 
	  ctrl = 2;
	  fold_mul_outputC = 2;
	  fold_mul_inputC = 4;
      break;
    case BIT8: 
	  ctrl = 0;
	  fold_mul_outputC = 2;
	  fold_mul_inputC = 2;
      break;
    case BIT16:
	  ctrl = 1;
	  fold_mul_outputC = 1;
	  fold_mul_inputC = 1;
      break;
  }

  int filter_slice_Ms = (output_depth + fold_mul_outputC * FIXED_INOUT_CHANNEL - 1)  / (fold_mul_outputC * FIXED_INOUT_CHANNEL);
  int filter_slice_Cs = (input_depth + fold_mul_inputC * FIXED_INOUT_CHANNEL - 1) / (fold_mul_inputC * FIXED_INOUT_CHANNEL);
  
  int input_max_HsWs = KitFastSqrt(4096 / filter_slice_Cs);
  int output_max_HsWs = (input_max_HsWs - filter_height + 2 * pad_height) / stride_height + 1;
      
  printf("%s,%d filter_slice_Ms %d, filter_slice_Cs %d input_max_HsWs %d\r\n",__FUNCTION__, __LINE__,filter_slice_Ms,filter_slice_Cs,input_max_HsWs);
#ifdef SET_SNICE
  pre_convert_data(params, input_data, filter_data, bit_type, filter_slice_Ms, filter_slice_Cs);
#endif
  gemmini_flush(0);
  systolic_fence();

  int processed_out_widths = 0;
  while(processed_out_widths < output_width)
  {
	int output_width_t = output_max_HsWs;
	if(output_width_t > output_width - processed_out_widths) output_width_t = output_width - processed_out_widths;
	int input_width_t = 0;
	int pad_value_t = 0;
	int mapping_input_w = GetInHFromOutH(processed_out_widths, stride_width, pad_width);
	if(processed_out_widths == 0) //判断是第一个W方向分块
	{
	  pad_value_t -= pad_width;
	}
	if(processed_out_widths + output_width_t >= output_width) //判断是最后一个W方向分块
	{
	  pad_value_t -= pad_width;
	}
	input_width_t = pad_value_t + (output_width_t - 1) * stride_width + filter_width; //实际上需要输入的宽度
#define FIX_OUT_MIN_WIDTH 21 // 21 是实验得到的数值，不知道为啥？
	int out_slot_size = output_width_t<FIX_OUT_MIN_WIDTH?FIX_OUT_MIN_WIDTH:output_width_t; //如果输出宽度不够，需要拓展
	printf("........line %d output_width_t %d, input_width_t %d, processed_out_widths %d, mapping_input_w %d\r\n",__LINE__,output_width_t,input_width_t,processed_out_widths,mapping_input_w);

    systolic_config_stride(stride_width);
    systolic_config_execute(pad_width, filter_width, out_slot_size, ctrl, 1);
    int padding_left = false;
	if(processed_out_widths == 0 && pad_width > 0) padding_left = true; //判断是第一个W方向分块，左侧padding
	systolic_padding_data(padding_left, 0);
    systolic_config_slot_size(input_width_t, out_slot_size, filter_height * filter_width * FIXED_INOUT_CHANNEL);
  
    /* 考虑优先将input_c传输进去，进行tiling计算；
	   输出扩展的情况下，需要重新根据计算的输入宽度算出输入最大H */
	int mvin_input_max_heights = 4096 / (input_width_t + (out_slot_size - output_width_t) * stride_width) / filter_slice_Cs;
    if(mvin_input_max_heights > input_height) mvin_input_max_heights = input_height;
  
    //先计算中间块，不带padding的输出块H大小
    int output_heights_of_block = (mvin_input_max_heights - filter_height) / stride_height + 1;
    printf("%s %d, mvin_input_max_heights %d \r\n",__FUNCTION__,__LINE__,mvin_input_max_heights);
    printf("%s %d, output_heights_of_block %d \r\n",__FUNCTION__,__LINE__,output_heights_of_block);

    /* 此变量记录已经处理的输出行数 */
    int processed_out_heights = 0;
    while(processed_out_heights < output_height)
    {
	  /* 计算输出高度，计算输入高度 */
	  int output_height_t = output_heights_of_block;
	  if(output_height_t > output_height - processed_out_heights) output_height_t = output_height - processed_out_heights;
	  int input_height_t = 0;
	  int mapping_input_h = GetInHFromOutH(processed_out_heights, stride_height, pad_height);
	  if(processed_out_heights == 0)
	  {
	    input_height_t -= pad_height;
	  }
	  if(processed_out_heights + output_height_t >= output_height) //判断是最后一个H方向分块
	  {
	    input_height_t -= pad_height;
	  }
	  input_height_t += (output_height_t - 1) * stride_height + filter_height;
    
	  //printf("%s %d, input_height_t %d \r\n",__FUNCTION__,__LINE__,input_height_t);
	  //printf("%s %d, output_height_t %d \r\n",__FUNCTION__,__LINE__,output_height_t);
	  for(size_t input_slice_C = 0; input_slice_C < filter_slice_Cs; input_slice_C++)
	  {
        for(int in_slot_id = 0; in_slot_id < input_height_t; in_slot_id ++)
	    {
#ifdef SET_DMA	
	      systolic_mvin_input(&padding_input_data_buffer[(mapping_input_h+in_slot_id)*filter_slice_Cs+input_slice_C][mapping_input_w], in_slot_id+input_height_t*input_slice_C, 0);
#endif		  
	    }		
	  }
	  
	  for(int output_slice_channel = 0; output_slice_channel < filter_slice_Ms; output_slice_channel++)
	  {
	    int out_start_channel = output_slice_channel * FIXED_INOUT_CHANNEL;
        for(int filter_slice_C = 0; filter_slice_C < filter_slice_Cs; filter_slice_C++)
        {
#ifdef SET_DMA			
          systolic_mvin_filter(filter_data_tranf_t[output_slice_channel * filter_slice_Cs + filter_slice_C], filter_slice_C);
#endif		  
	    }	  
	    switch (bit_type)
	    {
	      int16_t multipliet1;
          int16_t multipliet2;		  
	      case BIT4:
	      case BIT8: 
	        for (size_t i=0; i<FIXED_INOUT_CHANNEL; i++){
	          systolic_postproc(0,i, (((uint64_t)(uint16_t)(bias_data[2 * i + 1 + output_slice_channel * FIXED_INOUT_CHANNEL*2]) << 16)) | 
	  	    					   ((uint64_t)(uint16_t)bias_data[2 * i + output_slice_channel * FIXED_INOUT_CHANNEL*2]));
	        
	          multipliet1 = (int16_t)(output_multiplier[2 * i + output_slice_channel * FIXED_INOUT_CHANNEL*2] );
	          multipliet2 = (int16_t)(output_multiplier[2 * i + 1 + output_slice_channel * FIXED_INOUT_CHANNEL*2] );
	          systolic_postproc(1,i, (((uint64_t)(uint16_t)(multipliet2)            << 12)) | ((uint64_t)(uint16_t)multipliet1));
	        }
	    	break;
	      case BIT16:
	  	    for (size_t i=0; i<FIXED_INOUT_CHANNEL; ++i){
	  	  	  systolic_postproc(0,i,bias_data[i+output_slice_channel*FIXED_INOUT_CHANNEL]);
	  	  	  systolic_postproc(1,i,(int16_t)((output_multiplier[i+output_slice_channel*FIXED_INOUT_CHANNEL])));
	  	    }   
	    	break;
	    } 
	    systolic_postproc_offset(output_multiplier2, output_offset);
        systolic_fence();
	    
        for (int16_t h=0; h<output_height_t; h++)
        {		  
	  	  int zeroize = pad_height;
	  	  int pad_direction = 0;
	  	  int input_slot_id;
	  	  int pad_value = 0;
	  	  if(processed_out_heights == 0)
	  	  {
	  	  	pad_value = pad_height;
	  	  	if(h < pad_height)
	  	  	{
	  	      	zeroize = h;
	  	      	pad_direction = 0;
	  	  	}
	  	  } 	
            
	  	  if(processed_out_heights + output_height_t >= output_height && h >= output_height_t - pad_height) //判断是最后一个H方向分块
	  	  {
	  	    zeroize = h - (output_height_t - pad_height);
	  	    pad_direction = 1;
	  	  }
	  	  
	  	  input_slot_id = GetInHFromOutH(h, stride_height, pad_value);
		  
	  	  //if(processed_out_heights == 0)
	  	  //printf("input_slot_id %d , pad_height %d, zeroize %d, pad_direction %d, filter_slice_Cs %d\r\n",input_slot_id,pad_height,zeroize,pad_direction,filter_slice_Cs);
	  	  systolic_conv_pre(input_slot_id, 0, 0);
	  	  systolic_conv_exec(h, 1, pad_height-zeroize, pad_direction, 0);
	  	  for(size_t input_slice_C = 1; input_slice_C < filter_slice_Cs; input_slice_C++)
	  	  {
	  	    systolic_conv_pre(input_slot_id+input_height_t*input_slice_C, 0, input_slice_C);
            systolic_conv_exec(h, 1, pad_height-zeroize, pad_direction, 1);
	  	  }
	  	  gemmini_fence();
#ifdef SET_DMA			
          systolic_mvout(slice_out_data_buffer[h], h, 1);
#endif		  
          gemmini_fence();
	  	  __asm__ __volatile__(".word 0xfc000073" : : : "memory");
        }
#ifdef SET_SNICE
	    post_convert_data(params, output_height_t, processed_out_heights, out_start_channel, bit_type, output_data, processed_out_widths, output_width_t);
#endif		
	  }
	  processed_out_heights += output_height_t;
    }
    processed_out_widths += output_width_t;
  }
  uint64_t cycle_end = read_cycle();
  printf("........line %d, %lld cycles , %d ms\r\n",__LINE__,cycle_end-cycle_start,cycle2Ms(cycle_end-cycle_start));
}

