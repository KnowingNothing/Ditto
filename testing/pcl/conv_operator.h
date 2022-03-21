#ifndef CONV_OPERATOR_H
#define CONV_OPERATOR_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct Shape_t {
  int batch;
  int height;
  int width;
  int channel;
} Shape;

typedef struct GemConvParams_t {
  int32_t batches;
  int32_t pad_width;
  int32_t pad_height;
  int16_t stride_width;
  int16_t stride_height;
  int32_t input_offset;
  int32_t output_offset;
  const int64_t *bias_data;
  const int32_t *output_multiplier;
  int32_t output_multiplier2;
  const int32_t *output_shift;
  int32_t input_height;
  int32_t input_width;
  int32_t input_depth;
  int32_t filter_height;
  int32_t filter_width;
  int32_t output_height;
  int32_t output_width;
  int32_t output_depth;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t scale;
  Shape outputShape;
} GemConvParams;

typedef enum MulBitType_t {
  BIT4,
  BIT8,
  BIT16,
} MulBitType;

void Conv_4bit(GemConvParams *params, const int8_t *input_data,
               const int8_t *filter_data, int8_t *output_data);

void Conv_8bit(GemConvParams *params, const int8_t *input_data,
               const int8_t *filter_data, int8_t *output_data);

void Conv_8bit_Layer1(GemConvParams *params, const int8_t *input_data,
                      const int8_t *filter_data, int8_t *output_data);

void Conv_16bit(GemConvParams *params, const int16_t *input_data,
                const int16_t *filter_data, int16_t *output_data);

void Conv_Xbit(GemConvParams *params, const void *input_data,
               const void *filter_data, void *output_data, MulBitType bit_type);

void Conv_16bitLayer1(GemConvParams *params, const int16_t *input_data,
                      const int8_t *filter_data, int16_t *output_data);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
