#include <stddef.h>
#include <stdio.h>

#include <turboembed_prepared.h>

#define PRINT_LAYOUT(type) \
  printf("%s %zu %zu\n", #type, sizeof(type), _Alignof(type))
#define PRINT_FIELD(type, field) \
  printf("%s.%s %zu\n", #type, #field, offsetof(type, field))

int main(void) {
  PRINT_LAYOUT(te_error);
  PRINT_FIELD(te_error, code);
  PRINT_FIELD(te_error, message);

  PRINT_LAYOUT(te_context_options);
  PRINT_FIELD(te_context_options, struct_size);
  PRINT_FIELD(te_context_options, version);
  PRINT_FIELD(te_context_options, device);
  PRINT_FIELD(te_context_options, ordinal);

  PRINT_LAYOUT(te_context_info);
  PRINT_FIELD(te_context_info, struct_size);
  PRINT_FIELD(te_context_info, version);
  PRINT_FIELD(te_context_info, device);
  PRINT_FIELD(te_context_info, ordinal);
  PRINT_FIELD(te_context_info, capabilities);
  PRINT_FIELD(te_context_info, device_name);
  PRINT_FIELD(te_context_info, runtime_version);
  PRINT_FIELD(te_context_info, driver_version);

  PRINT_LAYOUT(te_model_info);
  PRINT_FIELD(te_model_info, struct_size);
  PRINT_FIELD(te_model_info, version);
  PRINT_FIELD(te_model_info, dimension);
  PRINT_FIELD(te_model_info, vocab_size);
  PRINT_FIELD(te_model_info, max_sequence_length);
  PRINT_FIELD(te_model_info, max_batch_size);
  PRINT_FIELD(te_model_info, normalized);
  PRINT_FIELD(te_model_info, reserved);
  PRINT_FIELD(te_model_info, model_id);
  PRINT_FIELD(te_model_info, revision);
  PRINT_FIELD(te_model_info, tokenizer_sha256);
  PRINT_FIELD(te_model_info, pooling);

  PRINT_LAYOUT(te_slot_options);
  PRINT_FIELD(te_slot_options, struct_size);
  PRINT_FIELD(te_slot_options, version);
  PRINT_FIELD(te_slot_options, batch);
  PRINT_FIELD(te_slot_options, sequence_length);

  PRINT_LAYOUT(te_result_info);
  PRINT_FIELD(te_result_info, struct_size);
  PRINT_FIELD(te_result_info, version);
  PRINT_FIELD(te_result_info, batch);
  PRINT_FIELD(te_result_info, dimension);
  PRINT_FIELD(te_result_info, byte_size);

  PRINT_LAYOUT(te_opencl_view);
  PRINT_FIELD(te_opencl_view, struct_size);
  PRINT_FIELD(te_opencl_view, version);
  PRINT_FIELD(te_opencl_view, context);
  PRINT_FIELD(te_opencl_view, queue);
  PRINT_FIELD(te_opencl_view, buffer);
  PRINT_FIELD(te_opencl_view, byte_size);
  PRINT_FIELD(te_opencl_view, batch);
  PRINT_FIELD(te_opencl_view, dimension);

  PRINT_LAYOUT(te_slot_stats);
  PRINT_FIELD(te_slot_stats, struct_size);
  PRINT_FIELD(te_slot_stats, version);
  PRINT_FIELD(te_slot_stats, executions);
  PRINT_FIELD(te_slot_stats, input_write_bytes);
  PRINT_FIELD(te_slot_stats, output_read_bytes);
  PRINT_FIELD(te_slot_stats, owned_input_bytes);
  PRINT_FIELD(te_slot_stats, owned_output_bytes);

  PRINT_LAYOUT(te_text);
  PRINT_FIELD(te_text, ptr);
  PRINT_FIELD(te_text, byte_length);

  PRINT_LAYOUT(te_device_info);
  PRINT_FIELD(te_device_info, struct_size);
  PRINT_FIELD(te_device_info, version);
  PRINT_FIELD(te_device_info, device);
  PRINT_FIELD(te_device_info, ordinal);
  PRINT_FIELD(te_device_info, capabilities);
  PRINT_FIELD(te_device_info, device_name);
  PRINT_FIELD(te_device_info, runtime_version);
  PRINT_FIELD(te_device_info, driver_version);
  return 0;
}
