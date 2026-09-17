/* SPDX-License-Identifier: Apache-2.0 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <turboembed_prepared.h>

static int report_error(const char *operation, uint32_t status, const te_error *error) {
  fprintf(stderr, "%s failed (status %u): %s\n", operation, status,
          error->message[0] ? error->message : "no error message");
  return 1;
}

int main(int argc, char **argv) {
  const char text[] = "hello world";
  const char *bundle_path;
  uint32_t device = TE_DEVICE_OPENVINO_GPU;
  te_context *context = NULL;
  te_model *model = NULL;
  te_slot *slot = NULL;
  te_result *result = NULL;
  te_error error = {0};
  int exit_code = 1;

  if (argc != 2 && argc != 3) {
    fprintf(stderr, "usage: %s <bundle-path> [cpu]\n", argv[0]);
    return 2;
  }
  if (argc == 3) {
    if (strcmp(argv[2], "cpu") != 0) {
      fprintf(stderr, "optional device argument must be the literal cpu\n");
      return 2;
    }
    device = TE_DEVICE_OPENVINO_CPU;
  }
  bundle_path = argv[1];

  /* Discovery lists selectable devices; selection below stays explicit and
   * still fails when the requested device is absent. */
  uint32_t device_count = 0;
  uint32_t status = turboembed_prepared_v1_device_count(&device_count, &error);
  if (status != TE_OK) {
    return report_error("device discovery", status, &error);
  }
  for (uint32_t index = 0; index < device_count; ++index) {
    te_device_info device_info = {
        .struct_size = sizeof(device_info),
        .version = TE_PREPARED_VERSION,
    };
    status = turboembed_prepared_v1_device_info(index, &device_info, &error);
    if (status != TE_OK) {
      return report_error("device info", status, &error);
    }
    printf("discovered device=%u ordinal=%u name=%s runtime=%s\n",
           device_info.device, device_info.ordinal, device_info.device_name,
           device_info.runtime_version);
  }

  te_context_options context_options = {
      .struct_size = sizeof(context_options),
      .version = TE_PREPARED_VERSION,
      .device = device,
      .ordinal = 0,
  };
  status = turboembed_prepared_v1_context_create(
      &context_options, &context, &error);
  if (status != TE_OK) {
    return report_error("context create", status, &error);
  }

  status = turboembed_prepared_v1_model_load(
      context, bundle_path, (uint64_t)strlen(bundle_path), &model, &error);
  if (status != TE_OK) {
    report_error("model load", status, &error);
    goto cleanup;
  }

  te_slot_options slot_options = {
      .struct_size = sizeof(slot_options),
      .version = TE_PREPARED_VERSION,
      .batch = 1,
      .sequence_length = 32,
  };
  status = turboembed_prepared_v1_slot_create(
      model, &slot_options, &slot, &error);
  if (status != TE_OK) {
    report_error("slot create", status, &error);
    goto cleanup;
  }

  te_text input = {.ptr = text, .byte_length = sizeof(text) - 1};
  status = turboembed_prepared_v1_slot_write_text(slot, &input, 1, &error);
  if (status != TE_OK) {
    report_error("text write", status, &error);
    goto cleanup;
  }

  status = turboembed_prepared_v1_slot_execute(slot, &result, &error);
  if (status != TE_OK) {
    report_error("execute", status, &error);
    goto cleanup;
  }

  te_result_info result_info = {
      .struct_size = sizeof(result_info),
      .version = TE_PREPARED_VERSION,
  };
  status = turboembed_prepared_v1_result_info(result, &result_info, &error);
  if (status != TE_OK) {
    report_error("result info", status, &error);
    goto cleanup;
  }
  if (result_info.batch != 1 || result_info.dimension != 384) {
    fprintf(stderr, "unexpected result shape: batch=%u dimension=%u\n",
            result_info.batch, result_info.dimension);
    goto cleanup;
  }

  float values[384];
  status = turboembed_prepared_v1_result_read(result, values, 384, &error);
  if (status != TE_OK) {
    report_error("result read", status, &error);
    goto cleanup;
  }

  te_context_info context_info = {
      .struct_size = sizeof(context_info),
      .version = TE_PREPARED_VERSION,
  };
  status = turboembed_prepared_v1_context_info(context, &context_info, &error);
  if (status != TE_OK) {
    report_error("context info", status, &error);
    goto cleanup;
  }

  double squared_norm = 0.0;
  for (size_t index = 0; index < 384; ++index) {
    squared_norm += (double)values[index] * values[index];
  }
  if (!isfinite(squared_norm) || fabs(sqrt(squared_norm) - 1.0) > 1e-4) {
    fprintf(stderr, "embedding is not finite and normalized\n");
    goto cleanup;
  }
  status = turboembed_prepared_v1_result_release(result, &error);
  result = NULL;
  if (status != TE_OK) {
    report_error("text result release", status, &error);
    goto cleanup;
  }

  /* These token IDs belong to this tokenizer, not to arbitrary models. */
  te_model_info model_info = {
      .struct_size = sizeof(model_info), .version = TE_PREPARED_VERSION};
  status = turboembed_prepared_v1_model_info(model, &model_info, &error);
  if (status != TE_OK) {
    report_error("model info", status, &error);
    goto cleanup;
  }
  if (strcmp(model_info.tokenizer_sha256,
             "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037") != 0) {
    fprintf(stderr, "prepared example requires the pinned MiniLM tokenizer\n");
    goto cleanup;
  }
  int32_t ids[32] = {101, 7592, 2088, 102};
  int32_t masks[32] = {1, 1, 1, 1};
  status = turboembed_prepared_v1_slot_write_tokens(slot, ids, masks, NULL, 32, &error);
  if (status != TE_OK) {
    report_error("token write", status, &error);
    goto cleanup;
  }
  float prepared[384];
  for (int repeat = 0; repeat < 3; ++repeat) {
    status = turboembed_prepared_v1_slot_execute(slot, &result, &error);
    if (status != TE_OK) {
      report_error("prepared execute", status, &error);
      goto cleanup;
    }
    status = turboembed_prepared_v1_result_read(result, prepared, 384, &error);
    if (status != TE_OK) {
      report_error("prepared read", status, &error);
      goto cleanup;
    }
    double squared_error = 0.0;
    for (size_t index = 0; index < 384; ++index) {
      double difference = (double)prepared[index] - values[index];
      if (!isfinite(difference) || fabs(difference) > 1e-6) {
        fprintf(stderr, "prepared/text maximum error exceeded\n");
        goto cleanup;
      }
      squared_error += difference * difference;
    }
    if (sqrt(squared_error / 384) > 1e-7) {
      fprintf(stderr, "prepared/text RMSE exceeded\n");
      goto cleanup;
    }
    status = turboembed_prepared_v1_result_release(result, &error);
    result = NULL;
    if (status != TE_OK) {
      report_error("prepared result release", status, &error);
      goto cleanup;
    }
  }
  printf("device=%s (%u) dimension=%u norm=%.6f prepared/text=PASS\n",
         context_info.device_name, context_info.device, result_info.dimension,
         sqrt(squared_norm));
  exit_code = 0;

cleanup:
  if (result != NULL) {
    status = turboembed_prepared_v1_result_release(result, &error);
    if (status != TE_OK) {
      report_error("result release", status, &error);
      exit_code = 1;
    }
  }
  if (slot != NULL) {
    turboembed_prepared_v1_slot_release(slot);
  }
  if (model != NULL) {
    turboembed_prepared_v1_model_release(model);
  }
  if (context != NULL) {
    turboembed_prepared_v1_context_release(context);
  }
  return exit_code;
}
