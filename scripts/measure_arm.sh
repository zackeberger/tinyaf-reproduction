#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
compiler="${TINYAF_ARM_CC:-arm-none-eabi-gcc}"
archiver="${TINYAF_ARM_AR:-arm-none-eabi-ar}"
size_tool="${TINYAF_ARM_SIZE:-arm-none-eabi-size}"
for tool in make "$compiler" "$archiver" "$size_tool"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'Missing required tool: %s\n' "$tool" >&2
    cat >&2 <<'HELP'
This script requires make and the Arm GNU bare-metal toolchain.
Add the toolchain's bin directory to PATH before running it, for example:
  export PATH="/path/to/arm-gnu-toolchain/bin:$PATH"
  bash scripts/measure_arm.sh
Alternatively, set TINYAF_ARM_CC, TINYAF_ARM_AR, and TINYAF_ARM_SIZE
to the full paths of arm-none-eabi-gcc, arm-none-eabi-ar, and arm-none-eabi-size.
HELP
    exit 1
  fi
done
mkdir -p logs
cat > logs/arm_size_harness.c <<'C'
#include "tinyaf.h"
uint16_t tinyaf_input_buffer[2560];
float tinyaf_output;
int main(void) { return predict_c(tinyaf_input_buffer,2560,TINYAF_USE_DEFAULT_THRESHOLD,&tinyaf_output); }
C
artifacts=()
for rung in S M L; do
  if ! make arm-library CC="$compiler" AR="$archiver" RUNG="$rung" MODE=optimized \
      > "logs/arm_build_${rung}.log" 2>&1; then
    cat "logs/arm_build_${rung}.log" >&2
    exit 1
  fi
  library="compile/optimized/${rung}/libtinyaf_arm.a"
  if ! "$compiler" -O3 -Ic_model -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
    -nostartfiles -ffunction-sections -fdata-sections \
    -Wl,-e,main,--gc-sections,-Map="logs/arm_${rung}.map" \
    logs/arm_size_harness.c "$library" -lm -lc -lnosys -o "logs/arm_${rung}.elf" \
    > "logs/arm_link_${rung}.log" 2>&1; then
    cat "logs/arm_link_${rung}.log" >&2
    exit 1
  fi
  artifacts+=("logs/arm_${rung}.elf")
done
"$size_tool" -B -d "${artifacts[@]}" > logs/arm_sizes_raw.txt
awk '
  BEGIN { printf "%-8s %15s %20s\n", "Model", "Flash (bytes)", "Static RAM (bytes)" }
  NR > 1 {
    model = $NF
    sub(/^.*arm_/, "", model)
    sub(/\.elf$/, "", model)
    printf "%-8s %15d %20d\n", model, $1 + $2, $2 + $3
  }
' logs/arm_sizes_raw.txt
