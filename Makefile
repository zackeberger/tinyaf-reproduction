CC ?= cc
AR ?= ar
RUNG ?= S
MODE ?= optimized
ifeq ($(filter $(RUNG),S M L),)
$(error RUNG must be S, M, or L)
endif
ifeq ($(filter $(MODE),reference optimized),)
$(error MODE must be reference or optimized)
endif
OPTIMIZED = $(if $(filter optimized,$(MODE)),1,0)
CFLAGS ?= -O3 -std=c99 -Wall -Wextra -Wpedantic
CPPFLAGS += -Ic_model -DMODEL_WEIGHTS_HEADER='"TINYAF_$(RUNG)_weights.h"' -DTINYAF_OPTIMIZED=$(OPTIMIZED)
NUMERICS = -fno-fast-math -ffp-contract=off
OUT = compile/$(MODE)/$(RUNG)
.PHONY: all clean arm
all: $(OUT)/libtinyaf.so $(OUT)/tinyaf_caller
$(OUT):
	mkdir -p $@
$(OUT)/libtinyaf.so: c_model/tinyaf.c c_model/tinyaf.h c_model/TINYAF_$(RUNG)_weights.h c_model/filter_coefficients.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(NUMERICS) -DTINYAF_DIAGNOSTICS -fPIC -shared $< -lm -o $@
$(OUT)/tinyaf_caller: tinyaf_caller.c c_model/tinyaf.c c_model/tinyaf.h c_model/TINYAF_$(RUNG)_weights.h c_model/filter_coefficients.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(NUMERICS) tinyaf_caller.c c_model/tinyaf.c -lm -o $@
arm:
	$(MAKE) CC=arm-none-eabi-gcc AR=arm-none-eabi-ar MODE=optimized arm-library
.PHONY: arm-library
arm-library: | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(NUMERICS) -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -DTINYAF_ENABLE_NATIVE=0 -ffunction-sections -fdata-sections -fstack-usage -c c_model/tinyaf.c -o $(OUT)/tinyaf_arm.o
	$(AR) rcs $(OUT)/libtinyaf_arm.a $(OUT)/tinyaf_arm.o
clean:
	rm -rf compile
