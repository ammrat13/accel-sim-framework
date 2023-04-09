NAME = gasan-alloc-record

CUDA_INSTALL_PATH ?= /usr/local/cuda/

CFLAGS = -v -O2 -I$(CUDA_INSTALL_PATH)/include/

.PHONY: all
all: $(NAME).so

%.so: %.c
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $<

.PHONY: clean
clean:
	rm -fv $(NAME).so
