CC = g++
CFLAGS_OPENCL_VERSION = -DCL_TARGET_OPENCL_VERSION=300 -DCL_HPP_TARGET_OPENCL_VERSION=300
CFLAGS = -DDEBUG -fPIC -Wall -Wextra -Iinclude

DEMO_DIR = ./demo
DEMO_CFLAGS = -lOpenCL -L./lib -lDeepCpp -Iinclude

CPP_FILES = $(wildcard src/*.cpp src/**/*.cpp)
OBJ = $(patsubst src/%.cpp, $(ODIR)/%.o, $(CPP_FILES))

ODIR=obj
LLIB=lib

$(DEMO_DIR)/%: $(DEMO_DIR)/%.cpp
	$(CC) $< -o $@.out $(CFLAGS_OPENCL_VERSION) $(DEMO_CFLAGS)

$(ODIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS_OPENCL_VERSION) $(CFLAGS)

build-lib: $(OBJ)
	$(CC) $(OBJ) -shared -o $(LLIB)/libDeepCpp.so $(CFLAGS_OPENCL_VERSION) $(CFLAGS)

HEADER_FILE = include/deepCpp.h
HEADER_FILES = $(wildcard src/*.h src/**/.h)

$(HEADER_FILE): $(HEADER_FILES)
	cat $^ > $@

.PHONY: build-header
build-header: $(HEADER_FILE)

clean:
	rm -rf $(LLIB)/*.so $(ODIR)/*
	rm -rf $(DEMO_DIR)/*.out
