CC = g++
CFLAGS_OPENCL_VERSION = -DCL_TARGET_OPENCL_VERSION=300 -DCL_HPP_TARGET_OPENCL_VERSION=300
CFLAGS = -DDEBUG -fPIC -Wall -Wextra -Iinclude

DEMO_DIR = ./demo
DEMO_CFLAGS = -L./lib -Iinclude -lOpenCL -lclblast -lcgraph -lCppDiff

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
	$(CC) $(OBJ) -shared -o $(LLIB)/libCppDiff.so $(CFLAGS_OPENCL_VERSION) $(CFLAGS)

.PHONY: test
test:
	$(CC) ./test/test_univariate.cpp -o ./test/runtest -lgtest_main -lgtest $(DEMO_CFLAGS)
	LD_LIBRARY_PATH=./lib ./test/runtest

.PHONY: build-header
build-header: $(HEADER_FILE)

clean:
	rm -rf $(LLIB)/*.so $(ODIR)/*
	rm -rf $(DEMO_DIR)/*.out
