CC = g++
CFLAGS_OPENCL_VERSION = -DCL_TARGET_OPENCL_VERSION=300 -DCL_HPP_TARGET_OPENCL_VERSION=300
CFLAGS = -DDEBUG -fPIC -Wall -Wextra -Iinclude

DEMO_DIR = ./demo
DEMO_CFLAGS = -L./lib -Iinclude -lOpenCL -lclblast -lcgraph -lCppDiff

TEST_DIR = ./test
TEST_ODIR = ./test/obj
TEST_CFLAGS = -lgtest

CPP_FILES = $(wildcard src/*.cpp src/**/*.cpp)
OBJ = $(patsubst src/%.cpp, $(ODIR)/%.o, $(CPP_FILES))

TEST_FILES = $(wildcard test/*.cpp)
TESTS = $(patsubst test/%.cpp, $(TEST_ODIR)/%.o, $(TEST_FILES))

ODIR=obj
LLIB=lib

$(DEMO_DIR)/%: $(DEMO_DIR)/%.cpp
	$(CC) $< -o $@.out $(CFLAGS_OPENCL_VERSION) $(DEMO_CFLAGS)

$(ODIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS_OPENCL_VERSION) $(CFLAGS)

build-lib: $(OBJ)
	$(CC) $(OBJ) -shared -o $(LLIB)/libCppDiff.so $(CFLAGS_OPENCL_VERSION) $(CFLAGS)

$(TEST_ODIR)/%.o: $(TEST_DIR)/%.cpp
	$(CC) -c -o $@ $< $(CFLAGS_OPENCL_VERSION) $(DEMO_CFLAGS) $(TEST_CFLAGS)

.PHONY: test
test: $(TESTS)
	$(CC) $(TESTS) -o $(TEST_DIR)/runtest $(DEMO_CFLAGS) $(TEST_CFLAGS) -lgtest_main
	LD_LIBRARY_PATH=./lib ./test/runtest

.PHONY: build-header
build-header: $(HEADER_FILE)

clean:
	rm -rf $(LLIB)/*.so $(ODIR)/*
	rm -rf $(DEMO_DIR)/*.out
	rm $(TEST_ODIR)/*
	rm $(TEST_DIR)/runtest
