

cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.o)
cpp_objs := $(cpp_objs:src/%=build/%)
cpp_mk   := $(cpp_objs:.o=.mk)

cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cuo)
cu_objs := $(cu_objs:src/%=build/%)
cu_mk   := $(cu_objs:.cuo=.cumk)

# 配置你的库路径
lean_protobuf  := /home/wicri/fy/code/cpp_cuda_centernet/protobuf/aarch64/protobuf
lean_tensor_rt := /usr/lib/aarch64-linux-gnu/
lean_cudnn     := /usr/local/cuda
lean_opencv    := /usr/include/
lean_cuda      := /usr/local/cuda
use_python     := false
python_root    := /datav/software/anaconda3
python_name    := python3.9

include_paths := src        \
			src/application \
			src/camerasdk \
			src/tensorRT	\
			src/tensorRT/common  \
			$(lean_protobuf)/include \
			$(lean_opencv)/opencv4/ \
			/usr/include/aarch64-linux-gnu/ \
			$(lean_cuda)/include  \
			$(lean_cudnn)/include 

library_paths := $(lean_protobuf)/lib \
			$(lean_opencv)/opencv4/    \
			/usr/include/aarch64-linux-gnu/ \
			$(lean_cuda)/lib64  \
			$(lean_cudnn)/lib

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvinfer_plugin \
			cuda cublas cudart cudnn \
			stdc++ protobuf dl  opencv_highgui opencv_ml

# HAS_PYTHON表示是否编译python支持
support_define    := 

ifeq ($(use_python), true) 
include_paths  += $(lean_python)/include/python3.8
library_paths  += $(lean_python)/lib
link_librarys  += python3.8
support_define += -DHAS_PYTHON
endif

paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
cpp_compile_flags := -std=c++11 -fPIC -g -fopenmp -w -O0 $(support_define)
cu_compile_flags  := -std=c++11 -m64 -Xcompiler -fPIC -g -w -gencode=arch=compute_72,code=sm_72 -O0 $(support_define)
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags 		  += $(library_paths) $(link_librarys) $(paths)

cpp_compile_flags += -I/opt/ros/melodic/include

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

pro    : workspace/pro

workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(link_flags)


build/%.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@g++ -c $< -o $@ $(cpp_compile_flags)

build/%.cuo : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@nvcc -c $< -o $@ $(cu_compile_flags)

build/%.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@g++ -M $< -MF $@ -MT $(@:.mk=.o) $(cpp_compile_flags)
	
build/%.cumk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@nvcc -M $< -MF $@ -MT $(@:.cumk=.o) $(cu_compile_flags)


bev : workspace/pro
	@cd workspace && ./pro fastbev

clean :
	@rm -rf build workspace/pro python/trtpy/libtrtpyc.so python/build python/dist python/trtpy.egg-info python/trtpy/__pycache__
	@rm -rf build
