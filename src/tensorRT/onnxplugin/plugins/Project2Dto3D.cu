
#include <onnxplugin/onnxplugin.hpp>
#include <cuda_fp16.hpp>

#include <fstream>

#include <typeinfo>

// #include <tensor.hpp>
enum class DeviceType: int32_t {
    kHOST = 0,
    kGPU = 1
}; // emum class DeviceType

using namespace ONNXPlugin;




__global__ void build_LUT_kernel(int32_t n_x_voxels, int32_t n_y_voxels, int32_t n_z_voxels,
                                    float* voxel_size, float* origin, float* projection,
                                    int32_t* LUT, 
                                    int32_t n_images, int32_t height, int32_t width) {
   
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t zi = idx % n_z_voxels;
    idx /= n_z_voxels;
    int32_t yi = idx % n_y_voxels;
    idx /= n_y_voxels;
    int32_t xi = idx % n_x_voxels;
    idx /= n_x_voxels;
    int32_t img = idx;
    // printf("Current line: %d\n", __LINE__);
    if (img < n_images && LUT[(xi * n_y_voxels + yi) * n_z_voxels + zi] == -1 ) {
        // printf("Current line: %d\n", __LINE__);

        float size_x = voxel_size[0];
        // printf("Current line: %d\n", __LINE__);

        float size_y = voxel_size[1];
        float size_z = voxel_size[2];
        // printf("Current line: %d\n", __LINE__);

        float ar[3];
        float pt[3];
        // printf("Current line: %d\n", __LINE__);
        pt[0] = (xi - n_x_voxels / 2.0f) * size_x + origin[0];
        pt[1] = (yi - n_y_voxels / 2.0f) * size_y + origin[1];
        pt[2] = (zi - n_z_voxels / 2.0f) * size_z + origin[2];

        // printf("Current line: %d\n", __LINE__);
        for (int i = 0; i < 3; ++i) {
            ar[i] = 0;
            for (int j = 0; j < 3; ++j) {
                ar[i] += projection[(img * 3 + i) * 4 + j] * pt[j];
            }
            ar[i] += projection[((img * 3) + i) * 4 + 3];
        }
        // printf("Current line: %d\n", __LINE__);
        int32_t x = round(ar[0] / ar[2]);
        int32_t y = round(ar[1] / ar[2]);
        float z = ar[2];

        // printf("Current line: %d\n", __LINE__);
        bool fit_in = (x >= 0) && (y >= 0) && (x < width) && (y < height) && (z > 0);
        int32_t target;
        if (fit_in) {
            target = (img * height + y) * width + x;
            
            int offset = (xi * n_y_voxels + yi) * n_z_voxels + zi;  // [xi,yi,zi]
            LUT[offset] = target;
            
            // valid[offset] = fit_in;
            // printf("Current line: %d\n", __LINE__);

        }
        else {
            target = -1;
            int offset = (xi * n_y_voxels + yi) * n_z_voxels + zi;  // [xi,yi,zi]
            LUT[offset] = target;
            
        }

        // printf("Current line: %d\n", __LINE__);
    }

}

__global__ void backproject_LUT_kernel(float* features, int32_t* LUT, float* volume,
    size_t total_nrof_voxels, int32_t n_channels) {
    int32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nrof_float4_copies_per_iter = n_channels / 4; // We assume n_channels % 4 == 0
    if (offset < total_nrof_voxels) {
        int32_t target = LUT[offset];
        if (target >= 0) {
            float4* src = (float4*)(features + target * n_channels);
            float4* dst = (float4*)(volume + offset * n_channels);
            for (size_t i = 0; i < nrof_float4_copies_per_iter; ++i) {
                dst[i] = src[i];
            }
        }
    }
}

void backproject_LUT_CUDA(float* features_dev, int32_t* LUT_dev, float* volume_dev,
                        int32_t n_images,  int32_t n_channels,
                        float* n_voxels) {
    // int32_t n_x_voxels = int32_t(n_voxels[0]);
    // int32_t n_y_voxels = int32_t(n_voxels[1]);
    // int32_t n_z_voxels = int32_t(n_voxels[2]);
    int32_t n_x_voxels = 200;
    int32_t n_y_voxels = 200;
    int32_t n_z_voxels = 4;
    size_t total_nrof_voxels = n_images * n_x_voxels * n_y_voxels * n_z_voxels;
    #define BLOCK_SIZE 1024
    dim3 thread_per_block(BLOCK_SIZE);
    dim3 block_per_grid((total_nrof_voxels + thread_per_block.x - 1) / thread_per_block.x);
    backproject_LUT_kernel<<< block_per_grid, thread_per_block >>>(features_dev, LUT_dev, volume_dev,
        total_nrof_voxels, n_channels
    );
}

void backproject_LUT_GPU(float * features, int32_t * LUT, float* volume,
                        float * n_voxels,int32_t n_images,int32_t n_channels) {
    backproject_LUT_CUDA(features, LUT, volume,
        n_images, n_channels,
        n_voxels
    );
}


void build_LUT_cuda(float* n_voxels, float* voxel_size_dev, float* origin_dev, float* projection,
                    int32_t* LUT, 
                    int32_t n_images, int32_t height, int32_t width) {
    // int32_t n_x_voxels = int32_t(n_voxels[0]);
    // int32_t n_y_voxels = int32_t(n_voxels[1]);
    // int32_t n_z_voxels = int32_t(n_voxels[2]);
    int32_t n_x_voxels = 200;
    int32_t n_y_voxels = 200;
    int32_t n_z_voxels = 4;
    size_t total_nrof_voxels = n_images * n_x_voxels * n_y_voxels * n_z_voxels;
    #define BLOCK_SIZE 1024
    dim3 thread_per_block(BLOCK_SIZE);
    dim3 block_per_grid((total_nrof_voxels + thread_per_block.x - 1) / thread_per_block.x);

    // printf("build here\n");
    build_LUT_kernel<<< block_per_grid, thread_per_block >>>(n_x_voxels, n_y_voxels, n_z_voxels, 
                        voxel_size_dev, origin_dev, projection,
                        LUT, 
                        n_images, height, width);
}

void build_LUT_GPU(float* n_voxels, float* voxel_size, float* origin,
                    float* projection, int32_t n_images, int32_t height, int32_t width, int32_t n_channels,
                    int32_t* LUT) {

    build_LUT_cuda(n_voxels, voxel_size, origin, projection,
                    LUT,
                    n_images, height, width
                    );

}

// 初始化空间，等价于cudaMemset(LUT,-1,lutsize*sizeof(int32_t));但是它不会产生异常
__global__ void initializeWorkspaceKernel(int32_t* workspace, size_t numElements, int32_t initValue) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        workspace[idx] = initValue;
    }
}
void initializeWorkspace(void* workspace, size_t workspaceSize, cudaStream_t stream) {
    int32_t* workspaceData = static_cast<int32_t*>(workspace);
    size_t numElements = workspaceSize / sizeof(int32_t);
    const int32_t initValue = -1;

    // Use a block size of 256 threads
    const int32_t blockSize = 1024;
    const int32_t numBlocks = (numElements + blockSize - 1) / blockSize;

    // Launch the CUDA kernel to initialize the workspace with -1
    initializeWorkspaceKernel<<<numBlocks, blockSize, 0, stream>>>(workspaceData, numElements, initValue);
}


class Project2Dto3D : public TRTPlugin {
public:
	SetupPlugin(Project2Dto3D);

	virtual void config_finish() override{
	}

	virtual std::shared_ptr<LayerConfig> new_config() override{
		auto cfg = TRTPlugin::new_config();
		cfg->support_dtype_set_ = {nvinfer1::DataType::kFLOAT};

		return cfg;
	}

	size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept
	{   
		int32_t nSlices =  outputs[0].dims.d[0] *outputs[0].dims.d[1] *outputs[0].dims.d[2] *outputs[0].dims.d[3];
        printf("nSlices %d \n",nSlices);
		return nSlices * sizeof(int32_t);
	}


    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept{
        nvinfer1::DimsExprs output_dims;
        std::vector<int32_t> n_voxels{200, 200, 4};
        output_dims.nbDims = 4;
        output_dims.d[0] = exprBuilder.constant(n_voxels[0]);
        output_dims.d[1] = exprBuilder.constant(n_voxels[1]);
        output_dims.d[2] = exprBuilder.constant(n_voxels[2]);
        output_dims.d[3] = inputs[0].d[3]; //64

        return output_dims;
    }

    void cal_debug(float *input, int32_t size,int line,cudaStream_t stream,std::string input_name="") {

        std::cout << "================" << std::endl;
        std::cout << "Input parameter name: " << input_name << std::endl;

        cudaStreamSynchronize(stream);
        float * cal_features;
        cal_features = (float*)malloc(size*sizeof(float));
        cudaMemcpy(cal_features, input, size*sizeof(float), cudaMemcpyDeviceToHost);

        std::cout.setf(std::ios::fixed,std::ios::floatfield);
        std::cout.precision(4);
        auto print_size = size>100?100:size;
        for(int i=0;i<print_size;i++)
            std::cout << " "  << cal_features[i] ;
        std::cout << " line " << line << " size "<< size<<std::endl;
        
        float sum = 0.0f;
        float sumabs = 0.0f;
        float mean = 0.0f;
        float max = 0.0f;
        float min = 0.0f;
        
        // std::ofstream outfile("./feather.txt");
	    // outfile.open("./feather.txt", ios::out);
        
        for (int i = 0; i < size; i++) {
            sum += cal_features[i];  // 将当前元素加入到总和中
            sumabs += (cal_features[i]>0? cal_features[i]:-cal_features[i]);  // 将当前元素加入到总和中
            if (i == 0 || cal_features[i] > max) {  // 如果当前元素大于最大值，或者是第一个元素
                max = cal_features[i];  // 更新最大值
            }
            if (i == 0 || cal_features[i] < min) {  // 如果当前元素小于最小值，或者是第一个元素
                min = cal_features[i];  // 更新最小值
            }

            // outfile<<cal_features[i] << " ";
        }

        // outfile.close();
        mean = sum / size;  // 计算平均值
        std::cout << "size: " << size<< std::endl ;
        std::cout << "Line: " << line<< std::endl ;
        std::cout << "Sum: " << sum<< std::endl ;
        std::cout << "Sumabs: " << sumabs<< std::endl ;
        std::cout << "Mean: " << mean<< std::endl ;
        std::cout << "Max: " << max<< std::endl ;
        std::cout << "Min: " << min << std::endl;
        std::cout << "================" << std::endl;

    }
    void cal_debug(int32_t *input, int32_t size,int line,cudaStream_t stream,std::string input_name="") {
        std::cout << "================" << std::endl;
        std::cout << "Input parameter name: " << input_name << std::endl;
        cudaStreamSynchronize(stream);
        int32_t * cal_features;
        cal_features = (int32_t*)malloc(size*sizeof(int32_t));
        cudaMemcpy(cal_features, input, size*sizeof(int32_t), cudaMemcpyDeviceToHost);

        std::cout.setf(std::ios::fixed,std::ios::floatfield);
        std::cout.precision(4);
        for(int i=0;i<100;i++)
            std::cout << " "  << cal_features[i] ;
        std::cout << " line " << line << " size "<< size<<std::endl;
        
        int32_t sum = 0;
        int32_t sumabs = 0;
        int32_t mean = 0;
        int32_t max = 0;
        int32_t min = 0;
        
        // std::ofstream outfile("./feather.txt");
	    // outfile.open("./feather.txt", ios::out);
        
        for (int i = 0; i < size; i++) {
            sum += cal_features[i];  // 将当前元素加入到总和中
            sumabs += (cal_features[i]>0? cal_features[i]:-cal_features[i]);  // 将当前元素加入到总和中
            if (i == 0 || cal_features[i] > max) {  // 如果当前元素大于最大值，或者是第一个元素
                max = cal_features[i];  // 更新最大值
            }
            if (i == 0 || cal_features[i] < min) {  // 如果当前元素小于最小值，或者是第一个元素
                min = cal_features[i];  // 更新最小值
            }

            // outfile<<cal_features[i] << " ";
        }

        // outfile.close();
        mean = sum / size;  // 计算平均值
        std::cout << "size: " << size<< std::endl ;
        std::cout << "Line: " << line<< std::endl ;
        std::cout << "Sum: " << sum<< std::endl ;
        std::cout << "Sumabs: " << sumabs<< std::endl ;
        std::cout << "Mean: " << mean<< std::endl ;
        std::cout << "Max: " << max<< std::endl ;
        std::cout << "Min: " << min << std::endl;
        std::cout << "================" << std::endl;

    }
    

	int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override{
        // std::chrono::high_resolution_clock::time_point t1, t2;
        
        /////////////////////////////
        /////////////////////////////
        // 各种初始化，各种各种
        auto &features_tensor = inputs[0];
        auto &param_tensor = weights[0];
        auto &volume_output = outputs[0];

		if (config_->usage_dtype_ == TRT::DataType::Float) {
		}
		else if (config_->usage_dtype_ == TRT::DataType::Float16) { // TODO FP16需要数据流转，很不合理，待优化FP16Plugin
            return 1;
			INFOF("not implement function");
		}

        int32_t n_images = features_tensor.shape_[0];
        int32_t height = features_tensor.shape_[1];
        int32_t width = features_tensor.shape_[2];
        int32_t n_channels = features_tensor.shape_[3];

        float * features = features_tensor.ptr<float>();

        float * param_ = param_tensor.ptr<float>();

        float * n_voxels_float = param_;
        float * voxel_size_tensor = n_voxels_float + 3;
        float * origin_tensor = voxel_size_tensor +3;
        float * projection_tensor = origin_tensor + 3;

        
        int32_t *LUT  = (int32_t *)workspace;
        size_t lutsize = volume_output.shape_[0]*volume_output.shape_[1]*volume_output.shape_[2]*volume_output.shape_[3];


        /////////////////////////////
        // 初始化LUT  -1
        initializeWorkspace(LUT, lutsize, stream);
        
        /////////////////////////////
        // 创建LUT 映射表 TODO代优化 初步思路在pytorch中实现，然后传进来
        build_LUT_GPU(n_voxels_float, voxel_size_tensor, origin_tensor, projection_tensor,
                    n_images, height, width, n_channels, LUT);


        //////////////////////////////////
        // 投影
        backproject_LUT_GPU(features, LUT, volume_output.ptr<float>(), n_voxels_float,n_images, n_channels);

       
		return 0;
	}
};

RegisterPlugin(Project2Dto3D);