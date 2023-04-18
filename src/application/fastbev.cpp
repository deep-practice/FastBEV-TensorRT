#include "fastbev.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>

void print_first_100_pixels(const cv::Mat& image) {
    // 获取图像的行和列数
    int rows = image.rows;
    int cols = image.cols;
    
    // 计算图像的通道数
    int channels = image.channels();
    
    // 定义变量来迭代前100个像素
    int count = 0;
    
    // 迭代所有像素
    for (int row = 0; row < rows && count < 2000; row++) {
        for (int col = 0; col < cols && count < 2000; col++) {
            // 访问像素
            const float* pixel = (float*)image.data + (row * cols + col) * channels;
            
            // 打印像素值
            for (int c = 0; c < channels; c++) {
                std::cout << pixel[c]  << " ";
            }
            std::cout << std::endl;
            
            // 增加计数器
            count++;
        }
    }
    std::cout <<  "-------------over------------"<<std::endl;
}

#include <unistd.h>
void saveimg(float * image_host,int width,int height,int i=0){
    printf("float\n");
    usleep(1000000);
    std::string file = "test_cuda_"+std::to_string(i)+"_.ppm";
    FILE *fp = fopen(file.c_str(), "wb");
    // 将float类型的像素值转换为uint8_t类型，并写入ppm文件
    uint8_t* pixels = new uint8_t[width * height * 3];
    for (int i = 0; i < width * height * 3; i++) {
        pixels[i] = static_cast<uint8_t>(std::round(image_host[i] * 255));
    }
    fprintf(fp, "P6\n%u %u\n255\n", width, height);
    fwrite(image_host, 1, width * height * 3, fp);
    fclose(fp);
    cv::Mat cmat(height, width, CV_32FC3,image_host);

    file = "test_mat_"+std::to_string(i)+"_.png";
    cv::imwrite(file.c_str(),cmat);
}

void saveimg(uint8_t* image_host,int width,int height,int i=0){
    printf("uint8_t\n");
    usleep(1000000);
    std::string file = "test_cuda_"+std::to_string(i)+"_.ppm";
    FILE *fp = fopen(file.c_str(), "wb");
    fprintf(fp, "P6\n%u %u\n255\n", width, height);
    fwrite(image_host, 1, width * height * 3, fp);
    fclose(fp);
    cv::Mat cmat(height, width, CV_8UC3,image_host);

    file = "test_mat_"+std::to_string(i)+"_.png";
    cv::imwrite(file.c_str(),cmat);
}

void resize_normal_mat(cv::Mat &image, int width, int height, float mean[3], float std[3]) {
    // 计算缩放比例
    float scale = std::min(float(width) / image.cols, float(height) / image.rows);

    // 计算缩放后的大小
    int new_width = int(image.cols * scale);
    int new_height = int(image.rows * scale);

    // 缩放图像
    cv::resize(image, image, cv::Size(new_width, new_height));
    
    // 在上下两端用0填充
    int top_pad = (height - new_height) / 2;
    int bottom_pad = height - new_height - top_pad;
    cv::copyMakeBorder(image, image, top_pad, bottom_pad, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // std::cout << new_width <<"  "  << new_height  <<"  "<< top_pad  <<"  " << bottom_pad <<std::endl;
    // // 将图像大小调整为指定的宽度和高度
    // cv::resize(image, image, cv::Size(width, height));
    
    // 转换图像数据类型为float
    image.convertTo(image, CV_32FC3,1.0);

    cv::Mat ms[3];
    cv::split(image, ms);

    for (int c = 0; c < 3; ++c)
        ms[c] = (ms[c] - mean[c]) / std[c];

    cv::Mat output;
    cv::merge(ms, 3,output);
    image = output;
}

namespace Fastbev{
    using namespace cv;
    using namespace std;


    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* parray,
        int max_objects, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );

    void test_kernel_invoker(
        
    );

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;

            // 这里取min的理由是
            // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
            // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
            // **
            float scale = std::min(scale_x, scale_y);

            /**
            这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
            scale, 0, -scale * from.width * 0.5 + to.width * 0.5
            0, scale, -scale * from.height * 0.5 + to.height * 0.5
            
            这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
            例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
            S = [
            scale,     0,      0
            0,     scale,      0
            0,         0,      1
            ]
            
            P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
            P = [
            1,        0,      -scale * from.width * 0.5
            0,        1,      -scale * from.height * 0.5
            0,        0,                1
            ]

            T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
            T = [
            1,        0,      to.width * 0.5,
            0,        1,      to.height * 0.5,
            0,        0,            1
            ]

            通过将3个矩阵顺序乘起来，即可得到下面的表达式：
            M = [
            scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
            0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
            0,        0,                     1
            ]
            去掉第三行就得到opencv需要的输入2x3矩阵
            **/

            /* 
                 + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
                参考：https://www.iteye.com/blog/handspeaker-1545126
            */
            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
            
            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    // static float iou(const Box& a, const Box& b){
    //     float cleft 	= max(a.left, b.left);
    //     float ctop 		= max(a.top, b.top);
    //     float cright 	= min(a.right, b.right);
    //     float cbottom 	= min(a.bottom, b.bottom);
        
    //     float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    //     if(c_area == 0.0f)
    //         return 0.0f;
        
    //     float a_area = max(0.0f, a.right - a.left) * max(0.0f, a.bottom - a.top);
    //     float b_area = max(0.0f, b.right - b.left) * max(0.0f, b.bottom - b.top);
    //     return c_area / (a_area + b_area - c_area);
    // }

    // static BoxArray cpu_nms(BoxArray& boxes, float threshold){

    //     std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b){
    //         return a.confidence > b.confidence;
    //     });

    //     BoxArray output;
    //     output.reserve(boxes.size());

    //     vector<bool> remove_flags(boxes.size());
    //     for(int i = 0; i < boxes.size(); ++i){

    //         if(remove_flags[i]) continue;

    //         auto& a = boxes[i];
    //         output.emplace_back(a);

    //         for(int j = i + 1; j < boxes.size(); ++j){
    //             if(remove_flags[j]) continue;
                
    //             auto& b = boxes[j];
    //             if(b.class_label == a.class_label){
    //                 if(iou(a, b) >= threshold)
    //                     remove_flags[j] = true;
    //             }
    //         }
    //     }
    //     return output;
    // }

    using ControllerImpl = InferController
    <
        Image,                  // input
        BoxArray,               // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(
            const string& file, int gpuid, 
            float confidence_threshold, float nms_threshold,
            NMSMethod nms_method, int max_objects,
            bool use_multi_preprocess_stream
        ){
           
            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            nms_method_           = nms_method;
            max_objects_          = max_objects;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        virtual void worker(promise<bool>& result) override{

            string file = get<0>(start_param_);
            int gpuid   = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            const int MAX_IMAGE_BBOX  = max_objects_;
            const int NUM_BOX_ELEMENT = 10;       // fastbev:  x y z dx dy dz r  conf class  keepflag
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("input");
            auto output        = engine->tensor("output");
            int num_classes    = output->size(2) - 9;  //  fastbev classnum fastbev: 7+ 1 +11 x y z w l h r d classnum
            printf("num_classes : [%d] \n",num_classes);

            input_cam_num_      = input->size(1);
            input_height_      = input->size(2);
            input_width_       = input->size(3);
            input_channel_      = input->size(4);

            printf("input  cam_num[%d] width[%d] height [%d] channel_ [%d]] \n",input_cam_num_,input_width_,input_height_,input_channel_);

            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    CUDATools::AutoDevice auto_device_exchange(mono->device());

                    if(mono->get_stream() != stream_){
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }

                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_based_output = output->gpu<float>(ibatch);
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));

                    decode_kernel_invoker(image_based_output, output->size(1), num_classes, confidence_threshold_, output_array_ptr, MAX_IMAGE_BBOX, stream_);

                    if(nms_method_ == NMSMethod::FastGPU){
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                }

                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float *parray = output_array_device.cpu<float>(ibatch);
                    int count = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto &job = fetch_jobs[ibatch];
                    auto &image_based_boxes = job.output;
                    for(int i = 0; i < count; ++i){
                        
                        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;

                        int keepflag = pbox[9];
                        if (keepflag == 1)
                        {                                  // x        y        z       dx      dy      dz      r       conf      label
                            image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5],pbox[6], pbox[7] ,int(pbox[8]) );
                        }
                    }

                    job.pro->set_value(image_based_boxes);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Image& image) override{

            if(tensor_allocator_ == nullptr){
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            if(image.empty()){
                INFOE("Image is empty");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            int current_device = gpu_;
            if(image.type == ImageType::GPUYUV){
                current_device = image.device_id;
            }
            
            CUDATools::AutoDevice auto_device(current_device);
            auto& tensor = job.mono_tensor->data();
            TRT::CUStream preprocess_stream = nullptr;

            if(use_multi_preprocess_stream_){
                if(image.type == ImageType::GPUYUV){
                    INFOW("Will ignore use_multi_preprocess_stream_ flag during hard decode");
                    use_multi_preprocess_stream_ = false;
                }
            }

            // device changed
            if(tensor != nullptr && tensor->device() != current_device)
                tensor.reset();

            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());

                if(use_multi_preprocess_stream_){
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));

                    // owner = true, stream needs to be free during deconstruction
                    tensor->set_stream(preprocess_stream, true);
                }else{
                    if(image.type == ImageType::GPUYUV){
                        preprocess_stream = image.stream != nullptr ? image.stream : stream_;
                    }else{
                        preprocess_stream = stream_;
                    }

                    // owner = false, tensor ignored the stream
                    tensor->set_stream(preprocess_stream, false);
                }
            }

            // Size input_size(input_width_, input_height_*3);
            // job.additional.compute(image.get_size(), input_size);
            
            preprocess_stream = tensor->get_stream();
            tensor->resize(1, input_cam_num_,input_height_, input_width_,input_channel_ );

            size_t size_image      = image.get_width() * image.get_height() * 3;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image*3);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image*3);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;


            float mean[3]={103.53, 116.28, 123.675 }; // bgr
            float std[3]={ 57.375, 57.12, 58.395};

            auto tensor_input_size = input_width_*input_height_*3;
            int count=0;
            for (int i = 0; i < image.cvmats.size(); ++i) {
                cv::Mat img = image.cvmats[i];
                resize_normal_mat(img,input_width_,input_height_,mean,std);
                checkCudaRuntime(cudaMemcpyAsync(tensor->gpu<float>() + i * tensor_input_size, reinterpret_cast<float*>(img.data), tensor_input_size*sizeof(float), cudaMemcpyHostToDevice, preprocess_stream));

            }
            cudaStreamSynchronize(preprocess_stream);

            // // 打印前100个float
            // float* input_data = static_cast<float*>(tensor->cpu<float>());
            // for (int i = 0; i < 100; ++i) {
            //     std::cout << input_data[i] << " ";
            // }
            // std::cout << "INPUT === "<<std::endl;
            // printf(" \ntensor->cpu<float>() \n ");
            return true;
        }



        virtual std::shared_future<BoxArray> commit(const Image& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int input_cam_num_          = 0;
        int input_channel_          = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        int max_objects_            = 1024;
        NMSMethod nms_method_       = NMSMethod::FastGPU;
        TRT::CUStream stream_       = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file,  int gpuid, 
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects,
        bool use_multi_preprocess_stream
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(
            engine_file, gpuid, confidence_threshold, 
            nms_threshold, nms_method, max_objects, use_multi_preprocess_stream)
        ){
            instance.reset();
        }
        return instance;
    }

};

