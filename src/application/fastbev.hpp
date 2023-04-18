
#ifndef FASTBEV_HPP
#define FASTBEV_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

#include <common/cuda_tools.hpp>
#include <common/preprocess_kernel.cuh>
namespace Fastbev{

    using namespace std;

    enum class ImageType : int{
        CVMat  = 0,
        GPUYUV = 1    // nv12
    };

    struct Image{
        ImageType type = ImageType::CVMat;
        cv::Mat cvmat;
        std::vector<cv::Mat> cvmats;

        // GPU YUV image
        TRT::CUStream stream = nullptr;
        // uint8_t* device_data = nullptr;
        int width = 0, height = 0;
        int device_id = 0;

        Image() = default;
        Image(const std::vector<cv::Mat>& cvmats):cvmats(cvmats), type(ImageType::CVMat),
                                                width(cvmats[0].cols),height(cvmats[0].cols){}

        int get_nums() const{return  cvmats.size(); }
        int get_width() const{return cvmats[0].cols;}
        int get_height() const{return cvmats[0].rows;}
        cv::Size get_size() const{return cv::Size(get_width(), get_height()*3);}
        bool empty() const{return cvmats.size()==0 || cvmats[0].empty();}


    };


    struct Box{
        float x,y,z,dx,dy,dz,rot,confidence;
        int label;

        Box() = default;

        Box(float x, float y, float z, float dx, float dy, float dz,  float rot, float confidence, int label)
        :x(x), y(y), z(z), dx(dx),dy(dy),dz(dz),rot(rot), confidence(confidence), label(label){}
    };

    typedef std::vector<Box> BoxArray;


    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    // void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const Image& image) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file,  int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );


}; // namespace Fastbev

#endif 