
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "fastbev.hpp"
#include<ctime>

using namespace std;


static const char* label_map[] = {
 "Pedestrian", "Car","MotorcyleRider", "Crane", "Motorcycle", "Bus", "BicycleRider", "Van", "Excavator", "TricycleRider","Truck"
};
static void append_to_file(const string& file, const string& data){
    FILE* f = fopen(file.c_str(), "a+");
    if(f == nullptr){
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

std::vector<float> rotate_box(float x1, float y1, float x2, float y2, float r) {
    // Step 1: Translate coordinates to top-left corner
    float cx = (x1 + x2) / 2.0f;
    float cy = (y1 + y2) / 2.0f;
    x1 -= cx;
    y1 -= cy;
    x2 -= cx;
    y2 -= cy;

    // Step 2: Convert angle to radians
    r = r * M_PI / 180.0f;

    // Step 3: Compute rotation matrix
    float cos_r = cos(r);
    float sin_r = sin(r);

    // Step 4: Rotate box vertices
    float x1_new = cos_r * x1 - sin_r * y1;
    float y1_new = sin_r * x1 + cos_r * y1;
    float x2_new = cos_r * x2 - sin_r * y2;
    float y2_new = sin_r * x2 + cos_r * y2;

    // Step 5: Translate coordinates back to original position
    x1_new += cx;
    y1_new += cy;
    x2_new += cx;
    y2_new += cy;

    // Step 6: Pack rotated box coordinates into vector and return
    std::vector<float> rotated_box = {x1_new, y1_new, x2_new, y2_new};
    return rotated_box;
}

void forward(shared_ptr<Fastbev::Infer> &engine, Fastbev::Image &images,cv::Mat &bevimg){

    auto boxes = engine->commit(images).get();
    // printf("boxes [%d]\n",boxes.size());
    int bevsize_w = 1000;
    int bevsize_h = 600;
    cv::Mat img(bevsize_h, bevsize_w, CV_8UC3, cv::Scalar(255,255,255));
    for(auto& obj : boxes){
        // printf("class[%s] confidence[%f] label[%d] x[%f] y[%f] z[%f] dx[%f] dy[%f] dz[%f] rot[%f]  \n",
            // label_map[obj.label],obj.confidence,obj.label,obj.x,obj.y,obj.z,obj.dx,obj.dy,obj.dz,obj.rot);

        // Calculate the four corner points of the rotated rectangle
        uint8_t b, g, r;
        tie(b, g, r) = iLogger::random_color(obj.label + 1);

        int x = bevsize_w - (obj.y + 50)*10;
        int y = bevsize_h - obj.x * 10;
        int w = obj.dx * 10;
        int h = obj.dy * 10;
        int rot = int(90 - obj.rot/3.1415926*180 + 360)%180;
        // printf("%d %d %d %d %d \n",x,y,x+w,y+h,r);
        cv::RotatedRect box(cv::Point(x, y), cv::Size(w, h), rot);
        cv::Point2f vertex[4];
	    box.points(vertex);
        for (int i = 0; i < 4; i++)
            cv::line(img, vertex[i], vertex[(i + 1) % 4], cv::Scalar(b, g, r),10,cv::LINE_AA);

        auto caption = iLogger::format("[%s %.2f]",  label_map[obj.label],obj.confidence);
        cv::putText(img, caption, (cv::Point(x, y-w-10)), 0, 0.5, cv::Scalar(b, g, r), 1, 16);

        cv::circle(img, cv::Point(bevsize_w/2,bevsize_h), 20, cv::Scalar(0, 0, 0), cv::FILLED);

    }
    bevimg = img;
    cv::imwrite("result.png", img);
}



static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name,const string& imgpath){

    auto engine = Fastbev::create_infer(
        engine_file,                // engine file
        deviceid,                   // gpu id
        0.9f,                      // confidence threshold
        0.45f,                      // nms threshold
        Fastbev::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                       // max objects
        false                       // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }
    
    
    std::vector<cv::Mat> images_mat;
    auto imagef = cv::imread("./images/2022-05-12-11-24-22_000005_front.png");
    auto imagel = cv::imread("./images/2022-05-12-11-24-22_000005_left.png");
    auto imager = cv::imread("./images/2022-05-12-11-24-22_000005_right.png");

    images_mat.emplace_back(imagef);
    images_mat.emplace_back(imagel);
    images_mat.emplace_back(imager);

    Fastbev::Image images(images_mat);
    auto boxes = engine->commit(images).get();
    for(auto& obj : boxes)
        printf("class[%s] confidence[%f] label[%d] x[%f] y[%f] z[%f] dx[%f] dy[%f] dz[%f] rot[%f]  \n",
            label_map[obj.label],obj.confidence,obj.label,obj.x,obj.y,obj.z,obj.dx,obj.dy,obj.dz,obj.rot);

    printf("input images height %d width %d nums %d \n",images.get_height(),images.get_width(),images.get_nums());


    // warmup
    for(int i = 0; i < 10; ++i)
        auto boxes = engine->commit(images).get();


    int test_nums = 20;
    auto begin_timer = iLogger::timestamp_now_float();
    for(int i = 0; i < test_nums; ++i)
        auto boxes = engine->commit(images).get();
    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / test_nums;
    INFO(" ==== average: %.2f ms / iter, FPS: %.2f === ", inference_average_time, 1000 / inference_average_time);

    std::vector<cv::Mat> allimages;

    for(int idx =0 ;idx < 160 ;idx ++){
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << idx;

        std::string imgfile = "./roadsideimages/2022-05-09-08-47-43_" +oss.str(); //+ "_front.png"
        imagef = cv::imread(imgfile+"_front.png");
        imagel = cv::imread(imgfile+"_left.png");
        imager = cv::imread(imgfile+"_right.png");
        std::cout << imgfile+"_front.png" <<std::endl;
        images_mat.clear();
        images_mat.emplace_back(imagef);
        images_mat.emplace_back(imagel);
        images_mat.emplace_back(imager);
        Fastbev::Image images_(images_mat);

        cv::Mat bevimg;
        forward(engine,images_,bevimg);
        std::string resultfile = "./results/2022-05-09-08-47-43_" +oss.str()+ "_bev.png"; 


        cv::Mat imageflr;
        cv::Mat allimg;
        cv::hconcat(imagef, imagel, imageflr);
        cv::hconcat(imageflr, imager, imageflr); // 3840, 720
        cv::resize(imageflr, imageflr, cv::Size(3000, 720));
        
        cv::resize(bevimg, bevimg, cv::Size(3000, bevimg.rows*3));
        cv::vconcat(imageflr, bevimg, allimg); 
        cv::imwrite(resultfile, allimg);
        allimages.push_back(allimg);
    }
    std::string outputmp4 = "./results/output.mp4"; 
    cv::Size frame_size = allimages[0].size();
    cv::VideoWriter video(outputmp4, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10, frame_size, true);
    for (const auto& image : allimages) {
        video.write(image);
    }
    video.release();

    
    // /////////////////////////////////////////////////////////
    // const int ntest = 100;
    // auto begin_timer = iLogger::timestamp_now_float();

    // for(int i  = 0; i < ntest; ++i)
    //     boxes_array = engine->commits(images);
    // for (auto & image :images){
    //     boxes_array.emplace_back(engine->commit(image));
    // }

    // // wait all result
    // boxes_array.back().get();

    // float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.get_nums();
    // auto mode_name = TRT::mode_string(mode);
    // INFO("%s average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
    // append_to_file("perf.result.log", iLogger::format("%s,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    
    // string root = iLogger::format("%s_%s_%s_result", imgpath.c_str(),model_name.c_str(), mode_name);
    // iLogger::rmtree(root);
    // iLogger::mkdir(root);

    // for(int i = 0; i < boxes_array.size(); ++i){

    //     // auto& image = images[i].cvmat;
    //     auto boxes  = boxes_array[i].get();
        
    //     for(auto& obj : boxes){
    //         printf("[%f %f %f][%d %f]\n",obj.x,obj.y,obj.z,obj.label,obj.confidence);
    //         // uint8_t b, g, r;
    //         // tie(b, g, r) = iLogger::random_color(obj.class_label);
    //         // cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 2);

    //         // auto name    = cocolabels[obj.class_label];
    //         // auto caption = iLogger::format("%s [%.2f %.2f]",  name,obj.confidence,obj.depth);
    //         // int width    = cv::getTextSize(caption, 0, 0.5, 1, nullptr).width + 1;
    //         // cv::rectangle(image, cv::Point(obj.left-3, obj.top-20), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    //         // cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 0.5, cv::Scalar::all(0), 1, 16);
    //     }

    //     // string file_name = iLogger::file_name(files[i], false);
    //     // string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
    //     // INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
    //     // cv::imwrite(save_path, image);
    //     // cv::imshow("f", image);cv::waitKey(0);
    // }
    engine.reset();
}

static void test(TRT::Mode mode, const string& model,const string& imgpath,const int batch_size){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);


    const char* name = model.c_str();
    INFO("===================== test %s %s ==================================", mode_name, name);


    string onnx_file = iLogger::format("onnxs/%s.onnx", name);
    string model_file = iLogger::format("engines/%s.%s.bs%d.engine", name, mode_name, batch_size);
    std::cout << model_file<< std::endl;
    int test_batch_size = batch_size;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            nullptr,
            "inference"
        );
    }

    inference_and_performance(deviceid, model_file, mode, name,imgpath);
    
}


int app_fastbev(){

    // test(TRT::Mode::FP32, "roadside_train_half_res_aug_20230405-2208—epoch_50_20230414-1135","images",1);
    test(TRT::Mode::FP16, "roadside_train_half_res_aug_20230405-2208—epoch_50_20230414-1135","images",1);

    return 0;
}