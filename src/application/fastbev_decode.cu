

#include <common/cuda_tools.hpp>

namespace Fastbev{

    const int NUM_BOX_ELEMENT = 10;       // fastbev:  x y z dx dy dz r  conf class  keepflag
    // static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    //     *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    //     *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    // }

    static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* parray, int max_objects){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= num_bboxes) return;
 
        float* pitem     = predict + 20 * position; //  fastbev: 7+2+11  x y z dx dy dz r d classnum 
        // float objectness = pitem[4];
        // if(objectness < confidence_threshold)
        //     return;
        float* class_confidence = pitem;
        float *conf_tmp = class_confidence;
        float confidence = *class_confidence++;
        int label               = 0;
        for(int i = 1; i < num_classes; ++i, ++class_confidence){
            if(*class_confidence > confidence){
                confidence = *class_confidence;
                label      = i;
            }
        }
        // printf("confidence_threshold[%f] %f",confidence_threshold,*pitem);

        // confidence *= objectness;
        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;
        
        pitem = pitem+num_classes;
        float x     = *pitem++;
        float y     = *pitem++;
        float z     = *pitem++;
        float dx    = *pitem++;
        float dy    = *pitem++;
        float dz    = *pitem++;
        float r     = *pitem++;
        float dir1   = *pitem++;
        float dir2   = *pitem++;

        // printf("label[%d] | confidence[%f] | x[%f] | y[%f] | z[%f] | dx[%f] | dy[%f] | dz[%f] | r[%f] | dir[%f]   \n",label,confidence,x,y,z,dx,dy,dz,r, dir);
        
        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = x;
        *pout_item++ = y;
        *pout_item++ = z;
        *pout_item++ = dx;
        *pout_item++ = dy;
        *pout_item++ = dz;
        *pout_item++ = r + (dir1 > dir2 ? 0 : 1)*3.1415926;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1.0; // 1 = keep, 0 = ignore

    }

    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom
    ){

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return;

        // x y z dx dy dz r  conf class  keepflag
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[8] != pitem[8]) continue;
            
            if(pitem[7] >= pcurrent[7]){
                if(pitem[7] == pcurrent[7] && i < position)
                    continue;

                float p_x=pcurrent[0]; float p_y=pcurrent[1];float p_dx=pcurrent[3];float p_dy=pcurrent[4];
                float n_x=pitem[0]; float n_y=pitem[1];float n_dx=pitem[3];float n_dy=pitem[4];
                
                float iou = box_iou(
                    p_x,p_y,p_x + p_dx,p_y + p_dy,
                    n_x,n_y,n_x +n_dx,n_y +n_dy
                );
                if(iou > threshold){
                    pcurrent[9] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    }

    void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold,  float* parray, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);


        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, parray, max_objects));
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(max_objects);
        auto block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }
};