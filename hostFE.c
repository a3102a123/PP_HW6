#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status,ret;
    int img_size = imageWidth * imageHeight;
    int filterSize = filterWidth * filterWidth;
    // create command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &ret);
    // create kernel mem
    cl_mem input_img_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, img_size * sizeof(float), NULL, &ret);
    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float), NULL, &ret);
    cl_mem width = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem height = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem halffilterSize = clCreateBuffer(*context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
    cl_mem output_img_mem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, img_size * sizeof(float), NULL, &ret);
    // init mem
    ret = clEnqueueWriteBuffer(command_queue, input_img_mem, CL_TRUE, 0, img_size * sizeof(float), inputImage, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, width, CL_TRUE, 0, sizeof(int), &imageWidth, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, height, CL_TRUE, 0, sizeof(int), &imageHeight, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, halffilterSize, CL_TRUE, 0, sizeof(int), &filterWidth, 0, NULL, NULL);
    // create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &ret);
    // set kernel arg
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_img_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&width);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&height);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&halffilterSize);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&output_img_mem);
    // run kernel
    size_t global_item_size = img_size;
    size_t local_item_size = 1;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    // copy the result back to host
    ret = clEnqueueReadBuffer(command_queue, output_img_mem, CL_TRUE, 0, img_size * sizeof(float), outputImage, 0, NULL, NULL);
}