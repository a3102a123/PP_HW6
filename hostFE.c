#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void check_err(cl_int ret){
    if(!ret)
        print("Error code: %d\n",ret);
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status,ret;
    int img_size = imageWidth * imageHeight;
    int filterSize = filterWidth * filterWidth;
    // create command queue
    printf("Check command queue\n");
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &ret);
    check_err(ret);
    // create kernel mem
    printf("Check mem\n");
    cl_mem input_img_mem = clCreateBuffer(*context, CL_MEM_READ_WRITE, img_size * sizeof(float), NULL, &ret);
    check_err(ret);
    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_READ_WRITE, filterSize * sizeof(float), NULL, &ret);
    check_err(ret);
    cl_mem width = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    check_err(ret);
    cl_mem height = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    check_err(ret);
    cl_mem filterWIdth_mem = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(int), NULL, &ret);
    check_err(ret);
    cl_mem output_img_mem = clCreateBuffer(*context, CL_MEM_READ_WRITE, img_size * sizeof(float), NULL, &ret);
    check_err(ret);
    // init mem
    printf("check mem cpy\n");
    ret = clEnqueueWriteBuffer(command_queue, input_img_mem, CL_TRUE, 0, img_size * sizeof(float), inputImage, 0, NULL, NULL);
    check_err(ret);
    ret = clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
    check_err(ret);
    ret = clEnqueueWriteBuffer(command_queue, width, CL_TRUE, 0, sizeof(int), &imageWidth, 0, NULL, NULL);
    check_err(ret);
    ret = clEnqueueWriteBuffer(command_queue, height, CL_TRUE, 0, sizeof(int), &imageHeight, 0, NULL, NULL);
    check_err(ret);
    ret = clEnqueueWriteBuffer(command_queue, filterWIdth_mem, CL_TRUE, 0, sizeof(int), &filterWidth, 0, NULL, NULL);
    check_err(ret);
    // create kernel
    printf("check kernel create\n");
    cl_kernel kernel = clCreateKernel(*program, "convolution", &ret);
    check_err(ret);
    // set kernel arg
    printf("check kernel arg\n");
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_img_mem);
    check_err(ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem);
    check_err(ret);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&width);
    check_err(ret);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&height);
    check_err(ret);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&filterSize);
    check_err(ret);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&output_img_mem);
    check_err(ret);
    // run kernel
    printf("check kernel runing\n");
    size_t global_item_size = img_size;
    size_t local_item_size = 1;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    check_err(ret);
    // copy the result back to host
    printf("check result cpy\n");
    ret = clEnqueueReadBuffer(command_queue, output_img_mem, CL_TRUE, 0, img_size * sizeof(float), outputImage, 0, NULL, NULL);
    check_err(ret);
    printf("check free\n");
    ret = clFlush(command_queue);
    check_err(ret);
    ret = clFinish(command_queue);
    check_err(ret);
    ret = clReleaseKernel(kernel);
    check_err(ret);
    ret = clReleaseMemObject(input_img_mem);
    check_err(ret);
    ret = clReleaseMemObject(filter_mem);
    check_err(ret);
    ret = clReleaseMemObject(width);
    check_err(ret);
    ret = clReleaseMemObject(height);
    check_err(ret);
    ret = clReleaseMemObject(filterWIdth_mem);
    check_err(ret);
    ret = clReleaseMemObject(output_img_mem);
    check_err(ret);
    ret = clReleaseCommandQueue(command_queue);
    check_err(ret);
}