#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "helper.h"

// This function reads in a text file and stores it as a char pointer
char *readSource(char *kernelPath)
{
    cl_int status;
    FILE *fp;
    char *source;
    long int size;

    printf("Program file is: %s\n", kernelPath);

    fp = fopen(kernelPath, "rb");
    if (!fp)
    {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if (status != 0)
    {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if (size < 0)
    {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char *)malloc(size + 1);

    int i;
    for (i = 0; i < size + 1; i++)
    {
        source[i] = '\0';
    }

    if (source == NULL)
    {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    fread(source, 1, size, fp);
    source[size] = '\0';

    return source;
}

void initCL(cl_device_id *device, cl_context *context, cl_program *program)
{
    // Set up the OpenCL environment
    cl_int status;

    // Discovery platform
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK(status, "clGetPlatformIDs");

    // Discover device
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, device, NULL);
    CHECK(status, "clGetDeviceIDs");

    // Create context
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)(platform), 0};
    *context = clCreateContext(props, 1, device, NULL, NULL, &status);
    CHECK(status, "clCreateContext");

    const char *source = readSource("kernel.cl");

    // Create a program object with source and build it
    *program = clCreateProgramWithSource(*context, 1, &source, NULL, NULL);
    CHECK(status, "clCreateProgramWithSource");
    status = clBuildProgram(*program, 1, device, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        char *buff_erro;
        cl_int errcode;
        size_t build_log_len;
        errcode = clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
        if (errcode) {
                    printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
                    exit(-1);
                }

            buff_erro = malloc(build_log_len);
            if (!buff_erro) {
                printf("malloc failed at line %d\n", __LINE__);
                exit(-2);
            }

            errcode = clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
            if (errcode) {
                printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
                exit(-3);
            }

            fprintf(stderr,"Build log: \n%s\n", buff_erro); //Be careful with  the fprint
            free(buff_erro);
            fprintf(stderr,"clBuildProgram failed\n");
            exit(EXIT_FAILURE);
    }
    CHECK(status, "clBuildProgram");
    return;
}

float *readFilter(const char *filename, int *filterWidth)
{
    printf("Reading filter data from %s\n", filename);

    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        printf("Could not open filter file\n");
        exit(-1);
    }

    fscanf(fp, "%d", filterWidth);

    float *filter = (float *)malloc(*filterWidth * *filterWidth * sizeof(int));

    float tmp;
    for (int i = 0; i < *filterWidth * *filterWidth; i++)
    {
        fscanf(fp, "%f", &tmp);
        filter[i] = tmp;
    }

    printf("Filter width: %d\n", *filterWidth);

    fclose(fp);
    return filter;
}
