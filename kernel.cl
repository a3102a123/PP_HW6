__kernel void convolution(__global const float *input_img, __global const float *filter,  __global int imageWidth, __global int imageHeight, __global int halffilterSize, __global float *output_img) 
{
    int i = get_global_id(0) % imageWidth;
    int j = get_global_id(0) / imageWidth;
    int k,l;
    float sum;
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (i + k >= 0 && i + k < imageHeight &&
                j + l >= 0 && j + l < imageWidth)
            {
                sum += inputImage[(i + k) * imageWidth + j + l] *
                        filter[(k + halffilterSize) * filterWidth +
                                l + halffilterSize];
            }
        }
    }
    output_img[i * imageWidth + j] = sum;
}
