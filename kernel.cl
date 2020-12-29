__kernel void convolution(__global const float *input_img, __global const float *filter,  __global int *imageWidth, __global int *imageHeight, __global int *filterWidth, __global float *output_img) 
{
    int id = get_global_id(0);
    int i = id / *imageWidth;
    int j = id % *imageWidth;
    int k,l;
    float sum;
    int halffilterSize = *filterWidth / 2;
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (i + k >= 0 && i + k < *imageHeight &&
                j + l >= 0 && j + l < *imageWidth)
            {
                sum += input_img[(i + k) * *imageWidth + j + l] *
                        filter[(k + halffilterSize) * *filterWidth +
                                l + halffilterSize];
            }
        }
    }
    if( id < 5)
        printf("%f\n",input_img[id]);
    /*if( id == 0){
        printf("%d %d %d\n",*imageWidth,*imageHeight,*filterWidth);
        for (k = -halffilterSize; k <= halffilterSize; k++)
        {
            for (l = -halffilterSize; l <= halffilterSize; l++)
            {
                printf("%f ",filter[(k + halffilterSize) * *filterWidth + l + halffilterSize]);
            }
            printf("\n");
        }
    }*/
    output_img[i * *imageWidth + j] = sum;
}
