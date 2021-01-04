__kernel void convolution(__global const float *input_img, __global const float *filter,  __global int *imageWidth, __global int *imageHeight, __global int *filterWidth, __global float *output_img) 
{
    // int g_id_x = get_group_id(0);
    // int g_id_y = get_group_id(1);
    // int g_size_x = get_global_size(0);
    // int g_size_y = get_global_size(1);
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int i = id_x ;
    int j = id_y;
    int k,l;
    float sum = 0;
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
    /*if( id < 5)
        printf("ID %d : %f\n",sum);*/
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
