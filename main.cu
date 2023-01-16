%%cu
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#define KERNEL_SIZE 1
#define OFFSET 1
#define thresh_max 90
#define thresh_min 30
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

__constant__ double gaussian_kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
__constant__ int gaussian_kernel_sum = 16;
__constant__ int8_t Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int8_t Gy[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__global__ void gaussian_blur(const uint8_t *input_image, int height, int width, uint8_t *output_image)
{
    // calculate the row and col of this pixel in the image
    int row = threadIdx.x + blockIdx.x * (blockDim.x);
    int col = threadIdx.y + blockIdx.y * (blockDim.y);
    
    if (col < OFFSET || col >= width - OFFSET || row < OFFSET ||row >= height - OFFSET) return;
    
    double output_intensity = 0;
    int kernel_index = 0;
    int pixel_index = col + (row * width);
    for (int krow = -OFFSET; krow <= OFFSET; krow++)
    {
        for (int kcol = -OFFSET; kcol <= OFFSET; kcol++)
        {
            
            output_intensity += input_image[pixel_index + (kcol + (krow * width))] * gaussian_kernel[kernel_index];
            kernel_index++;
        }
    }
    output_image[pixel_index] = (uint8_t)(output_intensity / gaussian_kernel_sum);
}

__global__ void gradient_magnitude_direction(const uint8_t *input_image,int height, int width,double *magnitude,uint8_t *direction) {
  
  // calculate the row and col of this pixel in the image
   int row = threadIdx.x + blockIdx.x * (blockDim.x);
   int col = threadIdx.y + blockIdx.y * (blockDim.y);
  // if pixel is outside the given offset don't do anything
  if (col < OFFSET || col >= width - OFFSET || row < OFFSET || row >= height - OFFSET) return;

  double grad_x_sum = 0.0;
  double grad_y_sum = 0.0;
  int kernel_index = 0;
  int pixel_index = col + (row * width);

  for (int krow = -OFFSET; krow <= OFFSET; krow++) {
    for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
      grad_x_sum +=
          input_image[pixel_index + (kcol + (krow * width))] * Gx[kernel_index];
      grad_y_sum +=
          input_image[pixel_index + (kcol + (krow * width))] * Gy[kernel_index];
      kernel_index++;
    }
  }

  int pixel_direction = 0;

  if (grad_x_sum == 0.0 || grad_y_sum == 0.0) {
    magnitude[pixel_index] = 0;
  } else {
    magnitude[pixel_index] =
        ((std::sqrt((grad_x_sum * grad_x_sum) + (grad_y_sum * grad_y_sum))));
    double theta = std::atan2(grad_y_sum, grad_x_sum);
    theta = theta * (360.0 / (2.0 * M_PI));

    if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) ||
        (theta >= 157.5))
      pixel_direction = 1; // horizontal -
    else if ((theta > 22.5 && theta <= 67.5) ||
             (theta > -157.5 && theta <= -112.5))
      pixel_direction = 2; // north-east -> south-west/
    else if ((theta > 67.5 && theta <= 112.5) ||
             (theta >= -112.5 && theta < -67.5))
      pixel_direction = 3; // vertical |
    else if ((theta >= -67.5 && theta < -22.5) ||
             (theta > 112.5 && theta < 157.5))
      pixel_direction = 4; // north-west -> south-east \'
  }
  direction[pixel_index] = (uint8_t)pixel_direction;
}

__global__ void non_max_suppression(double *gradient_magnitude,uint8_t *gradient_direction, int height,int width, double *output_image) {
    // calculate the row and col of this pixel in the image
   int row = threadIdx.x + blockIdx.x * (blockDim.x);
   int col = threadIdx.y + blockIdx.y * (blockDim.y);
  // if pixel is outside the given offset don't do anything
  if (col < OFFSET || col >= width - OFFSET || row < OFFSET ||
      row >= height - OFFSET)
    return;
  int pixel_index = col + (row * width);

  // unconditionally suppress border pixels
  if (row == OFFSET || col == OFFSET || col == width - OFFSET - 1 ||
      row == height - OFFSET - 1) {
    output_image[pixel_index] = 0;
    return;
  }

  switch (gradient_direction[pixel_index]) {
  case 1:
    if (gradient_magnitude[pixel_index - 1] >=
            gradient_magnitude[pixel_index] ||
        gradient_magnitude[pixel_index + 1] > gradient_magnitude[pixel_index])
      output_image[pixel_index] = 0;
    break;
  case 2:
    if (gradient_magnitude[pixel_index - (width - 1)] >=
            gradient_magnitude[pixel_index] ||
        gradient_magnitude[pixel_index + (width - 1)] >
            gradient_magnitude[pixel_index])
      output_image[pixel_index] = 0;
    break;
  case 3:
    if (gradient_magnitude[pixel_index - (width)] >=
            gradient_magnitude[pixel_index] ||
        gradient_magnitude[pixel_index + (width)] >
            gradient_magnitude[pixel_index])
      output_image[pixel_index] = 0;
    break;
  case 4:
    if (gradient_magnitude[pixel_index - (width + 1)] >=
            gradient_magnitude[pixel_index] ||
        gradient_magnitude[pixel_index + (width + 1)] >
            gradient_magnitude[pixel_index])
      output_image[pixel_index] = 0;
    break;
  default:
    output_image[pixel_index] = 0;
    break;
  }
}


__global__ void thresholding(double *suppressed_image, int height, int width,
                  int high_threshold, int low_threshold, uint8_t *output_image) {

      int row = threadIdx.x + blockIdx.x * (blockDim.x) -1 ;
      int col = threadIdx.y + blockIdx.y * (blockDim.y) -1 ;             
      int pixel_index = col + (row * width);

      if (pixel_index < 0 || pixel_index >= height * width)
      return;

      if (suppressed_image[pixel_index] > high_threshold)
        output_image[pixel_index] = 255; // Strong edge
      else if (suppressed_image[pixel_index] > low_threshold)
        output_image[pixel_index] = 100; // Weak edge
      else
        output_image[pixel_index] = 0; // Not an edge
}




__global__ void hysteresis(uint8_t *input_image, int height, int width) {

      int row = threadIdx.x + blockIdx.x * (blockDim.x);
      int col = threadIdx.y + blockIdx.y * (blockDim.y);
      int pixel_index = col + (row * width);
      if (input_image[pixel_index] == 100) {
        if (input_image[pixel_index - 1] == 255 ||
            input_image[pixel_index + 1] == 255 ||
            input_image[pixel_index - width] == 255 ||
            input_image[pixel_index + width] == 255 ||
            input_image[pixel_index - width - 1] == 255 ||
            input_image[pixel_index - width + 1] == 255 ||
            input_image[pixel_index + width - 1] == 255 ||
            input_image[pixel_index + width + 1] == 255)

          input_image[pixel_index] = 255;
        else
          input_image[pixel_index] = 0;
      }
}


void canny_edge_detect(const uint8_t *input_image, int height, int width, int high_threshold, int low_threshold, uint8_t *output_image)
{

    /*cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    cudaEventRecord(start_time, 0);*/

    int image_size = height * width;
    dim3 block(12,12);
    dim3 grid((height+block.y-1) / block.y , (width+block.x-1) / block.x);
 
    /******** Gaussian Blur  *******************/
 
    uint8_t *gaussian_blur_input;
    uint8_t *gaussian_blur_output;
    cudaMalloc((void **)&gaussian_blur_input, image_size );
    cudaMalloc((void **)&gaussian_blur_output, image_size );
    cudaMemcpy(gaussian_blur_input, input_image, image_size, cudaMemcpyHostToDevice);
    /**"Launching gaussian blur kernel..." **/
    gaussian_blur<<<grid,block>>>(gaussian_blur_input, height, width, gaussian_blur_output);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    // "Gaussian blur kernel finished."
    cudaFree(gaussian_blur_input);
  /******** Gaussian Blur Done*******************/

    /********  Gradient Magnitude and Direction  *****************/
    double *gradient_magnitude;
    uint8_t *gradient_direction;

    double *gradient_magnitude_output;
    uint8_t *gradient_direction_output;

    cudaMalloc((void **)&gradient_magnitude, image_size * sizeof(double*) );
    cudaMalloc((void **)&gradient_direction, image_size * sizeof(double*) );
    gradient_magnitude_direction<<<grid, block>>>(gaussian_blur_output, height, width, gradient_magnitude, gradient_direction);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    cudaFree(gaussian_blur_output);
  /********  Gradient Magnitude and Direction  Done*****************/

  /********  Non-max Suppression        *****************/
    double *nms_output;

    cudaMalloc((void **)&nms_output, image_size * sizeof(double*) );
    cudaMemcpy(nms_output, gradient_magnitude,image_size * sizeof(double*) ,cudaMemcpyDeviceToDevice);
    non_max_suppression<<<grid,block>>>( gradient_magnitude, gradient_direction, height, width, nms_output);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    //"Non-max suppression kernel finished." << endl;
    cudaFree(gradient_magnitude);
    cudaFree(gradient_direction);
  /********  Non-max Suppression DONE        *****************/
 

    //thresholding
    uint8_t *threshold_output;
    cudaMalloc((void **)&threshold_output, image_size );
    thresholding<<<grid,block>>>(nms_output, height, width, high_threshold, low_threshold, threshold_output);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    cudaFree(nms_output);

    //hysterisis 
    uint8_t *hysteresis_output;
    cudaMalloc((void **)&hysteresis_output, image_size );
    hysteresis<<<grid,block>>>(threshold_output, height, width);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    cudaMemcpy(output_image, threshold_output, image_size, cudaMemcpyDeviceToHost);
    cudaFree(threshold_output);

    /*cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);
    float delta = 0;
    cudaEventElapsedTime(&delta, start_time, stop_time);
    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);
    printf("%lf\n", delta);*/
    

}

int main(int argc, char **argv)
{
      int width, height, channels, gray_channels;

      unsigned char *img = stbi_load("sky2.jpeg", &width, &height, &channels, 0);

      if (channels == 4)
          gray_channels = 2;
      else
          gray_channels = 1;

      int img_size = width * height * channels;
      int img_size_gray = width * height * gray_channels;

      unsigned char *gray_img = (unsigned char*) malloc(img_size_gray);

      if (img == NULL)
      {
          printf("Error in loading the image\n");
          exit(1);
      }
      printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
      if (gray_img == NULL)
      {
          printf("Error in loading the gray image\n");
          exit(1);
      }

      for (int i = 0, j = 0; i != img_size; i += channels, j += gray_channels)
      {
          gray_img[j] = (uint8_t)((img[i + 0] + img[i + 1] + img[i + 2]) / 3.0);
          if (channels == 4)
              gray_img[j + 1] = img[i + 3];
      }

      stbi_write_jpg("sky_gray.jpeg", width, height, gray_channels, gray_img, 100);
      printf("Wrote the gray image with a width of %dpx, a height of %dpx and %d channels\n", width, height, gray_channels);

      unsigned char *edge_img = (unsigned char*) malloc(img_size_gray);
 
      canny_edge_detect(gray_img, height, width, thresh_max, thresh_min,edge_img);
 
      stbi_write_jpg("sky_edge.jpeg", width, height, gray_channels, edge_img, 100);
      printf("Wrote the edge image with a width of %dpx, a height of %dpx and %d channels\n", width, height, gray_channels);

      stbi_image_free(img);
}