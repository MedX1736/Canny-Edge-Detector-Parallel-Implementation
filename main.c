#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#define KERNEL_SIZE 1
#define OFFSET 1
#define thresh_max 90
#define thresh_min 30
#include "omp.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

//application du filtre gaussien sur tous les pixel de l'image
void gaussian_blur(const uint8_t *input_image, int height, int width,
                   uint8_t *output_image) {

  const double kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  const int kernel_sum = 16;

  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      double output_intensity = 0;
      int kernel_index = 0;
      int pixel_index = col + (row * width);
      for (int krow = -OFFSET; krow <= OFFSET; krow++) {
        for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
          output_intensity += input_image[pixel_index + (kcol + (krow * width))] * kernel[kernel_index];
          kernel_index++;
        }
      }
      output_image[pixel_index] = (uint8_t)(output_intensity / kernel_sum);
    }
  }
}

//calcul des gradients avec leurs magnitudes et directions
void gradient_magnitude_direction(const uint8_t *input_image, int height,
                                  int width, double *magnitude,
                                  uint8_t *direction) {
  const int8_t Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const int8_t Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      double grad_x_sum = 0.0;
      double grad_y_sum = 0.0;
      int kernel_index = 0;
      int pixel_index = col + (row * width);

      for (int krow = -OFFSET; krow <= OFFSET; krow++) {
        for (int kcol = -OFFSET; kcol <= OFFSET; kcol++) {
          grad_x_sum += input_image[pixel_index + (kcol + (krow * width))] * Gx[kernel_index];
          grad_y_sum += input_image[pixel_index + (kcol + (krow * width))] * Gy[kernel_index];
          kernel_index++;
        }
      }

      int pixel_direction = 0;

      if (grad_x_sum == 0.0 || grad_y_sum == 0.0) {
        magnitude[pixel_index] = 0;
      } else {
        magnitude[pixel_index] = ((sqrt((grad_x_sum * grad_x_sum) + (grad_y_sum * grad_y_sum))));
        double theta = atan2(grad_y_sum, grad_x_sum);
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
        else
          printf("Wrong direction: %f",theta);
      }
      direction[pixel_index] = (uint8_t)pixel_direction;
    }
  }
}


//suppression des non maximum pour garder les bords les plus forts
void non_max_suppression(double *gradient_magnitude,
                         uint8_t *gradient_direction, int height, int width,
                         double *output_image) {

  memcpy(output_image, gradient_magnitude, width * height * sizeof(double));
  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      int pixel_index = col + (row * width);

      // unconditionally suppress border pixels
      if (row == OFFSET || col == OFFSET || col == width - OFFSET - 1 ||
          row == height - OFFSET - 1) {
        output_image[pixel_index] = 0;
        continue;
      }

      switch (gradient_direction[pixel_index]) {
      case 1:
        if (gradient_magnitude[pixel_index-1]>=gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index+1]>gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      case 2:
        if (gradient_magnitude[pixel_index -(width-1)]>=gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index+(width-1)]>gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      case 3:
        if (gradient_magnitude[pixel_index-(width)]>=gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index + (width)]>gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      case 4:
        if (gradient_magnitude[pixel_index-(width+1)]>=gradient_magnitude[pixel_index] ||
            gradient_magnitude[pixel_index+(width+1)]>gradient_magnitude[pixel_index])
          output_image[pixel_index] = 0;
        break;
      default:
        output_image[pixel_index] = 0;
        break;
      }
    }
  }
}

//classer les bords de l'image en forts faibles et non bords
void thresholding(double *suppressed_image, int height, int width,
                  int high_threshold, int low_threshold,
                  uint8_t *output_image) {

  for (int col = 0; col < width; col++) {
    for (int row = 0; row < height; row++) {
      int pixel_index = col + (row * width);
      if (suppressed_image[pixel_index] > high_threshold)
        output_image[pixel_index] = 255; // Strong edge
      else if (suppressed_image[pixel_index] > low_threshold)
        output_image[pixel_index] = 100; // Weak edge
      else
        output_image[pixel_index] = 0; // Not an edge
    }
  }
}

//traiter les bords faibles pour decider s'ils sont bords ou pas
void hysteresis(uint8_t *input_image, int height, int width,
                uint8_t *output_image) {

  memcpy(output_image, input_image, width * height * sizeof(uint8_t));
  for (int col = OFFSET; col < width - OFFSET; col++) {
    for (int row = OFFSET; row < height - OFFSET; row++) {
      int pixel_index = col + (row * width);
      if (output_image[pixel_index] == 100) {
        if (output_image[pixel_index - 1] == 255 ||
            output_image[pixel_index + 1] == 255 ||
            output_image[pixel_index - width] == 255 ||
            output_image[pixel_index + width] == 255 ||
            output_image[pixel_index - width - 1] == 255 ||
            output_image[pixel_index - width + 1] == 255 ||
            output_image[pixel_index + width - 1] == 255 ||
            output_image[pixel_index + width + 1] == 255)

          output_image[pixel_index] = 255;
        else
          output_image[pixel_index] = 0;
      }
    }
  }
}

void canny_edge_detect(const uint8_t *input_image, int height, int width,
                       int high_threshold, int low_threshold,int size,
                       uint8_t *output_image) {


      unsigned char *blur_img = malloc (size);
     double *magnitude = malloc(size*sizeof(double*));
     unsigned char *direction = malloc(size);
    double *nms = malloc(size *sizeof(double*));
     unsigned char *double_thresh =  malloc(size);

      gaussian_blur(input_image, height, width, blur_img);
      gradient_magnitude_direction(blur_img, height, width, magnitude, direction);
      non_max_suppression(magnitude, direction, height, width, nms);
      thresholding(nms, height, width, high_threshold, low_threshold, double_thresh);
      hysteresis(double_thresh, height, width, output_image);
}


int main(int argc, char **argv)
{
      int width, height, channels , gray_channels;
      clock_t t1, t2;
      char imagefilename[100] ;
      float timeInS;
      printf("Enter file Name : ");
      scanf("%s",imagefilename);

      //charger l'image a traiter 
      unsigned char *img = stbi_load(imagefilename, &width, &height, &channels, 0);
      if(img == NULL) {
          printf("Error in loading the image\n");
          exit(1);
      }else{
        printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

      }
      if (channels == 4 ) gray_channels = 2 ;
      else gray_channels = 1;
      int img_size = width * height * channels ;
      int img_size_gray = width * height * gray_channels ;
      unsigned char *gray_img = malloc (img_size_gray) ;

      if(gray_img == NULL) {
          printf("Error in loading the gray image\n");
          exit(1);
      }

    //transformer l'image qui est sur 3 coleurs RVB ou 4 en une seule couleur
    for(int i = 0 , j = 0 ; i != img_size; i += channels , j+= gray_channels) {
         gray_img[j] = (uint8_t)((img[i+0] + img[i+1] + img[i+2])/3.0);
         if(channels == 4)  gray_img[j+1] = img[i+3];
     }

     unsigned char *edge_img = malloc (img_size_gray) ;
     t1 = clock();
     //appliquer la fonction sur l'image grise
     canny_edge_detect(gray_img,height,width,thresh_max,thresh_min,img_size_gray,edge_img);
     t2 = clock();
     //enregistrer l'image
     stbi_write_jpg("result.jpeg", width, height, gray_channels, edge_img, 100);
     printf("Wrote the edge image with a width of %dpx, a height of %dpx and %d channels\n", width, height, gray_channels);
     timeInS = (float)(t2-t1)/CLOCKS_PER_SEC;
     printf("\n\ntemps d'execution = %f\n", timeInS);
    //Liberer les ressources
     stbi_image_free(img);


}
