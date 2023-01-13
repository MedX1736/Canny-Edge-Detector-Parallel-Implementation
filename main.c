#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

int main(void) {
      int width, height, channels , gray_channels;

      unsigned char *img = stbi_load("bird.jpeg", &width, &height, &channels, 0);

      if (channels == 4 ) gray_channels = 2 ;
      else gray_channels = 1;

      int img_size = width * height * channels ;
      int img_size_gray = width * height * gray_channels ;

      unsigned char *gray_img = malloc ( img_size_gray) ;

      if(img == NULL) {
          printf("Error in loading the image\n");
          exit(1);
      }

      if(gray_img == NULL) {
          printf("Error in loading the gray image\n");
          exit(1);
      }


     /*  for(unsigned char  *p = img, *pg = gray_img; p != img + img_size; p += channels, pg += gray_channels) {
         *pg = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
         if(channels == 4)  *(pg + 1) = *(p + 3);
     } */

    for(int i = 0 , j = 0 ; i != img_size; i += channels , j+= gray_channels) {
         gray_img[j] = (uint8_t)((img[i+0] + img[i+1] + img[i+2])/3.0);
         if(channels == 4)  gray_img[j+1] = img[i+3];
     }

     printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

     stbi_write_jpg("bird_gray.jpeg", width, height, gray_channels, gray_img, 100);

     printf("Writed the gray image with a width of %dpx, a height of %dpx and %d channels\n", width, height, gray_channels);




    stbi_image_free(img);
 }
