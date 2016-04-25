#include <stdio.h>
#include <stdlib.h>
/* A simple program to make arrays! */





void  print_arr(int* arr, int length)
  {
    printf("[");
    for (int i=0; i<length; i++)
	printf("%d,",arr[i]);
    printf("]\n");
  }



int main()
{


  int arr[4] = {1,3,5,7};

  print_arr(arr,4);

  getchar();

  return 0;
}
