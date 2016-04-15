#include <stdio.h>

int main()
{

  float add_number;
  printf("Input a number please: ");
  scanf("%f",&add_number);
  printf("You entered: %f\n",add_number);
  printf("Your number lives at '%p' lane\n",&add_number);
  
  int a=5;
  int b=6;
  int c = a*b;
  printf("%d*%d = %d\n",a,b,c);


  printf("%d means false and %d means true\n",2==1,2==2);


  getchar();

  int arr[5];

  int y;
  for(y=0; y<5; y++)
    {
      arr[y]=y;
    }
  printf("arr = [%d,%d,%d,%d]\n",arr[0],arr[1],arr[2],arr[3]);


  getchar();
  return 0;
}
