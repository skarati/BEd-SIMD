#include <stdio.h>
#include <math.h>

#define align __attribute__ ((aligned (32)))
#if !defined (ALIGN32)
        # if defined (__GNUC__)
                # define ALIGN32 __attribute__ ( (aligned (32)))
        # else
                # define ALIGN32 __declspec (align (32))
        # endif
#endif

#include "BEd257.h"
#include "measurement.h"



int main(){
  
    //scalar Multiplication
    ALIGN32 byte w0[33]={0x5d, 0x5d, 0x97, 0xe8, 0xef, 0x88, 0x71, 0x12, 0x9c, 0xdb, 0x2d, 0x50, 0x06, 0x02, 0x9a, 0x48,
                         0xcc, 0x3d, 0xb5, 0x7d, 0x20, 0xf6, 0xc9, 0x02, 0xf8, 0xeb, 0x0b, 0xed, 0x74, 0x6e, 0xd5, 0x0c, 0x01};
 
    
    ALIGN32 byte z0[33]={0x01,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
                  
    byte n[33] = {0xb4, 0xc9, 0x3c, 0xfd, 0xda, 0x5a, 0xc3, 0x82,
                  0x85, 0xbd, 0x72, 0x17, 0x15, 0xfd, 0x63, 0x3f,
                  0xa3, 0xcc, 0xd9, 0x62, 0xa1, 0x53, 0x29, 0x6e,
                  0xf2, 0xbf, 0x0a, 0xc1, 0xfc, 0x3d, 0x31, 0x05, 0x01};
    
        
    byte wn[33], zn[33], wnn[33], znn[33];
    
    printf("\n------------------Scalar Multiplication Fixed Base---------------\n");
    
    scalarMult_fixed_base(n, wn, zn);
    
    printf("\n wn = ");
    printBytes(wn,33);
    
    printf("\n----------------Scalar Multiplication Variable Base--------------\n");
    
    scalarMult_var_base(w0, n, wnn, znn);
    
    printf("\n wnn = ");
    printBytes(wnn,33);
    
    printf("\nComputing CPU-cycles. It will take some time. Please wait!\n\n");

    MEASURE({
		scalarMult_fixed_base(n, wn, zn);
	});
	printf("Total CPU cycles for fixed-base scalar multiplication Median: %.2f.\n", RDTSC_clk_median);
	printf("\n\n");
	
    
    printf("\nComputing CPU-cycles. It will take some time. Please wait!\n\n");
    
    MEASURE({
		scalarMult_var_base(w0, n, wnn, znn);
	});
	printf("Total CPU cycles for variable-base scalar multiplication Median: %.2f.\n", RDTSC_clk_median);
	printf("\n\n");
	
 
    return 0;
}
