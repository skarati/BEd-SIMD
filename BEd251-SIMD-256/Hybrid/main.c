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

#include "BEd251.h"
#include "measurement.h"


int main(){
    
    byte n[32] = {0xb4, 0xc9, 0x3c, 0xfd, 0xda, 0x5a, 0xc3, 0x82, 
                  0x85, 0xbd, 0x72, 0x17, 0x15, 0xfd, 0x63, 0x3f, 
                  0xa3, 0xcc, 0xd9, 0x62, 0xa1, 0x53, 0x29, 0x6e, 
                  0xf2, 0xbf, 0x0a, 0xc1, 0xfc, 0x3d, 0x31, 0x05};
                  
    byte w[32] = {0xE1, 0xF7, 0x4A, 0x7B, 0xC6, 0xB9, 0x55, 0x26,
                  0xDB, 0x9B, 0x40, 0xDD, 0x41, 0x3A, 0x81, 0x1D,
                  0xB6, 0xC7, 0x91, 0x73, 0xC7, 0x3E, 0x4D, 0x41,
                  0x62, 0xA0, 0x42, 0x84, 0xCB, 0xFE, 0x46, 0x04};
                  
    byte z[32] = {0x01};
   
    
    printf("\nn = ");
    printBytes(n, 32);
    printf("\n");    
    
    byte wn[32], zn[32];
    scalarMult_fixed_base(n, wn, zn);
    printf("\n-----------------Scalar  Multiplication Fixed Base---------------\n");
    printf("\nnP = (wn,zn)-\nwn: ");
    printBytes(wn,32);
    
    
    
    byte nPw[32], nPz[32];
    printf("\n----------------Scalar Multiplication Variable Base--------------\n");
    scalarMult_var_base( w, n, nPw, nPz );
    
    printf("\nnP = (wn,zn)-\nwn: ");
    printBytes(nPw,32);
    
    
    printf("\nComputing CPU-cycles. It will take some time. Please wait!\n\n");

    MEASURE({
		scalarMult_fixed_base(n, wn, zn);
	});
	printf("Total CPU cycles for fixed-base scalar multiplication Median: %.2f.\n", RDTSC_clk_median);
	printf("\n\n");
	
    
    printf("\nComputing CPU-cycles. It will take some time. Please wait!\n\n");
    
    MEASURE({
		scalarMult_var_base( w, n, nPw, nPz );
	});
	printf("Total CPU cycles for variable-base scalar multiplication Median: %.2f.\n", RDTSC_clk_median);
	printf("\n\n");
    
   
    
    return 0;
}
