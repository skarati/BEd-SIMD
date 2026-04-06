#include<immintrin.h>
#include<string.h>

#define byte unsigned char
#define vec __m256i
const vec mask64 = {-1, 0, -1, 0}; 			//LSB_64
const vec mask1 = {1, 0, 1, 0};
const vec par_d = {0x86e5,0,1,0}; 			// 1 | d
const vec cons[2] = {{0,0,0,0},{-1,-1,-1,-1}};
const vec zero = {0,0,0,0};
const vec one = {1,0,0,0}; 
const vec baseP[5] = {{1, 0, 0x127188efe8975d5d, 0},{0, 0, 0x489a0206502ddb9c, 0},{0, 0, 0x02c9f6207db53dcc,0},{0, 0, 0x0cd56e74ed0bebf8, 0},{0, 0, 1, 0}};
const vec _2baseP[5] = {{1, 0, 0x25a7df13a359d7cc, 0},{0, 0, 0xdfb1592b03c59a34, 0},{0, 0, 0xbbf36ae7202ce7b9, 0},{0, 0, 0x67d1330bc22fa122, 0},{0, 0, 0, 0}};

      


#define vadd(C,A,B) {C = _mm256_xor_si256(A, B);}
#define vsub(C,A,B) {C = _mm256_sub_epi64(A, B);}
#define vmult(C,A,B) {C = _mm256_clmulepi64_epi128(A, B, 0x00);}
#define vshiftl64(C,A,B) {C=_mm256_slli_epi64(A,B);}
#define vshiftr(C,A,B) {C=_mm256_srli_si256(A,B);}
#define vshiftr64(C,A,B) {C=_mm256_srli_epi64(A,B);}
#define vand(C,A,B) {C=_mm256_and_si256(A,B);}

#define permute_11(C, A, B) {C = _mm256_permute2x128_si256(A, B, 0x13);}    // A-hi | B-hi
#define permute_10(C, A, B) {C = _mm256_permute2x128_si256(A, B, 0x12);}    // A-hi | B-l0
#define permute_01(C, A, B) {C = _mm256_permute2x128_si256(A, B, 0x03);}    // A-lo | B-hi
#define permute_00(C, A, B) {C = _mm256_permute2x128_si256(A, B, 0x02);}    // A-lo | B-lo

#define vec128 __m128i
#define load(C,A) {C = _mm_loadu_si128((__m128i*)A);}




//Function to print a array consisting some specific number of bytes 
void printBytes(byte *data, int num){
    for (int i=num-1; i>=0; i--)
        printf("%02x",data[i]);
    printf("\n");
}    

//Function to pack two different numbers src1, src2 as (src1 | src2) into four 256 bits registers
void pack64(byte *src1, byte *src2, __m256i *dest){
    __m128i dest1[5], dest2[5];
    byte temp[8] = {0};
    
    temp[0] = src1[32];
    //Storing 'src1' in lower 64 parts of four 128 bits registers
    dest1[0] = _mm_loadu_si128((__m128i*)src1);
    dest1[1] = _mm_loadu_si128((__m128i*)(src1 + 8));
    dest1[2] = _mm_loadu_si128((__m128i*)(src1 + 16));
    dest1[3] = _mm_loadu_si128((__m128i*)(src1 + 24));
    dest1[4] = _mm_loadu_si128((__m128i*)(temp));

    temp[0] = src2[32];
    //Storing 'src2' in lower 64 parts of four 128 bits registers
    dest2[0] = _mm_loadu_si128((__m128i*)src2);
    dest2[1] = _mm_loadu_si128((__m128i*)(src2 + 8));
    dest2[2] = _mm_loadu_si128((__m128i*)(src2 + 16));
    dest2[3] = _mm_loadu_si128((__m128i*)(src2 + 24));
    dest2[4] = _mm_loadu_si128((__m128i*)(temp));


    //parallel alignment of src1 and src2 in 256 bits register respectively
    dest[0] = _mm256_set_m128i(dest2[0], dest1[0]);	vand(dest[0],dest[0],mask64);
    dest[1] = _mm256_set_m128i(dest2[1], dest1[1]); 	vand(dest[1],dest[1],mask64);
    dest[2] = _mm256_set_m128i(dest2[2], dest1[2]);	vand(dest[2],dest[2],mask64);
    dest[3] = _mm256_set_m128i(dest2[3], dest1[3]);	vand(dest[3],dest[3],mask64);
    dest[4] = _mm256_set_m128i(dest2[4], dest1[4]);	vand(dest[4],dest[4],mask64);
}
void unpack64(__m256i *src, byte *dest1, byte *dest2){
    // Store the 256-bit register into a temporary array
    int	i;
    byte temp[32];
    for(i = 0; i < 4; i++){
        _mm256_storeu_si256((__m256i*)temp, src[i]);
        // Extract the lower 64 bits (first 8 bytes) and third 64 bits (third 8 bytes)
    	memcpy(dest1 + i*8, temp, 8);        // Lower 64 bits are at offset 0
    	memcpy(dest2 + i*8, temp + 16, 8);   // Third 64 bits are at offset 16
    } 
    _mm256_storeu_si256((__m256i*)temp, src[4]);
    // Extract the lower 64 bits (first 8 bytes) and third 64 bits (third 8 bytes)
    memcpy(dest1 + i*8, temp, 1);        // Lower 64 bits are at offset 0
    memcpy(dest2 + i*8, temp + 16, 1);   // Third 64 bits are at offset 16
}

vec t00, t01, t10, t11, t02, t20, t03, t30, t12, t21, t04, t40, t22, t13, t31, t23, t32, t14, t41, t24, t42, t33, t34, t43, t44;
vec t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, c1, c2;

vec t, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30;
vec u1v1, u2v2, u3v3, u0u1, v0v1, u0u2, v0v2, u1u2, v1v2, u0u3, v0v3, u0u4, v0v4, u1u3, v1v3, u2u3, v2v3, u1u4, v1v4, u2u4, v2v4, u3u4, v3v4 ;

//Pure Karatsuba multiplication
#define vmult5(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,w0,w1,w2,w3,w4,w5,w6,w7,w8) { \
               vmult(w0, u0, v0); \
               vmult(u1v1, u1, v1); \
               vmult(u2v2, u2, v2); \
               vmult(u3v3, u3, v3); \
               \
               vand(w8, u4, v4); \
               vadd(u0u1, u0, u1); \
               vadd(v0v1, v0, v1); \
               vmult(t0, u0u1, v0v1) \
               \
               vadd(t1, w0, u1v1); \
               vadd(w1, t0, t1); \
               vadd(u0u2, u0, u2); \
               vadd(v0v2, v0, v2); \
               \
               vmult(t2, u0u2, v0v2); \
               vadd(t3, w0, u2v2); \
               vadd(t4, t2, t3); \
               vadd(w2, t4, u1v1); \
               \
               vadd(u1u2, u1, u2); \
               vadd(v1v2, v1, v2); \
               vmult(t5, u1u2, v1v2); \
               vadd(t6, u1v1, u2v2); \
               \
               vadd(t7, t5, t6); \
               vadd(u0u3, u0, u3) ; \
               vadd(v0v3, v0, v3); \
               vmult(t8, u0u3, v0v3); \
               \
               vadd(t9, w0, u3v3); \
               vadd(t10, t8, t9); \
               vadd(w3, t7, t10); \
               vadd(u0u4, u0, u4); \
               \
               vadd(v0v4, v0, v4); \
               vmult(t11, u0u4, v0v4); \
               vadd(t12, w0, w8); \
               vadd(t13, t11, t12); \
               \
               vadd(u1u3, u1, u3); \
               vadd(v1v3, v1, v3); \
               vmult(t14, u1u3, v1v3); \
               vadd(t15, u1v1, u3v3); \
               \
               vadd(t16, t14, t15); \
               vadd(t17, t13, t16); \
               vadd(w4, t17, u2v2); \
               vadd(u2u3, u2, u3); \
               \
               vadd(v2v3, v2, v3); \
               vmult(t18, u2u3, v2v3); \
               vadd(t19, u2v2, u3v3); \
               vadd(t20, t18, t19); \
               \
               vadd(u1u4, u1, u4); \
               vadd(v1v4, v1, v4); \
               vmult(t21, u1u4, v1v4); \
               vadd(t22, u1v1, w8); \
               \
               vadd(t23, t21, t22); \
               vadd(w5, t20, t23); \
               vadd(u2u4, u2, u4); \
               vadd(v2v4, v2, v4); \
               \
               vmult(t24, u2u4, v2v4); \
               vadd(t25, u2v2, w8); \
               vadd(t26, t24, t25); \
               vadd(w6, u3v3, t26); \
               \
               vadd(u3u4, u3, u4); \
               vadd(v3v4, v3, v4); \
               vmult(t27, u3u4, v3v4); \
               vadd(t28, u3v3, w8); \
               \
               vadd(w7, t27, t28); \
} 


#define vsq5(u0,u1,u2,u3,u4,w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
	vmult(w0, u0, u0); \
	vmult(w2, u1, u1); \
	vmult(w4, u2, u2); \
	vmult(w6, u3, u3); \
	vand(w8, u4, u4); \
}


#define vmulC(u0,u1,u2,u3,u4,c,wf0,wf1,wf2,wf3,wf4){ \
	vmult(wf0, u0, c); \
	vmult(wf1, u1, c); \
	vmult(wf2, u2, c); \
	vmult(wf3, u3, c); \
	vmult(wf4, u4, c); \
}

#define add5(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,w0,w1,w2,w3,w4){ \
	vadd(w0, u0, v0); \
	vadd(w1, u1, v1); \
	vadd(w2, u2, v2); \
	vadd(w3, u3, v3); \
	vadd(w4, u4, v4); \
}

#define and5(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,w0,w1,w2,w3,w4){ \
	vand(w0, u0, v0); \
	vand(w1, u1, v1); \
	vand(w2, u2, v2); \
	vand(w3, u3, v3); \
	vand(w4, u4, v4); \
}

#define add9(u0,u1,u2,u3,u4,u5,u6,u7,u8,v0,v1,v2,v3,v4,v5,v6,v7,v8,w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
	vadd(w0, u0, v0); \
	vadd(w1, u1, v1); \
	vadd(w2, u2, v2); \
	vadd(w3, u3, v3); \
	vadd(w4, u4, v4); \
	vadd(w5, u5, v5); \
	vadd(w6, u6, v6); \
	vadd(w7, u7, v7); \
	vadd(w8, u8, v8); \
}


#define expandM(w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
    	vshiftr(t1, w0, 8); vshiftr(t2, w1, 8); vshiftr(t3, w2, 8); vshiftr(t4, w3, 8); \
    	vshiftr(t5, w4, 8); vshiftr(t6, w5, 8); vshiftr(t7, w6, 8); \
    	\
    	vadd(w1, w1, t1); vadd(w2, w2, t2); vadd(w3, w3, t3); vadd(w4, w4, t4); \
    	vadd(w5, w5, t5); vadd(w6, w6, t6); vadd(w7, w7, t7); \
    	\
    	vand(w0, w0, mask64); vand(w1, w1, mask64); vand(w2, w2, mask64); vand(w3, w3, mask64); \
    	vand(w4, w4, mask64); vand(w5, w5, mask64); vand(w6, w6, mask64); vand(w7, w7, mask64); \
}


#define expandS(w0,w1,w2,w3,w4,w5,w6,w7,w8){ \
    	vshiftr(w1, w0, 8); vshiftr(w3, w2, 8); vshiftr(w5, w4, 8); vshiftr(w7, w6, 8); \
    	vand(w0, w0, mask64); vand(w2, w2, mask64); vand(w4, w4, mask64); vand(w6, w6, mask64);\
}

vec tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
vec tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, tmp17;
vec a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, b0;


#define foldM(w0,w1,w2,w3,w4,w5,w6,w7,w8,wf0,wf1,wf2,wf3,wf4){\
        vshiftl64(tmp0, w5, 63); vshiftl64(tmp1, w5, 11); vshiftr64(tmp2, w5, 1); vshiftr64(tmp3, w5, 53); \
        vshiftl64(tmp4, w6, 63); vshiftl64(tmp5, w6, 11); vshiftr64(tmp6, w6, 1); vshiftr64(tmp7, w6, 53); \
        vshiftl64(tmp8, w7, 63); vshiftl64(tmp9, w7, 11); vshiftr64(tmp10, w7, 1); vshiftr64(tmp11, w7, 53); \
        vshiftl64(tmp12, w8, 63); vshiftl64(tmp13, w8, 11); vshiftr64(tmp14, w8, 1); \
        \
        vadd(wf0, w0, tmp0); vadd(a0, w1, tmp1); vadd(a1, a0, tmp2); vadd(wf1, a1, tmp4); \
        vadd(a2, w2, tmp3); vadd(a3, a2, tmp5); vadd(a4, a3, tmp6); vadd(wf2, a4, tmp8); \
        vadd(a5, w3, tmp7); vadd(a6, a5, tmp9); vadd(a7, a6, tmp10); vadd(wf3, a7, tmp12); \
        vadd(a8, w4, tmp11); vadd(a9, a8, tmp13); vadd(wf4, a9, tmp14); \
} 

#define foldS(w0,w1,w2,w3,w4,w5,w6,w7,w8,wf0,wf1,wf2,wf3,wf4){\
        vshiftl64(tmp0, w5, 63); vshiftl64(tmp1, w5, 11); vshiftr64(tmp2, w5, 1); vshiftr64(tmp3, w5, 53); \
        vshiftl64(tmp4, w6, 63); vshiftl64(tmp5, w6, 11); vshiftr64(tmp6, w6, 1); vshiftr64(tmp7, w6, 53); \
        vshiftl64(tmp8, w7, 63); vshiftl64(tmp9, w7, 11); vshiftr64(tmp10, w7, 1); vshiftr64(tmp11, w7, 53); \
        vshiftl64(tmp12, w8, 63); vshiftl64(tmp13, w8, 11); \
        \
        vadd(wf0, w0, tmp0); vadd(a0, w1, tmp1); vadd(a1, a0, tmp2); vadd(wf1, a1, tmp4); \
        vadd(a2, w2, tmp3); vadd(a3, a2, tmp5); vadd(a4, a3, tmp6); vadd(wf2, a4, tmp8); \
        vadd(a5, w3, tmp7); vadd(a6, a5, tmp9); vadd(a7, a6, tmp10); vadd(wf3, a7, tmp12); \
        vadd(a8, w4, tmp11); vadd(wf4, a8, tmp13); \
}

#define reduce(wf0,wf1,wf2,wf3,wf4){\
        vshiftr64(tmp15, wf4, 1); vand(wf4, wf4, mask1); vshiftl64(tmp16, tmp15, 12); vshiftr64(tmp17, tmp15, 52); \
        vadd(b0, wf0, tmp15); vadd(wf0, b0, tmp16); vadd(wf1, wf1, tmp17); \
}

#define reduceC(u0,u1,u2,u3,u4,wf0,wf1,wf2,wf3,wf4){\
        vand(t0, u0, mask64); \
        vand(t1, u1, mask64); \
        vand(t2, u2, mask64); \
        vand(t3, u3, mask64); \
        \
        vshiftr(tmp0, u0, 8); \
        vshiftr(tmp1, u1, 8); \
        vshiftr(tmp2, u2, 8); \
        vshiftr(tmp3, u3, 8); \
        \
        vadd(wf1, t1, tmp0); \
        vadd(wf2, t2, tmp1); \
        vadd(wf3, t3, tmp2); \
        vadd(u4, u4, tmp3); \
        \
        vshiftr64(t4, u4, 1); \
        vand(wf4, u4, mask1); \
        vshiftl64(t5, t4, 12); \
        vshiftr64(t6, t4, 52); \
        \
        vadd(t7, t4, t5); \
        vadd(wf0, t0, t7); \
        vadd(wf1, wf1, t6); \
}


#define swap(u0,u1,u2,u3,u4,v0,v1,v2,v3,v4,B){ \
	vadd(t0, u0, v0); vadd(t1, u1, v1); vadd(t2, u2, v2); vadd(t3, u3, v3); vadd(t4, u4, v4); \
	vand(t0, t0, B); vand(t1, t1, B); vand(t2, t2, B); vand(t3, t3, B); vand(t4, t4, B); \
	vadd(u0, u0, t0); vadd(u1, u1, t1); vadd(u2, u2, t2); vadd(u3, u3, t3); vadd(u4, u4, t4); \
	vadd(v0, v0, t0); vadd(v1, v1, t1); vadd(v2, v2, t2); vadd(v3, v3, t3); vadd(v4, v4, t4); \
}



//Ladder-step
void ladderStep(vec *w2z2, vec *w3z3, vec *w1z1) {
            
	vec temp1[5], temp2[5], temp3[5], A[9], B[9], E1[5], E2[5], res[5];
	vec B1 [5], B2[5], B3[5], C[5], C1[5], C2[5], C3[5], F1[5], F2[5], F3[5], G[5], G1[5], G2[5], G3[5], H1[5], H2[5], H3[5];
	vec w[9],wf[5];
	
		
	// B2 = z2 | 0
	permute_00(B2[0], w2z2[0], zero);
	permute_00(B2[1], w2z2[1], zero);
	permute_00(B2[2], w2z2[2], zero);
	permute_00(B2[3], w2z2[3], zero);
	permute_00(B2[4], w2z2[4], zero);
	
	
	// C1 = z3 | 0
    	permute_00(C1[0], w3z3[0], zero);
	permute_00(C1[1], w3z3[1], zero);
	permute_00(C1[2], w3z3[2], zero);
	permute_00(C1[3], w3z3[3], zero);
	permute_00(C1[4], w3z3[4], zero);
	
		
	//C2 = w3 | z2
	permute_10(C2[0], w3z3[0], w2z2[0]);
	permute_10(C2[1], w3z3[1], w2z2[1]);
	permute_10(C2[2], w3z3[2], w2z2[2]);
	permute_10(C2[3], w3z3[3], w2z2[3]);
	permute_10(C2[4], w3z3[4], w2z2[4]);
	
		
	
	// temp1 = w2+z2 | z2
	add5(w2z2[0], w2z2[1], w2z2[2], w2z2[3], w2z2[4],
	       B2[0], B2[1], B2[2], B2[3], B2[4],
	     temp1[0], temp1[1], temp1[2], temp1[3], temp1[4]);
	  
	//C3 = w3+z3 | z2
	add5(C2[0], C2[1], C2[2], C2[3], C2[4],
	     C1[0], C1[1], C1[2], C1[3], C1[4],
	     C3[0], C3[1], C3[2], C3[3], C3[4]);
	     
	//F1 = w3*(w3+z3) | z2z3
	vmult5(C3[0], C3[1], C3[2], C3[3], C3[4],
	       w3z3[0], w3z3[1], w3z3[2], w3z3[3], w3z3[4],
	       w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
	  
	       
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    	
    	foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
    	      F1[0],F1[1],F1[2],F1[3],F1[4]);
    	      
    	reduce(F1[0],F1[1],F1[2],F1[3],F1[4]);
    	
    	
	        
	//C = w2*(w2+z2) | z2^2
	vmult5(temp1[0], temp1[1], temp1[2], temp1[3], temp1[4],
	           w2z2[0], w2z2[1], w2z2[2], w2z2[3], w2z2[4],
      	        w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
	       
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    	
    	foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
    	      C[0],C[1],C[2],C[3],C[4]);
    	      
    	reduce(C[0],C[1],C[2],C[3],C[4]);                          // C must be unchanged for upcoming dbl
    	
    	
    	
    	
    	// diffadd : Q = P + Q
	
    	//F2 = w2(w2+z2) | z2z3
    	permute_10(F2[0], C[0], F1[0]);
	permute_10(F2[1], C[1], F1[1]);
	permute_10(F2[2], C[2], F1[2]);
	permute_10(F2[3], C[3], F1[3]);
	permute_10(F2[4], C[4], F1[4]);
	
	
	//G1 = F1 * F2 = v | (z2z3)^2
	vmult5(F1[0], F1[1], F1[2], F1[3], F1[4],
	       F2[0], F2[1], F2[2], F2[3], F2[4],
	       w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
	  
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    	
    	foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
    	      G1[0],G1[1],G1[2],G1[3],G1[4]);
    	      
    	reduce(G1[0],G1[1],G1[2],G1[3],G1[4]);
    	
    
    	
    	/*note: v = c*w3*(w3+z3) where c = w2*(w2+z2)*/
    	
    	//G2 = v | d(z2z3)^2
    	vmulC(G1[0], G1[1], G1[2], G1[3], G1[4],
    	      par_d, wf[0], wf[1], wf[2], wf[3], wf[4]);
    	
    	reduceC(wf[0],wf[1],wf[2],wf[3],wf[4],
    	        G2[0],G2[1],G2[2],G2[3],G2[4]);    
    	
  
    	//F3 = 0 | v
    	permute_11(F3[0], zero, G1[0]);
	permute_11(F3[1], zero, G1[1]);
	permute_11(F3[2], zero, G1[2]);
	permute_11(F3[3], zero, G1[3]);
	permute_11(F3[4], zero, G1[4]);
	
	
    	//G3 = G2 + F1 = v | v + d(z2z3)^2 = v | z5
    	add5(G2[0], G2[1], G2[2], G2[3], G2[4],
    	     F3[0], F3[1], F3[2], F3[3], F3[4],
    	     G3[0], G3[1], G3[2], G3[3], G3[4]);   // G3 = v | z5 should not be changed
    	     
  
    	//H1 = z5 | 0
    	permute_00(H1[0], G3[0], zero);
	permute_00(H1[1], G3[1], zero);
	permute_00(H1[2], G3[2], zero);
	permute_00(H1[3], G3[3], zero);
	permute_00(H1[4], G3[4], zero);
	
	
	//H3 = H1 * wz = w1z5 | 0
	vmult5(H1[0], H1[1], H1[2], H1[3], H1[4],
	       w1z1[0], w1z1[1], w1z1[2], w1z1[3], w1z1[4],
	       w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
	  
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    	
    	foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
    	      H3[0],H3[1],H3[2],H3[3],H3[4]);
    	      
    	reduce(H3[0],H3[1],H3[2],H3[3],H3[4]);
    	
    	
    	// w3z3 = v + w1z5 | z5
    	add5(G3[0], G3[1], G3[2], G3[3], G3[4],
    	     H3[0], H3[1], H3[2], H3[3], H3[4],
    	     w3z3[0], w3z3[1], w3z3[2], w3z3[3], w3z3[4]);  // Q = P + Q 
    	     
    	 
    	
    	// dbl : P = 2P
    	
    	//temp3 = c^2 | z2^4 where c = w2(w2+z2)
    	vsq5(C[0], C[1], C[2], C[3], C[4],
    	     w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	     
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	  
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6], w[7],w[8],
    	      temp3[0],temp3[1],temp3[2],temp3[3],temp3[4]);
    	      
    	reduce(temp3[0],temp3[1],temp3[2],temp3[3],temp3[4]);
    	
    	 
    	//E1 = 0 | c^2
    	permute_01(E1[0], zero, temp3[0]);
	permute_01(E1[1], zero, temp3[1]);
	permute_01(E1[2], zero, temp3[2]);
	permute_01(E1[3], zero, temp3[3]);
	permute_01(E1[4], zero, temp3[4]);
	
	
	
	
	//E2 = c^2 | d*z2^4
    	vmulC(temp3[0], temp3[1], temp3[2], temp3[3], temp3[4],
    	      par_d, wf[0], wf[1], wf[2], wf[3], wf[4]);
    	      
    	reduceC(wf[0],wf[1],wf[2],wf[3],wf[4],
    	        E2[0],E2[1],E2[2],E2[3],E2[4]);
    	
    	
    	
    	//w2 | z2 = w4 | z4 = c^2 | c^2 + d*z2^4
    	add5(E1[0], E1[1], E1[2], E1[3], E1[4],
    	     E2[0], E2[1], E2[2], E2[3], E2[4],
    	     w2z2[0], w2z2[1], w2z2[2], w2z2[3], w2z2[4]);     
    	
    
}


// Function for field inversion
void invert(vec *in, vec *op){
    vec t[5];
    vec x2[5],x3[5],x4[5],x7[5],x_6_1[5],x_12_1[5],x_24_1[5],x_25_1[5],x_50_1[5],x_100_1[5],x_125_1[5],x_250_1[5],x_256_1[5];
    vec temp[5];
    
    vec w[9],wf[5];
    
    // 2
    vsq5(in[0], in[1], in[2], in[3],in[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x2[0],x2[1],x2[2],x2[3],x2[4]);
    reduce(x2[0],x2[1],x2[2],x2[3],x2[4]);
    
    // 3
    vmult5(in[0],in[1],in[2],in[3],in[4], 
           x2[0],x2[1],x2[2],x2[3],x2[4],
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x3[0],x3[1],x3[2],x3[3],x3[4]);
    
    reduce(x3[0],x3[1],x3[2],x3[3],x3[4]);
    
    // 4
    vsq5(x2[0],x2[1],x2[2],x2[3],x2[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x4[0],x4[1],x4[2],x4[3],x4[4]);
    
    reduce(x4[0],x4[1],x4[2],x4[3],x4[4]);
    
    // 7
    vmult5(x3[0],x3[1],x3[2],x3[3],x3[4], 
           x4[0],x4[1],x4[2],x4[3],x4[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x7[0],x7[1],x7[2],x7[3],x7[4]);
    
    reduce(x7[0],x7[1],x7[2],x7[3],x7[4]);
    
    for(int i=0; i<5;i++){temp[i] = x7[i];}
    
    // 2^6-8
    for(int i=0;i<3;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    
    // 2^6-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x7[0],x7[1],x7[2],x7[3],x7[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4]);
    
    reduce(x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4]);
    
    
    for(int i=0; i<5;i++){temp[i] = x_6_1[i];}
    // 2^12-2^6
    for(int i=0;i<6;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    // 2^12-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3],x_12_1[4]);
    
    reduce(x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3],x_12_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_12_1[i];}
    //2^24-2^12
    for(int i=0;i<12;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^24-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3],x_12_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_24_1[0],x_24_1[1],x_24_1[2],x_24_1[3],x_24_1[4]);
    
    reduce(x_24_1[0],x_24_1[1],x_24_1[2],x_24_1[3],x_24_1[4]);
    
    //2^25-2
    vsq5(x_24_1[0], x_24_1[1], x_24_1[2], x_24_1[3],x_24_1[4],w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    reduce(x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    //2^25-1
    vmult5(in[0],in[1],in[2],in[3],in[4], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    reduce(x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4]);
    
    
    for(int i=0; i<5;i++){temp[i] = x_25_1[i];}
    //2^50-2^25
    for(int i=0;i<25;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^50-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3],x_50_1[4]);
    
    reduce(x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3],x_50_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_50_1[i];}
    //2^100-2^50
    for(int i=0;i<50;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    // 2^100-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3],x_50_1[4], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_100_1[0],x_100_1[1],x_100_1[2],x_100_1[3],x_100_1[4]);
    
    reduce(x_100_1[0],x_100_1[1],x_100_1[2],x_100_1[3],x_100_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_100_1[i];}
    // 2^125-2^25
    for(int i=0;i<25;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^125-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3],x_25_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3],x_125_1[4]);
    
    reduce(x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3],x_125_1[4]);
    
    for(int i=0; i<5;i++){temp[i] = x_125_1[i];}
    //2^250-2^125
    for(int i=0;i<125;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    //2^250-1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3],x_125_1[4], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_250_1[0],x_250_1[1],x_250_1[2],x_250_1[3],x_250_1[4]);
    
    reduce(x_250_1[0],x_250_1[1],x_250_1[2],x_250_1[3],x_250_1[4]);
    
    
    for(int i=0; i<5;i++){temp[i] = x_250_1[i];}
    // 2^256 - 2^6
    for(int i=0;i<6;i++){
        vsq5(temp[0],temp[1],temp[2],temp[3],temp[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    	foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], temp[0],temp[1],temp[2],temp[3],temp[4]);
    	
    	reduce(temp[0],temp[1],temp[2],temp[3],temp[4]);
    }
    
    // 2^256 -1
    vmult5(temp[0],temp[1],temp[2],temp[3],temp[4], 
           x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3],x_6_1[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], x_256_1[0],x_256_1[1],x_256_1[2],x_256_1[3],x_256_1[4]);
    
    reduce(x_256_1[0],x_256_1[1],x_256_1[2],x_256_1[3],x_256_1[4]);
    
    // 2^257 -2
    vsq5(x_256_1[0],x_256_1[1],x_256_1[2],x_256_1[3],x_256_1[4], w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], op[0],op[1],op[2],op[3],op[4]);
    
    reduce(op[0],op[1],op[2],op[3],op[4]);    
    
}



//Function to clamp the scalar
void clamp(byte n[33]){
	n[0] = n[0] & 0xfc;
	
	n[32] = n[32] | 0x01;
	
}



//------------------SCALAR MULTIPLICATION-------------------------------------------------------



// Function to compute scalar multiplication using left to right Montgomery ladder
void scalarMult_fixed_base( byte *n, byte *nw, byte *nz ){  
    
    clamp(n);
       
    vec nPwz[5], w[9], wf[5];
    vec wz[5];
                              
    vec S[5], R[5];
         
    S[0] = baseP[0];	S[1] = baseP[1];	S[2] = baseP[2];	S[3] = baseP[3];	S[4] = baseP[4];
    wz[0] = baseP[0];	wz[1] = baseP[1];	wz[2] = baseP[2];	wz[3] = baseP[3];	wz[4] = baseP[4];
    R[0] = _2baseP[0];	R[1] = _2baseP[1];	R[2] = _2baseP[2];	R[3] = _2baseP[3];	R[4] = _2baseP[4];
        
    byte pb = 0, b, ni;

    int j;
    for(int i = 31; i >= 0 ; i--){
        j = 7;
        for(; j >= 0 ; j--){
            ni = (n[i] >> j) & 1;
            b = pb ^ ni;
            
            swap(S[0],S[1],S[2],S[3],S[4],
                 R[0],R[1],R[2],R[3],R[4],cons[b]);
            
            ladderStep(S, R, wz);
            
            pb = ni;
        }
        
    }
    swap(S[0],S[1],S[2],S[3],S[4],
                 R[0],R[1],R[2],R[3],R[4],cons[pb]);;
    
    
    vec W[5],Z[5];
    
    invert(S,Z);
    
    //0 | nw
    permute_11(W[0], S[0], S[0]);
    permute_11(W[1], S[1], S[1]);
    permute_11(W[2], S[2], S[2]);
    permute_11(W[3], S[3], S[3]);
    permute_11(W[4], S[4], S[4]);
    
  
    
    vmult5(W[0],W[1],W[2],W[3],W[4],
           Z[0],Z[1],Z[2],Z[3],Z[4], 
           w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
           
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], 
          nPwz[0],nPwz[1],nPwz[2],nPwz[3],nPwz[4]);
          
    reduce(nPwz[0],nPwz[1],nPwz[2],nPwz[3],nPwz[4]);
                  
   
    unpack64(nPwz, nw, nz);
    
}


// Function to compute scalar multiplication using left to right Montgomery ladder
void scalarMult_var_base( byte *w1, byte *n, byte *nw, byte *nz ){  
    
    clamp(n);
    vec Pxz[5], nP[5], S[5], R[5];
    byte z1[33] = {0x01,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    vec w[9],wf[5];
    pack64(z1, w1, S); //S = w1 | z1
    vec wz[5];

       
    wz[0] = S[0];
    wz[1] = S[1];
    wz[2] = S[2];
    wz[3] = S[3];
    wz[4] = S[4];
    
   
    vec B1[5], B2[5], C[5], C1[5], C2[5], E1[5], E2[5], G[5], G2[5], temp1[5], temp2[5], temp3[5];
        
    byte pb = 0, b, ni;
    
    
    // B2 = z1 | 0
    permute_00(B2[0], S[0], zero);
    permute_00(B2[1], S[1], zero);
    permute_00(B2[2], S[2], zero);
    permute_00(B2[3], S[3], zero);
    permute_00(B2[4], S[4], zero);
    
    // temp1 = w1 + z1 | z1
    add5(S[0], S[1], S[2], S[3], S[4],
	 B2[0], B2[1], B2[2], B2[3], B2[4],
       temp1[0], temp1[1], temp1[2], temp1[3], temp1[4]);
    
    
    	
    //C = w1*(w1+z1) | z1^2
    vmult5(S[0], S[1], S[2], S[3], S[4],
           temp1[0], temp1[1], temp1[2], temp1[3], temp1[4], 
      	   w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]);
	       
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);     
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
          C[0],C[1],C[2],C[3],C[4]);
          
    reduce(C[0],C[1],C[2],C[3],C[4]);      

    //temp3 = c^2 | z1^4 where c = w1(w1+z1)
    vsq5(C[0], C[1], C[2], C[3], C[4],
    	 w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    	     
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
      
    foldS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],
          temp3[0],temp3[1],temp3[2],temp3[3],temp3[4]);
          
    reduce(temp3[0],temp3[1],temp3[2],temp3[3],temp3[4]);
    	
    	
    //E1 = 0 | c^2
    permute_01(E1[0], zero, temp3[0]);
    permute_01(E1[1], zero, temp3[1]);
    permute_01(E1[2], zero, temp3[2]);
    permute_01(E1[3], zero, temp3[3]);
    permute_01(E1[4], zero, temp3[4]);
	
    
    //E2 = c^2 | d*z2^4
    vmulC(temp3[0], temp3[1], temp3[2], temp3[3], temp3[4],
          par_d, wf[0], wf[1], wf[2], wf[3], wf[4]);
          
    reduceC(wf[0],wf[1],wf[2],wf[3],wf[4],
            E2[0],E2[1],E2[2],E2[3],E2[4]);
    
    // R = c^2 | c^2 + d*z1^4 = 2P
    add5(E1[0],E1[1],E1[2],E1[3],E1[4],
    	 E2[0],E2[1],E2[2],E2[3],E2[4],
    	 R[0],R[1],R[2],R[3],R[4]);
    	 
    
    int j;
    for(int i = 31; i >= 0 ; i--){
        j=7;
        for(; j >= 0 ; j--){
            ni = (n[i] >> j) & 1;
            b = pb ^ ni;
            
            swap(S[0],S[1],S[2],S[3],S[4],
                 R[0],R[1],R[2],R[3],R[4],cons[b]);
           
            ladderStep(S, R, wz);
            
            pb = ni;
        }
    
    }
    swap(S[0],S[1],S[2],S[3],S[4],
         R[0],R[1],R[2],R[3],R[4],cons[pb]);
    
    vec W[5],Z[5];
    
    invert(S,Z);
    
    permute_11(W[0], S[0], S[0]);
    permute_11(W[1], S[1], S[1]);
    permute_11(W[2], S[2], S[2]);
    permute_11(W[3], S[3], S[3]);
    permute_11(W[4], S[4], S[4]);
    
    vmult5(W[0],W[1],W[2],W[3],W[4], 
           Z[0],Z[1],Z[2],Z[3],Z[4], 
	   w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
	   
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8]);
    
    foldM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8], 
          nP[0], nP[1],nP[2],nP[3], nP[4]);
          
    reduce(nP[0], nP[1],nP[2],nP[3], nP[4]);
                  

    unpack64(nP, nw, nz);
    
}




