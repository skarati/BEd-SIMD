#include<immintrin.h>
#include<string.h>

#define byte unsigned char
#define vec __m256i
const vec mask64 = {-1, 0, -1, 0}; 					//LSB_64
const vec mask59 = {0x07FFFFFFFFFFFFFF, 0x0, 0x07FFFFFFFFFFFFFF, 0x0}; //LSB_59
const vec par_d = {0x0240100000000001,0,1,0}; 				// 1 | d
const vec cons[2] = {{0,0,0,0},{-1,-1,-1,-1}};
const vec zero = {0,0,0,0};
const vec one = {1,0,0,0};
const vec rt5 = {0x00000000000012a0, 0x0, 0x00000000000012a0, 0x0}; 
const vec r = {0x0000000000000095, 0x0, 0x0000000000000095, 0x0};
const vec baseP[4] = {{1, 0, 0x2655B9C67B4AF7E1, 0},{0, 0, 0x1D813A41DD409BDB, 0},{0, 0, 0x414D3EC77391C7B6,0},{0, 0, 0x446FECB8442A062, 0}};
const vec _2baseP[4] = {{1, 0, 0x40987066DBA86430, 0},{0, 0, 0x37E96944A14A6480, 0},{0, 0, 0xFCEE474F59FD6D71, 0},{0, 0, 0x46C6F7978923FB, 0}};





#define vadd(C,A,B) {C = _mm256_xor_si256(A, B);}
#define vmult(C,A,B) {C = _mm256_clmulepi64_epi128(A, B, 0x00);}
#define vshiftl(C,A,B) {C=_mm256_slli_epi64(A,B);}
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
    vec128 dest1[4], dest2[4];
    //Storing 'src1' in lower 64 parts of four 128 bits registers
    dest1[0] = _mm_loadu_si128((__m128i*)src1);
    dest1[1] = _mm_loadu_si128((__m128i*)(src1 + 8));
    dest1[2] = _mm_loadu_si128((__m128i*)(src1 + 16));
    dest1[3] = _mm_loadu_si128((__m128i*)(src1 + 24));

    //Storing 'src2' in lower 64 parts of four 128 bits registers
    dest2[0] = _mm_loadu_si128((__m128i*)src2);
    dest2[1] = _mm_loadu_si128((__m128i*)(src2 + 8));
    dest2[2] = _mm_loadu_si128((__m128i*)(src2 + 16));
    dest2[3] = _mm_loadu_si128((__m128i*)(src2 + 24));
    

    //parallel alignment of src1 and src2 in 256 bits register respectively
    dest[0] = _mm256_set_m128i(dest2[0], dest1[0]);   vand(dest[0], dest[0], mask64);
    dest[1] = _mm256_set_m128i(dest2[1], dest1[1]);   vand(dest[1], dest[1], mask64);
    dest[2] = _mm256_set_m128i(dest2[2], dest1[2]);   vand(dest[2], dest[2], mask64);
    dest[3] = _mm256_set_m128i(dest2[3], dest1[3]);   vand(dest[3], dest[3], mask64);
}

void unpack64(__m256i *src, byte *dest1, byte *dest2){
    for(int i = 0; i < 4; i++){
        // Store the 256-bit register into a temporary array
        byte temp[32];
        _mm256_storeu_si256((__m256i*)temp, src[i]);
        
        // Extract the lower 64 bits (first 8 bytes) and third 64 bits (third 8 bytes)
    	memcpy(dest1 + i*8, temp, 8);        // Lower 64 bits are at offset 0
    	memcpy(dest2 + i*8, temp + 16, 8);   // Third 64 bits are at offset 16
    } 
}


vec t, t0, t1, t2, t3, t4, t5, t6, t7;
vec u0u1,u2u3,u1u3,u0u2,v0v1,v1v3,v2v3,v0v2,u0123,v0123;


#define vmult4(u0,u1,u2,u3,v0,v1,v2,v3,w0,w1,w2,w3,w4,w5,w6){ \
	vmult(w0, u0, v0); \
	vmult(w2, u1, v1);\
	vadd(u0u1, u0, u1);\
	vadd(v0v1, v0, v1);\
	vmult(t0, u0u1, v0v1);\
	vadd(t1, w0, w2);\
        vadd(w1, t0, t1);\
	\
	vmult(w4, u2, v2); \
	vmult(w6, u3, v3); \
	vadd(u2u3, u2, u3);\
	vadd(v2v3, v2, v3);\
	vmult(t0, u2u3, v2v3);\
	vadd(t1, w4, w6);\
	vadd(w5, t0, t1);\
	\
	vadd(u1u3, u1, u3);\
	vadd(v1v3, v1, v3);\
	vadd(u0u2, u0, u2);\
	vadd(v0v2, v0, v2);\
	\
	vmult(t0, u0u2, v0v2);\
	vmult(t2, u1u3, v1v3);\
	vadd(u0123, u0u2, u1u3);\
	vadd(v0123, v0v2, v1v3);\
	vmult(t3, u0123, v0123);\
	vadd(t4, t0, t2);\
	vadd(t1, t3, t4);\
	\
	vadd(t5, w0, w4);\
	vadd(t0, t0, t5);\
	vadd(t6, w1, w5);\
	vadd(w3, t1, t6);\
	vadd(t7, w2, w6);\
	vadd(t2, t2, t7);\
	\
	vadd(w2, t0, w2);\
	vadd(w4, t2, w4);\
}


#define vsq4(u0,u1,u2,u3,w0,w1,w2,w3,w4,w5,w6){ \
	vmult(w0, u0, u0); \
	vmult(w2, u1, u1); \
	vmult(w4, u2, u2); \
	vmult(w6, u3, u3); \
}

#define vmulC(u0,u1,u2,u3,c,wf0,wf1,wf2,wf3){ \
	vmult(wf0, u0, c); \
	vmult(wf1, u1, c); \
	vmult(wf2, u2, c); \
	vmult(wf3, u3, c); \
}

#define add4(u0,u1,u2,u3,v0,v1,v2,v3,w0,w1,w2,w3){ \
	vadd(w0, u0, v0); \
	vadd(w1, u1, v1); \
	vadd(w2, u2, v2); \
	vadd(w3, u3, v3); \
}

#define and4(u0,u1,u2,u3,v0,v1,v2,v3,w0,w1,w2,w3){ \
	vand(w0, u0, v0); \
	vand(w1, u1, v1); \
	vand(w2, u2, v2); \
	vand(w3, u3, v3); \
}


#define add7(u0,u1,u2,u3,u4,u5,u6,v0,v1,v2,v3,v4,v5,v6,w0,w1,w2,w3,w4,w5,w6){ \
	vadd(w0, u0, v0); \
	vadd(w1, u1, v1); \
	vadd(w2, u2, v2); \
	vadd(w3, u3, v3); \
	vadd(w4, u4, v4); \
	vadd(w5, u5, v5); \
	vadd(w6, u6, v6); \
}


#define expandM(w0,w1,w2,w3,w4,w5,w6,w7){ \
    	vshiftr(t1, w0, 8); vshiftr(t2, w1, 8); vshiftr(t3, w2, 8); vshiftr(t4, w3, 8); \
    	vshiftr(t5, w4, 8); vshiftr(t6, w5, 8); vshiftr(w7, w6, 8); \
    	\
    	vadd(w1, w1, t1); vadd(w2, w2, t2); vadd(w3, w3, t3); vadd(w4, w4, t4); \
    	vadd(w5, w5, t5); vadd(w6, w6, t6); \
    	\
    	vand(w0, w0, mask64); vand(w1, w1, mask64); vand(w2, w2, mask64); vand(w3, w3, mask64); \
    	vand(w4, w4, mask64); vand(w5, w5, mask64); vand(w6, w6, mask64); \
}


#define expandS(w0,w1,w2,w3,w4,w5,w6,w7){ \
    	vshiftr(w1, w0, 8); vshiftr(w3, w2, 8); vshiftr(w5, w4, 8); vshiftr(w7, w6, 8); \
    	vand(w0, w0, mask64); vand(w2, w2, mask64); vand(w4, w4, mask64);vand(w6, w6, mask64);\
}


#define foldm(w0,w1,w2,w3,w4,w5,w6,w7,wf0,wf1,wf2,wf3){ \
	vmult(t1, w4, rt5); vmult(t2, w5, rt5); vmult(t3, w6, rt5); vmult(t4, w7, rt5); \
	vadd(wf0, w0, t1); vadd(wf1, w1, t2); vadd(wf2, w2, t3); vadd(wf3, w3, t4); \
}



#define reducem(wf0,wf1,wf2,wf3,u0,u1,u2,u3){\
	vshiftr(t1, wf0, 8); vshiftr(t2, wf1, 8); vshiftr(t3, wf2, 8); \
	vand(u0, wf0, mask64); vand(u1, wf1, mask64); vand(u2, wf2, mask64); vand(u3, wf3, mask59); \
	vadd(u1, u1, t1); vadd(u2, u2, t2); vadd(u3, u3, t3); \
	vshiftr(t4, wf3, 7); vshiftr64(t4, t4, 3); \
	\
	vmult(t4, t4, r); vadd(u0, u0, t4); \
    	vshiftr(t1, u0, 8); vand(u0, u0, mask64); vadd(u1, u1, t1); \
}


#define swap(u0,u1,u2,u3,v0,v1,v2,v3,B){ \
	vadd(t0, u0, v0); vadd(t1, u1, v1); vadd(t2, u2, v2); vadd(t3, u3, v3); \
	vand(t0, t0, B); vand(t1, t1, B); vand(t2, t2, B); vand(t3, t3, B); \
	vadd(u0, u0, t0); vadd(u1, u1, t1); vadd(u2, u2, t2); vadd(u3, u3, t3); \
	vadd(v0, v0, t0); vadd(v1, v1, t1); vadd(v2, v2, t2); vadd(v3, v3, t3); \
}


       






//Ladder-step 
void ladderStep(vec *w2z2, vec *w3z3, vec *w1z1) {
            
	vec temp1[4], temp2[4], temp3[4], A[7], B[7], E1[4], E2[4], res[4];
	vec B1 [4], B2[4], B3[4], C[4], C1[4], C2[4], C3[4], F1[4], F2[4], F3[4], G[4], G1[4], G2[4], G3[4], H1[4], H2[4], H3[4];
	vec w[8],wf[4];
	
	
	
	
	// B2 = z2 | 0
	permute_00(B2[0], w2z2[0], zero);
	permute_00(B2[1], w2z2[1], zero);
	permute_00(B2[2], w2z2[2], zero);
	permute_00(B2[3], w2z2[3], zero);
	
	
	
	// C1 = z3 | 0
    	permute_00(C1[0], w3z3[0], zero);
	permute_00(C1[1], w3z3[1], zero);
	permute_00(C1[2], w3z3[2], zero);
	permute_00(C1[3], w3z3[3], zero);
	
	
	
	//C2 = w3 | z2
	permute_10(C2[0], w3z3[0], w2z2[0]);
	permute_10(C2[1], w3z3[1], w2z2[1]);
	permute_10(C2[2], w3z3[2], w2z2[2]);
	permute_10(C2[3], w3z3[3], w2z2[3]);
	
		
	
	// temp1 = w2+z2 | z2
	add4(w2z2[0], w2z2[1], w2z2[2], w2z2[3],
	       B2[0], B2[1], B2[2], B2[3],
	   temp1[0], temp1[1], temp1[2], temp1[3]);
	  
	//C3 = w3+z3 | z2
	add4(C2[0], C2[1], C2[2], C2[3],
	     C1[0], C1[1], C1[2], C1[3],
	     C3[0], C3[1], C3[2], C3[3]);
	     
	//F1 = w3*(w3+z3) | z2z3
	vmult4(C3[0], C3[1], C3[2], C3[3],
	       w3z3[0], w3z3[1], w3z3[2], w3z3[3],
	  w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
	  
	       
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);     
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],wf[0],wf[1],wf[2],wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3],F1[0],F1[1],F1[2],F1[3]);
    	
    	
	        
	//C = w2*(w2+z2) | z2^2
	vmult4(temp1[0], temp1[1], temp1[2], temp1[3],
	           w2z2[0], w2z2[1], w2z2[2], w2z2[3],
      	        w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
	       
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);     
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],wf[0],wf[1],wf[2],wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3],C[0],C[1],C[2],C[3]);      // C must be unchanged for upcoming dbl
    	
    	
    	
    	
    	// diffadd : Q = P + Q
	
    	//F2 = w2(w2+z2) | z2z3
    	permute_10(F2[0], C[0], F1[0]);
	permute_10(F2[1], C[1], F1[1]);
	permute_10(F2[2], C[2], F1[2]);
	permute_10(F2[3], C[3], F1[3]);
	
	
	//G1 = F1 * F2 = v | (z2z3)^2
	vmult4(F1[0], F1[1], F1[2], F1[3],
	       F2[0], F2[1], F2[2], F2[3],
	  w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
	  
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);     
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],wf[0],wf[1],wf[2],wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3],G1[0],G1[1],G1[2],G1[3]);
    	
    
    	
    	/*note: v = c*w3*(w3+z3) where c = w2*(w2+z2)*/
    	
    	//G2 = v | d(z2z3)^2
    	vmulC(G1[0], G1[1], G1[2], G1[3], par_d,
    	      wf[0], wf[1], wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3],G2[0],G2[1],G2[2],G2[3]);    
    	
  
    	//F3 = 0 | v
    	permute_11(F3[0], zero, G1[0]);
	permute_11(F3[1], zero, G1[1]);
	permute_11(F3[2], zero, G1[2]);
	permute_11(F3[3], zero, G1[3]);
	
	
    	//G3 = G2 + F1 = v | v + d(z2z3)^2 = v | z5
    	add4(G2[0], G2[1], G2[2], G2[3],
    	     F3[0], F3[1], F3[2], F3[3],
    	     G3[0], G3[1], G3[2], G3[3]);   // G3 = v | z5 should not be changed
    	     
  
    	//H1 = z5 | 0
    	permute_00(H1[0], G3[0], zero);
	permute_00(H1[1], G3[1], zero);
	permute_00(H1[2], G3[2], zero);
	permute_00(H1[3], G3[3], zero);
	
	
	//H3 = H1 * wz = w1z5 | 0
	vmult4(H1[0], H1[1], H1[2], H1[3],
	       w1z1[0], w1z1[1], w1z1[2], w1z1[3],
	  w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
	  
	expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);     
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],wf[0],wf[1],wf[2],wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3],H3[0],H3[1],H3[2],H3[3]);
    	
    	
    	// w3z3 = v + w1z5 | z5
    	add4(G3[0], G3[1], G3[2], G3[3],
    	     H3[0], H3[1], H3[2], H3[3],
    	     w3z3[0], w3z3[1], w3z3[2], w3z3[3]);  // Q = P + Q 
    	     
    	 
    	
    	// dbl : P = 2P
    	
    	//temp3 = c^2 | z2^4 where c = w2(w2+z2)
    	vsq4(C[0], C[1], C[2], C[3],
    	     w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	     
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],wf[0],wf[1],wf[2],wf[3]);  
    	reducem(wf[0],wf[1],wf[2],wf[3],temp3[0],temp3[1],temp3[2],temp3[3]);
    	
    	  
  
    	//E1 = 0 | c^2
    	permute_01(E1[0], zero, temp3[0]);
	permute_01(E1[1], zero, temp3[1]);
	permute_01(E1[2], zero, temp3[2]);
	permute_01(E1[3], zero, temp3[3]);
	
	
	
	
	//E2 = c^2 | d*z2^4
    	vmulC(temp3[0], temp3[1], temp3[2], temp3[3], par_d,
    	      wf[0], wf[1], wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3],E2[0],E2[1],E2[2],E2[3]);
    	
    	
    	
    	//w2 | z2 = w4 | z4 = c^2 | c^2 + d*z2^4
    	add4(E1[0], E1[1], E1[2], E1[3],
    	     E2[0], E2[1], E2[2], E2[3],
    	     w2z2[0], w2z2[1], w2z2[2], w2z2[3]);     
    	
    
}

     


//Function for inversion
void invert(vec *in, vec *op){
    vec t[4];
    vec x2[4],x3[4],x4[4],x7[4],x_6_1[4],x_12_1[4],x_24_1[4],x_25_1[4],x_50_1[4],x_100_1[4],x_125_1[4],x_250_1[4];
    vec temp[4];
    
    vec w[8],wf[4];
    
    // 2
    vsq4(in[0], in[1], in[2], in[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x2[0],x2[1],x2[2],x2[3]);
    
    // 3
    vmult4(in[0],in[1],in[2],in[3], 
           x2[0],x2[1],x2[2],x2[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x3[0],x3[1],x3[2],x3[3]);
    
    // 4
    vsq4(x2[0],x2[1],x2[2],x2[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x4[0],x4[1],x4[2],x4[3]);
    
    // 7
    vmult4(x3[0],x3[1],x3[2],x3[3], 
           x4[0],x4[1],x4[2],x4[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x7[0],x7[1],x7[2],x7[3]);
    
    for(int i=0; i<4;i++){temp[i] = x7[i];}
    
    // 2^6-8
    for(int i=0;i<3;i++){
        vsq4(temp[0],temp[1],temp[2],temp[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], temp[0],temp[1],temp[2],temp[3]);
    }
    
    // 2^6-1
    vmult4(temp[0],temp[1],temp[2],temp[3], 
           x7[0],x7[1],x7[2],x7[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3]);
    
    
    for(int i=0; i<4;i++){temp[i] = x_6_1[i];}
    // 2^12-2^6
    for(int i=0;i<6;i++){
        vsq4(temp[0],temp[1],temp[2],temp[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], temp[0],temp[1],temp[2],temp[3]);
    }
    // 2^12-1
    vmult4(temp[0],temp[1],temp[2],temp[3], 
           x_6_1[0],x_6_1[1],x_6_1[2],x_6_1[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3]);
    
    for(int i=0; i<4;i++){temp[i] = x_12_1[i];}
    //2^24-2^12
    for(int i=0;i<12;i++){
        vsq4(temp[0],temp[1],temp[2],temp[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], temp[0],temp[1],temp[2],temp[3]);
    }
    //2^24-1
    vmult4(temp[0],temp[1],temp[2],temp[3], 
           x_12_1[0],x_12_1[1],x_12_1[2],x_12_1[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_24_1[0],x_24_1[1],x_24_1[2],x_24_1[3]);
    
    //2^25-2
    vsq4(x_24_1[0], x_24_1[1], x_24_1[2], x_24_1[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3]);
    
    //2^25-1
    vmult4(in[0],in[1],in[2],in[3], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3]);
    
    
    for(int i=0; i<4;i++){temp[i] = x_25_1[i];}
    //2^50-2^25
    for(int i=0;i<25;i++){
        vsq4(temp[0],temp[1],temp[2],temp[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], temp[0],temp[1],temp[2],temp[3]);
    }
    //2^50-1
    vmult4(temp[0],temp[1],temp[2],temp[3], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3]);
    
    for(int i=0; i<4;i++){temp[i] = x_50_1[i];}
    //2^100-2^50
    for(int i=0;i<50;i++){
        vsq4(temp[0],temp[1],temp[2],temp[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], temp[0],temp[1],temp[2],temp[3]);
    }
    // 2^100-1
    vmult4(temp[0],temp[1],temp[2],temp[3], 
           x_50_1[0],x_50_1[1],x_50_1[2],x_50_1[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_100_1[0],x_100_1[1],x_100_1[2],x_100_1[3]);
    
    for(int i=0; i<4;i++){temp[i] = x_100_1[i];}
    // 2^125-2^25
    for(int i=0;i<25;i++){
        vsq4(temp[0],temp[1],temp[2],temp[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], temp[0],temp[1],temp[2],temp[3]);
    }
    //2^125-1
    vmult4(temp[0],temp[1],temp[2],temp[3], 
           x_25_1[0],x_25_1[1],x_25_1[2],x_25_1[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3]);
    
    for(int i=0; i<4;i++){temp[i] = x_125_1[i];}
    //2^250-2^125
    for(int i=0;i<125;i++){
        vsq4(temp[0],temp[1],temp[2],temp[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], temp[0],temp[1],temp[2],temp[3]);
    }
    //2^250-1
    vmult4(temp[0],temp[1],temp[2],temp[3], 
           x_125_1[0],x_125_1[1],x_125_1[2],x_125_1[3], 
            w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], x_250_1[0],x_250_1[1],x_250_1[2],x_250_1[3]);
    
    // 2^251-2
    vsq4(x_250_1[0],x_250_1[1],x_250_1[2],x_250_1[3], w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    	foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    	reducem(wf[0],wf[1],wf[2],wf[3], op[0],op[1],op[2],op[3]);
}



//------------------SCALAR MULTIPLICATION-------------------------------------------------------



//Function to clamp the scalar
void clamp(byte n[32]){
	n[0] 	= n[0] & 0xfc;
	n[31] 	= n[31] & 0x07;
	n[31] 	= n[31] | 0x04;
}


// Function to compute scalar multiplication using left to right Montgomery ladder
void scalarMult_fixed_base( byte *n, byte *nw, byte *nz ){  
    
    clamp(n);
       
    vec nPwz[4], w[8], wf[4];
    vec wz[4];
                              
    vec S[4], R[4];
    
    S[0] = baseP[0];	S[1] = baseP[1];	S[2] = baseP[2];	S[3] = baseP[3];	
    wz[0] = baseP[0];	wz[1] = baseP[1];	wz[2] = baseP[2];	wz[3] = baseP[3];	
    R[0] = _2baseP[0];	R[1] = _2baseP[1];	R[2] = _2baseP[2];	R[3] = _2baseP[3];   
        
    byte pb = 0, b, ni;

    int j = 1;
    for(int i = 31; i >= 0 ; i--){
        for(; j >= 0 ; j--){
            ni = (n[i] >> j) & 1;
            b = pb ^ ni;
            
            swap(S[0],S[1],S[2],S[3],R[0],R[1],R[2],R[3],cons[b]);
            
            ladderStep(S, R, wz);
            
            pb = ni;
        }
        j = 7;
    }
    swap(S[0],S[1],S[2],S[3],R[0],R[1],R[2],R[3],cons[pb]);
    
    
    vec W[4],Z[4];
         
    invert(S,Z);
    
    //0 | nw
    permute_11(W[0],zero,S[0]);
    permute_11(W[1],zero,S[1]);
    permute_11(W[2],zero,S[2]);
    permute_11(W[3],zero,S[3]);
    
  
    
    vmult4(W[0],W[1],W[2],W[3], 
		Z[0],Z[1],Z[2],Z[3], 
		 w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], nPwz[0], nPwz[1],nPwz[2],nPwz[3]);
                  
   
    unpack64(nPwz, nw, nz);
    
}


// Function to compute scalar multiplication using left to right Montgomery ladder
void scalarMult_var_base( byte *w1, byte *n, byte *nw, byte *nz ){  
    
    clamp(n);
    vec Pxz[4], nP[4], S[4], R[4];
    vec w[8],wf[4];
    byte z1[32] = {1};
    pack64(z1, w1, S); //S = w1 | z1
    vec wz[4];

    
    
    wz[0] = S[0];
    wz[1] = S[1];
    wz[2] = S[2];
    wz[3] = S[3];
    
   
    vec B1[4], B2[4], C[4], C1[4], C2[4], E1[4], E2[4], G[4], G2[4], temp1[4], temp2[4], temp3[4];
        
    byte pb = 0, b, ni;
    
	
    // B2 = z1 | 0
    permute_00(B2[0], S[0], zero);
    permute_00(B2[1], S[1], zero);
    permute_00(B2[2], S[2], zero);
    permute_00(B2[3], S[3], zero);
    
    // temp1 = w1 + z1 | z1
    add4(S[0], S[1], S[2], S[3],
	   B2[0], B2[1], B2[2], B2[3],
       temp1[0], temp1[1], temp1[2], temp1[3]);
    
    
    	
    //C = w1*(w1+z1) | z1^2
    vmult4(temp1[0], temp1[1], temp1[2], temp1[3],
	    S[0], S[1], S[2], S[3],
      	w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
	       
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);     
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],wf[0],wf[1],wf[2],wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3],C[0],C[1],C[2],C[3]);      
    

    //temp3 = c^2 | z1^4 where c = w1(w1+z1)
    vsq4(C[0], C[1], C[2], C[3],
    	 w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    	     
    expandS(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],wf[0],wf[1],wf[2],wf[3]);  
    reducem(wf[0],wf[1],wf[2],wf[3],temp3[0],temp3[1],temp3[2],temp3[3]);
    	
  
    	
    //E1 = 0 | c^2
    permute_01(E1[0], zero, temp3[0]);
    permute_01(E1[1], zero, temp3[1]);
    permute_01(E1[2], zero, temp3[2]);
    permute_01(E1[3], zero, temp3[3]);
	
    
    //E2 = c^2 | d*z2^4
    vmulC(temp3[0], temp3[1], temp3[2], temp3[3], par_d,
    	      wf[0], wf[1], wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3],E2[0],E2[1],E2[2],E2[3]);;
    
    // R = c^2 | c^2 + d*z1^4 = 2P
    add4(E1[0], E1[1], E1[2], E1[3],
    	 E2[0], E2[1], E2[2], E2[3],
    	 R[0], R[1], R[2], R[3]);
    	 
    
    int j = 1;
    for(int i = 31; i >= 0 ; i--){
        for(; j >= 0 ; j--){
            ni = (n[i] >> j) & 1;
            b = pb ^ ni;
            
            swap(S[0],S[1],S[2],S[3],R[0],R[1],R[2],R[3],cons[b]);
           
            ladderStep(S, R, wz);
            
            pb = ni;
        }
        j = 7;
    }
    swap(S[0],S[1],S[2],S[3],R[0],R[1],R[2],R[3],cons[pb]);
    
    vec W[4],Z[4];
         
    invert(S,Z);
    
    permute_11(W[0], zero, S[0]);
    permute_11(W[1], zero, S[1]);
    permute_11(W[2], zero, S[2]);
    permute_11(W[3], zero, S[3]);
    
    vmult4(W[0],W[1],W[2],W[3], 
		Z[0],Z[1],Z[2],Z[3], 
		 w[0],w[1],w[2],w[3],w[4],w[5],w[6]);
    expandM(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]);
    foldm(w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7], wf[0],wf[1],wf[2], wf[3]);
    reducem(wf[0],wf[1],wf[2],wf[3], nP[0], nP[1],nP[2],nP[3]);
                  

    unpack64(nP, nw, nz);
    
}




