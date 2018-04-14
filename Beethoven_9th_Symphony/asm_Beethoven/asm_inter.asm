;
;  ======== load.asm ========
;
;  C-callable interface to assembly language utility functions for the
;  asmC example.

    .mmregs

    .global _asm_inter     ; make this function available everywhere


    .text                    ; start of executable mnemonics 

	
_asm_inter:
    ; Interpolation
	MOV T0,HI(AC0) ; Move the Integer part of Delta in T0 to the Higher Part of the Accumulator AC0
	BCLR SXMD ; Clear the sign-bit
	ADD T1,AC0 ; Add the Fractional part to AC0, so that the lower part has the Fractional-Part of Delta
	BSET SXMD ; Set the sign-bit 
	MOV #0,AC1 ; Clear AC1
	MOV AR1,AC2 ; Move the current sample value to AC2 
    BCC OUT, AC2==#0 ; If the sample is 0, goto label 'OUT'
   
   
Loop: ADD AC0,AC1 ; Add the Integer and Fractional Deltas stored in AC0 to AC1
      Bcc OUT, AC2==#0 ; If the sample is 0, goto label 'OUT'
      ADD #-1,AC2 ; Decrement sample
      B Loop ; Loop till AC2 becomes zero

OUT:	
		MOV HI(AC1),T2 ; Move index value stored in Higher-part of AC1 to T2
		AND #511,T2 ; Mask it with 511 since sine-table size is 512
	 	ADD T2,AR0 ; Shift the current Baseaddress of the sinetable by the index 
		MOV *AR0+,T0 ; Store the current value in T0 and increment the address pointer
		MOV *AR0,T1 ; Store the next sinetable value in T1
		SUB T0,T1 ; T1<-Baseaddress[l+1]-Baseaddress[l]
		MOV AC1,T3 ; Move the fractional part -> T3
		MOV #0, AC0 ; Clear AC0
		BCLR SXMD ; Clear the sign-bit
		MOV T3, HI(AC0) ; Move the fractional-part to High-part of AC0
		BSET SXMD ; Set the sign-bit
		MOV #0,AC1 ; Clear AC1	
	    
	    MPY T3,T1,AC1 ; AC1-> m*x
		MOV HI(AC1),T3 ; The High part of AC1-> T3 (Baseaddress[l])
	    ADD T3,T0 ; T0<-(m*x)+b
	
	
done:
    RET                         ;/* end load() */



