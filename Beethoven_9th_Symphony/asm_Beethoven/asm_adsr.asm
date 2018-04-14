;
;  ======== load.asm ========
;
;  C-callable interface to assembly language utility functions for the
;  asmC example.

    .mmregs

    .global _asm_adsr     ; make this function available everywhere


    .text                    ; start of executable mnemonics 

	
_asm_adsr:
 	;ADSR
 	MOV #0,AC0 ; Clear AC0
 	MOV #0,AC1 ; Clear AC1
 	MOV #0,AC2 ; CLear AC2
	MOV #30,T2 ; Move 30->T2
	CMP T0 < T2,TC1 ; If Duration<30ms, set TC1=1
	BCC LOOP1,TC1 ; Then goto label 'LOOP1'
	
	MOV #970,T3 ; Move 970->T3
	CMP T0<T3,TC2 ; If Duration<970ms, set TC2=1
	BCC LOOP2,TC2 ; Then goto label 'LOOP2'
	B LOOP3 ; Otherwise, if Duration>970ms, branch to 'LOOP3'

LOOP1: 
	B SKIP ; Goto SKIP
		
LOOP2: 
	ADD #1, AR0 ; Increment the 'g' address by 1
    ADD #1, AR1	; Increment the 't' address by 1
    B SKIP ; Then goto SKIP
 
LOOP3:  
	ADD #2, AR0 ; Increment the 'g' address by 2
	ADD #2, AR1	; Increment the 'g' address by 2
  	B SKIP ; Then goto SKIP
	    
SKIP:
	MOV #0,T2 ; Clear T2
	MOV #0,T0 ; Clear T0
	BCLR SXMD ; Clear sign-bit	
	MOV T1,HI(AC2); Move y_prev->High part of AC2
	MOV *AR1,T0 ; Move the current target value to T0
	MOV *AR0,T2 ; Move the current rise/decay value to T2
	MOV T0,HI(AC0) ;Move the target value to the high-part of AC0
	BSET SXMD ; Set the sign-bit
	BSET FRCT ; Set Fraction because of fractional multiplication 
    MPY T2,AC0,AC3 ; Mutiply the rise/decay value (g) with the target (t) value and store the result in AC3 [p=(g*t)]
    MPY T2,AC2,AC1 ; Mutiply the rise/decay value (g) with the previous value of y (y_prev) value and store the result in AC1 [g*y_prev]
   	BCLR FRCT ; Clear the fraction
    BCLR SXMD ; CLear the sign-bit
    SUB AC1,AC2 ;Subtract the result in AC1 from the preious y value (y_prev)[s=(y_prev-g*y_prev)]
    ADD AC3,AC2 ; Add the result to AC3 amd store in AC2 [s+p]
    MOV HI(AC2),T0 ; Move the higher part of AC2-> T0
	BSET SXMD ; Set the sign-bit
	
done:
    RET                         ;/* end load() */




