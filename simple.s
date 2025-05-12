00     XOR        ECX ,ECX
02     MOV        EAX ,ECX
    LAB_140001004
04     TEST       EAX ,EAX
06     JNZ        LAB_14000100e
08     INC        index
0a     INC        sum
    LAB_14000100e
0e     CMP        index ,0x1
11     JNZ        LAB_14000101a
13     INC        index
15     ADD        sum ,0x2
    LAB_14000101a
1a     CMP        index ,0x2
1d     JZ         LAB_14000102c
1f     ADD        sum ,0x4
22     INC        index
27     JL         LAB_140001004
2b    RET
    LAB_14000102c
2c     LEA        index ,[sum  + 0x3 ]
2f    RET