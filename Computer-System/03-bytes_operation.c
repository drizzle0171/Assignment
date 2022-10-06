#include <stdio.h>

char* show_binary_int(int param){
    char int2binary[33] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'};
    char* pt = int2binary;
    int plusOne = 0;
    if (param<0){
        param = (param*-1);
        for (int i=0; i <32; i++){ // 일단 2진수로 변환
            if (param == 0){
                pt[i] = (char)('0'+param%2);
            }
            pt[i] = (char)('0'+param%2);
            param = (int)(param/2);
        }
        pt[31] = '1'; // 부호비트 변환
        for (int i=0; i<31; i++){ // 부호비트 제외한 나머지 비트 not 연산
            if (pt[i]=='0'){
                pt[i]='1';
            }
            else{
                pt[i]='0';
            }
        }
        // 1 더하기
        if (pt[0] == '0'){
            pt[0] == '1';
        }
        else {
            pt[0] = '0';
            for (int i = 1; i<32; i++){
                if (pt[i] == '0'){
                    pt[i] = '1';
                    break;
                }
                else{
                    pt[i] = '0';
                }

            }
        }
        
    }
    else{
        for (int i=0; i <32; i++){
            if (param == 0){
                pt[i] = (char)('0'+param%2);
            }
            pt[i] = (char)('0'+param%2);
            param = (int)(param/2);
    }
    }
    return pt;
}

int main(void){
    int first, second, result;
    char operand;

    printf("Input the First integer: ");
    scanf("%d", &first);

    printf("Input the Second integer: ");
    scanf("%d", &second);

    printf("Select the Operand(+, -, *, /): ");
    scanf(" %c", &operand);

    switch (operand){
        case '+':
            result = first + second;
            break;
        case '-':
            result = first - second;
            break;
        case '*':
            result = first * second;
            break;

        case '/':
            result = first / second;
            break;
  
    };
    char* one;
    char* two;
    char* final;

    one = show_binary_int(first);
    printf("   ");
    for (int i = 31; i>=0; i--)
        printf(" %c", one[i]);
    printf("\n");
    printf("%c  ", operand);
    two = show_binary_int(second);
    for (int i = 31; i>=0; i--)
        printf(" %c", two[i]);
    printf("\n");
    printf("---------------------------------------------------------------------\n    ");
    final = show_binary_int(result);
    for (int i = 31; i>=0; i--)
        printf("%c ", final[i]);
    
    printf("\n\nResult is %d\n", result);
    return 0;
}