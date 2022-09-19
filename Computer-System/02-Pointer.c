#include <stdio.h>

int main(void){
    char str1[256];
    char str2[128];
    int len=0; // 기존 문자열의 길이 계산

    char *pt1 = str1; // str1을 가리키는 포인터 변수
    char *pt2 = str2; // str2을 가리키는 포인터 변수

    printf("문자열을 입력하세요: ");
    scanf("%s", str1);

    printf("붙일 문자열을 입력하세요: ");
    scanf("%s", str2);

    // 문자열 str1의 길이 계산
    while (*pt1 != '\0'){
        len++;
    }
    printf("%d\n", len);
    
    // 문자열을 붙임
    while (*pt2 != '\0') {
        *(pt1+len) = *pt2;
        len++;
        pt2++;
    };

    printf("done");
    // 문자열의 끝을 알리기 위해 문자열의 맨 끝에 NULL 문자 저장
    // *(pt1+len) = '\0';

    printf("결과: ");
    printf("%s", str1);

    return 0;
}