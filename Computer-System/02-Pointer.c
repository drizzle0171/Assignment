#include <stdio.h>

int main(void){
    char str1[256];
    char str2[128];
    int len=0; // str1의 문자열 길이 계산

    char *pt1 = str1; // str1을 가리키는 포인터 변수
    char *pt2 = str2; // str2을 가리키는 포인터 변수

    // 각각의 문자열을 입력 받음
    printf("문자열을 입력하세요: ");
    scanf("%s", str1);

    printf("붙일 문자열을 입력하세요: ");
    scanf("%s", str2);

    // 문자열 str1의 길이 계산 - str1의 문자열 길이를 계산하여 길이만큼 pointer가 뒤를 가리키게 만든 후 str2의 내용 복사
    while (str1[len] != '\0'){
        len++;
    };
    
    // 문자열을 붙임
    while (*pt2 != '\0') {
        *(pt1+len) = *pt2; //pt1은 str1의 가장 처음의 주소를 가리키고 있으므로 pt2는 pt1+len만큼, 즉 pt1의 마지막 부분+1부터 가리켜야 함
        len++; // len은 str1의 마지막+1만큼의 위치를 알려주는 역할을 하므로 한 칸 씩 뒤로 가야함
        pt2++; // pt2 역시 다음으로 붙일 문자열을 가리켜야 하므로 ++를 이용해줌
    };

    // 문자열의 끝을 알리기 위해 문자열의 맨 끝에 NULL 문자 저장
    *(pt1+len) = '\0'; //현재 len은 str1과 str2가 합쳐진 문장의 길이를 의미하므로 pt1의 첫 주소에서 len만큼 뒤로 간 후 NULL 문자 저장

    printf("결과: ");
    printf("%s\n", str1); //결과 출력

    return 0;
}