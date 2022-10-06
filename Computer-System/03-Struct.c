#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
      int degree;
      int coefficient;
      struct Node *next;
      struct Node *prev;  
    } Node;
    // Node 구조체는 하나의 타입으로 여겨져야 하므로 typedef 사용
    // Node 구조체의 별명을 Node라고 설정

Node *inputpoly(void){
    struct Node *first; // 가장 첫 번째 노드를 가리킬 구조체 포인터
    struct Node *last; // 이전에 입력한 노드를 기억할 구조체 포인터

    struct Node node; // 입력받는 노드를 저장할 구조체 (주석에서 언급하는 노드는 현재 입력 받는 단항을 의미하고, 이 구조체 node는 이 단항을 받는 구조체를 의미합니다.)
    struct Node *pt; // 현재 입력 받는 node 구조체를 가리킬 구조체 포인터
    pt = &node; // pt가 node를 가리키고 있다는 것을 알려줌
    
    struct Node *temp; // 차수 비교를 위한 temp 구조체 포인터
    struct Node *temp2; // 차수 비교를 위해 필요한 temp를 임시적으로 담아둘 구조체 포인터

    first = NULL; // first는 일단 아무도 가리키고 있지 않음
    last = NULL; // last는 일단 아무도 가리키고 있지 않음

    while(1) { // 계속해서 입력을 받아야 하므로 무한 루프로 설정
        float deg, coef; // 현재 입력 받는 두 값을 담아둘 실수 변수 선언 (deg = 차수, coef = 계수) -> 정수로는 아래에서 int casting

        printf("Input (degree) (coefficient): ");
        scanf("%f %f", &deg, &coef); // 변수를 입력 받음

        pt = (struct Node*)malloc(1*sizeof(Node)); // node 구조체 포인터 pt에 동적 메모리 할당
        pt -> degree = (int)deg; // pt가 가리키는 구조체 node의 degree에 deg 저장
        pt -> coefficient = (int)coef; // pt가 가리키는 구조체 node의 coefficient에 coef 저장

        if (deg < 0 && coef < 0){ //만약 deg와 coef 모두 음수라면 break를 통해 루프 종료
            printf("Done!! \n");
            last -> next = NULL; // 마지막 node의 next라는 구조체 포인터에는 NULL 값 저장
            break;
        }

        else if (deg-(int)deg != 0 || coef-(int)coef != 0){ // 실수에서 정수 부분을 뺀 값이 0과 동일하면 정수로 판단 -> deg와 coef가 모두 정수가 아닐 경우 다시 입력하라고 함
            printf("실수를 입력하셨네요. 양의 정수만 입력해주세요. \n(종료하고 싶을 경우 degree와 coefficient를 모두 음의 정수로 입력해주세요.)\n");
        }

        else if ((deg >= 0) && coef > 0) { // 만약 deg는 0 이상, coef가 양의 정수라면 else if 문 실행
                
                if (first == NULL){ // first가 NULL일 때
                    first = pt; // 지금까지 받은 deg와 coef의 정보가 담긴 node 구조체를 가리키는 pt를 first에 저장 (first와 pt는 현재 같은 주소값을 가리키고 있음)
                    first -> prev = NULL; // 여기서 받은 포인터는 첫 번째 node이므로 prev는 NULL값이 됨
                    temp = first;
                }
                
                else{ // 두 번째 노드부터 실행될 곳
                    while (1){ // 계산된 전체 노드를 돌면서 차수를 비교할 수 있도록 무한 루프 사용
                
                    if (deg == temp->degree){ // 만약 계산된 차수의 합이 node(first라고 가정)의 차수와 같으면
                        (temp -> coefficient) += coef; // node의 coefficient에 multiply_coef를 더해서 저장
                        break; // 그럼 해당 노드에 대한 계산은 끝났으니 break로 차수 비교를 위한 무한 루프를 빠져나옴
                    }
                
                    else{ // 만약 계산된 차수의 합이 node(first라고 가정)의 차수와 다를 때
                
                        if (temp -> next == NULL){ // node, 즉 차수를 비교하기 위한 node의 next가 NULL(이를 설정하지 않으면 같은 차수의 노드가 또 추가됨)일 때
                
                            if (last == NULL){ // last, 즉 마지막 node를 가리키는 last 포인터가 NULL일 때 (여기서는 두 번째 node를 가리킬 포인터가 됨)
                                first -> next = pt; // first의 next는 현재 받은 result_pt이므로 이를 저장
                                pt -> prev = first; // result_pt, 즉 현재 받은 node의 prev는 first이므로 이를 저장
                                last = pt; // 노드 연결 후 다음 번에 받을 노드를 위해 last가 가리키는 곳이 현재의 result_pt가 가리키는 곳으로 바꾸도록 함
                            }
                
                            else { // 세번째 노드부터 실행
                                pt -> prev = last; // result_pt(세번째 노드라고 가정 했을 때)의 이전은 last(last는 지금 두 번째 노드를 가리키고 있음)임
                                last -> next = pt; // last(두 번째 노드를 가리키고 있음)의 next는 result_pt(세 번째 노드를 가리키고 있음)임
                                last = pt;  // 노드 연결 후 다음 번에 받을 노드를 위해 last가 가리키는 곳이 현재의 result_pt가 가리키는 곳으로 바꾸도록 함
                                last -> prev = temp; // node는 단순히 일시적인 비교를 위해 사용되지만, 최종적으로는 last가 가리키는 곳의 이전이므로 이와 같이 설정
                            };
                            
                            break; // 계산이 완료되면 break로 무한 루프를 빠져나옴
                        }
                    }
                    temp2 = temp; // node가 가리키는 곳을 임시 저장
                    temp = temp2 -> next; // node의 next가 가리키는 곳을 다시 node에 넣음 (첫 번째 노드와의 차수 비교를 끝냈으면 다음 노드와의 차수 비교를 해야하기 때문)
                };
               }
            last = pt; // node 연결 후 다음 번에 받을 노드를 위해 last가 가리키는 곳이 현재의 pt가 가리키는 곳으로 바꾸도록 함
        }
        else { // 계수를 0으로 입력했을 때 등의 상황에 쓰이는 부분
            printf("계수는 양의 정수만 입력해주세요. \n(종료하고 싶을 경우 degree와 coefficient를 모두 음의 정수로 입력해주세요.)\n");
        };
    };

    free(pt); // pt에 할당된 동적 메모리를 해제
    pt = NULL; //pt는 다시 NULL 값을 가리키도록 함
    return first; // 첫 번째 노드를 가리키는 first 포인터 반환
};

void printNode(Node *inp){

    struct Node *temp; // inp를 저장할 임시 포인터 선언
    
    while (inp != NULL){ // inp가 가리키는 곳이 NULL이면 루프 종료
        if (inp->degree == 1){ // inp의 degree가 1, 즉 차수가 1일 때는 (숫자)x로 나오도록 함
            printf("%dx", inp->coefficient);
        }
        else if (inp->degree == 0){ // inp의 degree가 0, 즉 상수항일 때는 숫자만 나오도록 함
            printf("%d", inp->coefficient);
        }
        else { // 그 외의 경우에는 (계수)x^(차수)의 형태로 나오도록 함
            printf("%dx^%d", inp->coefficient, inp->degree);
        };
        if (inp->next == NULL){ // 만약 inp의 next가 NULL일 때는 엔터를 치고 종료할 수 있도록 함
            printf("\n");
        }
        else{ // 그 외의 경우에는 노드들을 +로 연결해줌
            printf(" + ");
        };
    
        temp = inp; // inp가 가리키는 노드를 temp가 가리키도록 함
        inp = temp -> next; // temp의 next, 즉 inp의 다음 노드 위치를 inp에 저장함
    };
};

Node *multiply(Node *a, Node *b){
    Node *pt1 = a; // a가 가리키는 곳을 가리키는 pt1
    Node *pt2 = b; // b가 가리키는 곳을 가리키는 pt2
    
    Node *first = NULL; // inputpoly 함수와 동일한 역할
    Node *last = NULL; // inputpoly 함수와 동일한 역할
    
    Node *temp; // pt1과 pt2가 다음 노드를 가리킬 수 있도록 도와주는 포인터 변수
    Node *temp2; // 차수 비교 시 아용되는 node라는 포인터가 다음 노드를 가리킬 수 있도록 도와주는 포인터 변수
    Node *node = NULL; // 차수 비교 시 이용되는 포인터 변수
    
    Node result; // 두 항을 곱한 결과를 담을 구조체
    Node *result_pt; // 두 항을 곱한 결과를 담을 구조체를 가리키는 포인터 변수
    result_pt = &result; // result와 result_pt를 연결
    
    int multiply_coef; // 두 항의 계수를 곱한 값을 저장할 변수
    int sum_deg; // 두 항의 차수를 더한 값을 저장할 변수

    while (pt1){
        while(pt2){
            result_pt = (struct Node*)malloc(1*sizeof(Node)); // result_pt에 동적 메모리 할당
            multiply_coef = pt1 -> coefficient * pt2 -> coefficient; // pt1의 coefficient와 pt2의 coefficient의 곱
            sum_deg = pt1 -> degree + pt2 -> degree; // pt1의 degree와 pt2의 degree의 합
            
            result_pt -> coefficient = multiply_coef; // result_pt가 가리키는 result 구조체의 coefficient에 multiply_coef 저장
            result_pt -> degree = sum_deg; // result_pt가 가리키는 result 구조체의 degree에 sum_deg 저장
            
            if (first == NULL){ // first가 NULL일 때
                first = result_pt; // 지금까지 받은 deg와 coef의 정보가 담긴 구조체를 가리키는 result_pt를 first와 동일시함
                first -> prev = NULL; // 여기서 받은 포인터는 첫 번째 node이므로 prev는 NULL값이 됨
                node = first; // 차수 비교에 사용되는 node는 first부터 시작될 수 있도록 first를 저장
            }
            
            else {
                while (1){ // 계산된 전체 노드를 돌면서 차수를 비교할 수 있도록 무한 루프 사용
                    if (sum_deg == node->degree){ // 만약 계산된 차수의 합이 node(first라고 가정)의 차수와 같으면
                        (node -> coefficient) += multiply_coef; // node의 coefficient에 multiply_coef를 더해서 저장
                        break; // 그럼 해당 노드에 대한 계산은 끝났으니 break로 차수 비교를 위한 무한 루프를 빠져나옴
                    }
                    else{ // 만약 계산된 차수의 합이 node(first라고 가정)의 차수와 다를 때
                        if (node -> next == NULL){ // node, 즉 차수를 비교하기 위한 node의 next가 NULL(이를 설정하지 않으면 같은 차수의 노드가 또 추가됨)일 때
                            if (last == NULL){ // last, 즉 마지막 node를 가리키는 last 포인터가 NULL일 때 (여기서는 두 번째 node를 가리킬 포인터가 됨)
                                first -> next = result_pt; // first의 next는 현재 받은 result_pt이므로 이를 저장
                                result_pt -> prev = first; // result_pt, 즉 현재 받은 node의 prev는 first이므로 이를 저장
                                last = result_pt; // 노드 연결 후 다음 번에 받을 노드를 위해 last가 가리키는 곳이 현재의 result_pt가 가리키는 곳으로 바꾸도록 함
                            }
                            else { // 세번째 노드부터 실행
                                result_pt -> prev = last; // result_pt(세번째 노드라고 가정 했을 때)의 이전은 last(last는 지금 두 번째 노드를 가리키고 있음)임
                                last -> next = result_pt; // last(두 번째 노드를 가리키고 있음)의 next는 result_pt(세 번째 노드를 가리키고 있음)임
                                last = result_pt;  // 노드 연결 후 다음 번에 받을 노드를 위해 last가 가리키는 곳이 현재의 result_pt가 가리키는 곳으로 바꾸도록 함
                                last -> prev = node; // node는 단순히 일시적인 비교를 위해 사용되지만, 최종적으로는 last가 가리키는 곳의 이전이므로 이와 같이 설정
                            };
                            break; // 계산이 완료되면 break로 무한 루프를 빠져나옴
                        }
                    }
                    temp2 = node; // node가 가리키는 곳을 임시 저장
                    node = temp2 -> next; // node의 next가 가리키는 곳을 다시 node에 넣음 (첫 번째 노드와의 차수 비교를 끝냈으면 다음 노드와의 차수 비교를 해야하기 때문)
                };
            };
            temp = pt2; // pt2가 가리키는 노드를 temp가 가리키도록 함
            pt2 = temp -> next;  // temp의 next, 즉 pt2의 다음 노드 위치를 pt2에 저장함
            node = first; // node는 새로운 항이 생겼을 때 이를 계속해서 차수 비교 해야하므로 다시 first로 선언
        }
        temp = pt1; // pt1가 가리키는 노드를 temp가 가리키도록 함
        pt1 = temp -> next; // temp의 next, 즉 pt1의 다음 노드 위치를 pt1에 저장함
        pt2 = b; // pt2는 다시 b, 즉 두 번째 다항식의 첫 번째 노드를 가리켜야 하므로 b로 저장
    }
    if (last == NULL){ // 만약 last가 처음과 같이 여전히 NULL이면
        result_pt = NULL; // first의 next가 여전히 NULL값을 가지고 있으므로 result_pt만 NULL을 향하게 해주면 됨
        free(result_pt); // result_pt에 할당된 동적 메모리를 해제
        return first; // 첫 번째 노드를 가리키는 first 포인터 반환
    }
    else{ // 만약 last가 무언가를 가리키고 있다면
        last -> next = NULL; // 이 last는 마지막 노드이므로 last의 next가 NULL임을 알려줌
        result_pt = NULL; // result_pt가 NULL을 향하게 해주면 됨
        free(result_pt); // result_pt에 할당된 동적 메모리를 해제
        return first; // 첫 번째 노드를 가리키는 first 포인터 반환
    };
};

int main(void){
    // 첫 번째 다항식 a 정의 및 출력
    struct Node *a = inputpoly();
    printNode(a);
    // 두 번째 다항식 b 정의 및 출력
    struct Node *b = inputpoly();
    printNode(b);
    // 다항식을 곱한 후 출력
    printNode(multiply(a, b));

    return 0;
}