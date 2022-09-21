#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
      int degree;
      int coefficient;
      struct Node *next;
      struct Node *prev;  
    } Node;

Node *inputpoly(void){
    struct Node *first;
    struct Node *last;
    struct Node node;
    struct Node *pt;
    pt = &node;

    first = NULL;
    last = NULL;
    while(1) {
        int deg, coef;
        printf("Input (degree) (coefficient): ");
        scanf("%d %d", &deg, &coef);
        pt = (struct Node*)malloc(1*sizeof(Node));
        pt -> degree = deg;
        pt -> coefficient = coef;
        if (deg < 0 && coef < 0){
            printf("Done!! \n");
            last -> next = NULL;
            break;
        }
        else if (deg > 0 && coef > 0) {
                // float로 받을 때 정의
                if (first == NULL){
                    first = pt;
                    first -> prev = NULL;
                }
                else{
                    if (last == NULL){
                        first -> next = pt;
                        pt -> prev = first;

                    }
                    else {
                        pt -> prev = last;
                        last -> next = pt;
                    };
               }
            last = pt;
        }
        else {
            printf("양의 정수만 입력해주세요. (종료하고 싶을 경우 degree와 coefficient를 모두 음의 정수로 입력해주세요.)\n");
        };
    };
    free(pt);
    pt = NULL;
    return first;     
};

void printNode(Node *inp){
    struct Node *temp;
    while (inp != NULL){
        if (inp->degree == 1){
            printf("%dx", inp->coefficient);
        }
        else {
            printf("%dx^%d", inp->coefficient, inp->degree);
        };
        if (inp->next == NULL){
            printf("\n");
        }
        else{
            printf(" + ");
        }
        temp = inp;
        inp = temp -> next;
    };
};

Node *multiply(Node *a, Node *b){
    Node *pt1 = a;
    Node *pt2 = b;
    Node *first = NULL;
    Node *last = NULL;
    Node *temp;
    Node *temp2;
    Node *node = NULL;
    Node result; //node
    Node *result_pt;
    result_pt = &result; //pt와 동일
    int multiply_coef;
    int sum_deg;

    while (pt1){
        while(pt2){
            printf("fis\n");
            result_pt = (struct Node*)malloc(1*sizeof(Node));
            multiply_coef = pt1 -> coefficient * pt2 -> coefficient;
            sum_deg = pt1 -> degree + pt2 -> degree;            
            result_pt -> coefficient = multiply_coef;
            result_pt -> degree = sum_deg;
            if (first == NULL){
                first = result_pt;
                first -> prev = NULL;
                node = first;
            }
            else {
                while (1){
                printf("%d %d\n", node ->degree, node ->coefficient);
                printf("%d %d\n", result_pt->degree, result_pt->coefficient);
                    if (sum_deg == node->degree){
                        (node -> coefficient) += multiply_coef;
                        last = node;
                        break;
                    }
                    else{
                        if (node -> next == NULL){
                            if (last == NULL){
                                first -> next = result_pt;
                                result_pt -> prev = first;
                            }
                            else {
                                result_pt -> prev = last;
                                last -> next = result_pt;
                                last = result_pt;
                                last -> prev = node;
                            };
                            break;
                        }
                    }
                    temp2 = node;
                    node = temp2 -> next;
                };
            };
            temp = pt2;
            pt2 = temp -> next;
            node = first;
        }
        temp = pt1;
        pt1 = temp -> next;
        pt2 = b;
    }
    last -> next = NULL;
    result_pt = NULL;
    free(result_pt);
    return first;
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