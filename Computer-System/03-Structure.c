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
        pt = malloc(1*sizeof(Node));
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
        if (inp->coefficient == 1){
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
