//
// Created by guanggang.bian on 2023/3/20.
//

#include "ros_modify.cpp"
#include <stdio.h>

using namespace std;

int main() {
    Message *cor = new Message(0, 0, 0, 0);
    wriShm<Message> writer(5, 66);
    for(int i = 0; i < 20; i++) {
        cor->x = i; cor->y = i; cor->z = i; cor->w = i;
        if(writer.write(*cor)) {
            printf("write success\n");
        } else {
            printf("write failure\n");
        }
        sleep(0.3);
    }
    delete cor;
    return 0;
}