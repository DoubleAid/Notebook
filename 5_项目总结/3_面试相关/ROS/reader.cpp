//
// Created by guanggang.bian on 2023/3/20.
//
#include "ros_modify.cpp"
#include <stdio.h>

using namespace std;

int main() {
    redShm<Message> reader(66);
    Message *redret = new Message(0, 0, 0, 0);
    if(!(reader.hasr())) {
        reader.hasr(1);
        for(int i = 0; i < 20; i++) {
            if (reader.readmove(redret)) {
                printf("****I gona move read point****\n")
                printf("read x result is:%d\n", (redret->x));
                printf("read y result is:%d\n", (redret->y));
                printf("read z result is:%d\n", (redret->z));
                printf("read w result is:%d\n", (redret->w));
            } else {
                printf("read failure\n");
            }
            sleep(0.3);
        }
    }
    else {
        for(int i = 0; i < 20; i++) {
            if(reader.read(redret)) {
                printf("read x result is:%d\n", (redret->x));
                printf("read y result is:%d\n", (redret->y));
                printf("read z result is:%d\n", (redret->z));
                printf("read w result is:%d\n", (redret->w));
            } else {
                printf("read failure\n");
            }
            sleep(0.3);
        }
    }
    reader.hasr(0);
    return 0;
}