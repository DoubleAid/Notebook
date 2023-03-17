//
// Created by guanggang.bian on 2023/3/16.
//


#include <stdio.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <stdlib.h>

using namespace std;

struct Message {
    int x;
    int y;
    int z;
    int w;
    Message(int xval, int yval, int zval, int wval) :
            x(xval), y(yval), z(zval), w(wval) {}
};

template <class T>
class wriShm {
public:
    wriShm(int number, int proj_id);
    virtual ~wriShm();
    bool write(const T &i);
private:
    int wfd;
    void *ptr;
    int* hasred, *size, *pwri, *pred;
    T *array;
};

template <class T>
wriShm<T>::wriShm(int number, int proj_id) {
    key_t key = ftok("./", proj_id);
    if(key == -1) {
        perror("ftok failure");
        exit(1);
    }
    this->ptr = shmget(key, (16+sizeof(T)*(number+1)), IPC_CREAT|0666);
    if (wfd == -1) {
        perror("shmget failure\n");
        exit(1);
    }
    this->ptr = shmat(wfd, 0, 0);
    if(ptr == (void*)(-1)) {
        perror("shmat failure\n");
        exit(1);
    }
    this->hasred = (int*)ptr;
    *hasred = 0;
    this->pwri = hasred + 1; this->size = hasred + 2;
    this->pred = hasred + 3; this->array = (T*)(hasred+4);
    *size = number, *pwri = 0, *pred = 0;
    printf("shm built succeed\n");
}

template <class T>
wriShm<T>::~wriShm() {
    if(shmdt(ptr) == -1) {
        perror("shmdt shm failure\n");
        exit(1);
    }
    if(shmctl(wfd, IPC_RMID, NULL) == -1) {
        perror("shmctl failure\n");
        exit(1);
    }
    printf("shm destoryed\n");
}

template <class T>
bool wriShm<T>::write(const T &i) {
    if(())
}