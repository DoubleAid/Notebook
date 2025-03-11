

int create_server_socket(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int (server_fd < 0) {
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    setsocket(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 10) < 0) {
        exit(EXIT_FAILURE);
    }

    return server_fd;
}

void epoll_loop(int server_fd, ThreadPool& pool) {
    int epoll_fd = epoll_create1(0);
    if (epoll_fd < 0) {
        exit(EXIT_FAILURE);
    }
    struct epoll_event event;
    event.data.fd = server_fd;
    return;
}