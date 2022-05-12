#ifndef APP_H
#define APP_H

#include "client.h"

class app
{
public:
    app();

    void start();

private:
    Client* client;
};

#endif // APP_H
