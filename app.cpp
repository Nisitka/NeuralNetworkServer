#include "app.h"

app::app()
{
    client = new Client;
}

void app::start()
{
    client->show();
}
