#ifndef CLIENT_H
#define CLIENT_H

#include <QWidget>

#include <QTcpSocket>
#include <QTime>

#include <QThread>

namespace Ui {
class Client;
}

class Client : public QWidget
{
    Q_OBJECT

public:
    explicit Client(QWidget *parent = nullptr);



    ~Client();

private slots:
    void slotReadyRead();
    void slotError(QAbstractSocket::SocketError);
    void slotSendToServer();
    void slotConnected();

private:
    QTcpSocket* tcpSocket;
    quint16 nNextBlockSize;

    QString strHost;
    int nPort;

    const int maxPackeds = 20;
    int numPacked;

    Ui::Client *ui;
};

#endif // CLIENT_H
