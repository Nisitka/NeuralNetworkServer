#include "client.h"
#include "ui_client.h"

Client::Client(QWidget *parent) :
    QWidget(parent),
    nNextBlockSize(0),
    ui(new Ui::Client)

{
    ui->setupUi(this);

    this->setWindowTitle("Client");

    strHost = "localhost";
    nPort = 2323;

    tcpSocket = new QTcpSocket(this);

    tcpSocket->connectToHost(strHost, nPort);
    connect(
            tcpSocket, SIGNAL(connected()),
            this, SLOT(slotConnected())
            );
    connect(
            tcpSocket, SIGNAL(readyRead()),
            this, SLOT(slotReadyRead())
            );
    connect(
            tcpSocket, SIGNAL(error(QAbstractSocket::SocketError)),
            this, SLOT(slotError(QAbstractSocket::SocketError))
            );

    ui->textEdit->setReadOnly(true);

    connect(
            ui->pushButton, SIGNAL(clicked()),
            this, SLOT(slotSendToServer())
            );

    ui->progressBar->setRange(0, maxPackeds);
    numPacked = 0;
    ui->progressBar->setValue(numPacked);
}

void Client::slotReadyRead()
{
    QByteArray arry = tcpSocket->readAll();
    // ui->textEdit->append(QTime::currentTime().toString() + " " + "Input: " + arry);

    if (arry.toStdString() == "Server get packed!")
    {
        slotSendToServer();

        numPacked++;
        ui->progressBar->setValue(numPacked);
    }
    else
    {
        numPacked++;
        ui->progressBar->setValue(numPacked);

        numPacked = 0;
        ui->progressBar->setValue(numPacked);

        ui->textEdit->append(QTime::currentTime().toString() + " " + "Input: " + arry);
    }
}

void Client::slotError(QAbstractSocket::SocketError err)
{
    QString strError =
        "Error: " + (err == QAbstractSocket::HostNotFoundError ?
        "The host was not found." :
        err == QAbstractSocket::RemoteHostClosedError ?
        "The remote host is closed." :
        err == QAbstractSocket::ConnectionRefusedError ?
        "The connection was refused." :
        QString(tcpSocket->errorString())
        );
    ui->textEdit->append(strError);
}

void Client::slotSendToServer()
{
    int const w = 150;
    int const l = 100;

    int16_t m[w * l * 3];
    QByteArray arrBlock;
    for (int i=0; i < 150 * 100 * 3; i++)
    {
        m[i] = rand()%256;
        arrBlock.append(m[i]);
    }
    tcpSocket->write(arrBlock.toHex());
}

void Client::slotConnected()
{
    ui->textEdit->append("Received the connected signal");
}

Client::~Client()
{
    delete ui;
}
