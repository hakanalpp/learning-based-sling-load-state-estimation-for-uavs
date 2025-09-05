using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Threading;
using UnityEngine;

public class TCPServer : MonoBehaviour
{
    TcpListener server;
    Dictionary<string, ICommandable> commandables = new Dictionary<string, ICommandable>();

    private object syncLock = new object();
    private object floatDataLock = new object();

    public Transform ghostBox;

    public Transform cargoConnection;
    public Transform drone;

    public Transform camera;

    protected string host = "localhost";
    protected int port = 9998;
    protected int portAI = 10001;
    protected int floatDataPort = 9997;
    protected float connectionTimeoutSeconds = 1.0f; // Timeout for connection attempts

    private Thread tcpCommandListenerThread;
    private Thread floatDataListenerThread;
    private TcpClient connectedTcpClient;

    private TcpClient client;
    private Stream stream;

    private TcpClient clientAI;
    private Stream streamAI;

    private Queue<string> messageQueue = new Queue<string>();
    private Queue<float[]> floatDataQueue = new Queue<float[]>(); // Queue for received float data
    private bool isConnecting = false;

    void StartCommandThread()
    {
        tcpCommandListenerThread = new Thread(new ThreadStart(ListenForCommandConnections));
        tcpCommandListenerThread.IsBackground = true;
        tcpCommandListenerThread.Start();
    }

    void StartFloatDataThread()
    {
        floatDataListenerThread = new Thread(new ThreadStart(ListenForFloatData));
        floatDataListenerThread.IsBackground = true;
        floatDataListenerThread.Start();
    }

    // Method for listening for float data on port 9997
    private void ListenForFloatData()
    {
        try
        {
            TcpListener floatDataListener = new TcpListener(IPAddress.Parse("127.0.0.1"), floatDataPort);
            floatDataListener.Start();
            Debug.Log("Float data server is listening on port " + floatDataPort);
            byte[] buffer = new byte[72]; // Buffer for 15 floats (4 bytes each)

            while (true)
            {
                try
                {
                    using (TcpClient dataClient = floatDataListener.AcceptTcpClient())
                    {
                        Debug.Log("Float data client connected");
                        dataClient.ReceiveTimeout = 5000; // 5 second timeout

                        using (NetworkStream stream = dataClient.GetStream())
                        {
                            int bytesRead;
                            while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) == 32) // Ensure we read all 32 bytes
                            {
                                float[] floatArray = new float[8];
                                for (int i = 0; i < 8; i++)
                                {
                                    floatArray[i] = BitConverter.ToSingle(buffer, i * 4);
                                }

                                // Queue the data for processing on the main thread
                                lock (floatDataLock)
                                {
                                    floatDataQueue.Enqueue(floatArray);
                                }
                            }
                        }
                    }
                }
                catch (SocketException ex)
                {
                    Debug.LogWarning($"Socket exception in float data listener: {ex.Message}");
                    Thread.Sleep(1000);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error in float data listener: {ex.Message}\n{ex.StackTrace}");
                    Thread.Sleep(1000);
                }
            }
        }
        catch (SocketException socketException)
        {
            Debug.LogError($"Fatal float data socket exception: {socketException.ToString()}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Fatal error in float data thread: {ex.Message}\n{ex.StackTrace}");
        }
    }

    // Use coroutines for non-blocking connection attempts
    IEnumerator ConnectAsync()
    {
        if (isConnecting)
            yield break;
        isConnecting = true;
        yield return StartCoroutine(ConnectMainClient());
        yield return StartCoroutine(ConnectAIClient());
        isConnecting = false;
    }

    IEnumerator ConnectMainClient()
    {
        if (client != null && client.Connected)
            yield break;

        yield return StartCoroutine(TryConnectClient(port, (newClient, newStream) =>
        {
            client = newClient;
            stream = newStream;
        }));
    }

    IEnumerator ConnectAIClient()
    {
        if (clientAI != null && clientAI.Connected)
            yield break;

        yield return StartCoroutine(TryConnectClient(portAI, (newClient, newStream) =>
        {
            clientAI = newClient;
            streamAI = newStream;
        }));
    }

    IEnumerator TryConnectClient(int portNum, Action<TcpClient, Stream> onSuccess)
    {
        TcpClient newClient = new TcpClient();

        IAsyncResult result = newClient.BeginConnect(host, portNum, null, null);

        float startTime = Time.time;
        while (!result.IsCompleted)
        {
            if (Time.time - startTime > connectionTimeoutSeconds)
            {
                Debug.Log($"Connection attempt to port {portNum} timed out");
                newClient.Close();
                yield break;
            }

            yield return null;
        }

        try
        {
            newClient.EndConnect(result);

            if (newClient.Connected)
            {
                Stream newStream = newClient.GetStream();
                onSuccess(newClient, newStream);
                Debug.Log($"Connected to port {portNum} successfully");
            }
        }
        catch (Exception ex)
        {
            // Debug.Log($"Error connecting to port {portNum}: {ex.Message}");
            newClient.Close();
        }
    }

    void Start()
    {
        QualitySettings.vSyncCount = 0;  // VSync must be disabled
        Application.targetFrameRate = 100;
        StartCommandThread();
        StartFloatDataThread(); // Start the float data listener thread
        StartCoroutine(ConnectAsync());
    }

    public void SendHeader(uint type, string name, long ticks)
    {
        try
        {
            SendData(BitConverter.GetBytes(0xDEADC0DE));
            SendData(BitConverter.GetBytes(type));
            SendData(BitConverter.GetBytes(ticks));
            SendData(name);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error sending header: {ex.Message}");
            client = null; // Mark for reconnection
        }
    }

    public void SendData(Quaternion data)
    {
        SendData(BitConverter.GetBytes(data.x));
        SendData(BitConverter.GetBytes(data.y));
        SendData(BitConverter.GetBytes(data.z));
        SendData(BitConverter.GetBytes(data.w));
    }

    public void SendData(Vector3 data)
    {
        SendData(BitConverter.GetBytes(data.x));
        SendData(BitConverter.GetBytes(data.y));
        SendData(BitConverter.GetBytes(data.z));
    }

    public void SendData(string data)
    {
        SendData(Encoding.ASCII.GetBytes(data + char.MinValue));
    }

    public void SendData(float data)
    {
        SendData(BitConverter.GetBytes(data));
    }

    public void SendData(int data)
    {
        SendData(BitConverter.GetBytes(data));
    }

    public void SendDataToAI(byte[] data, float[] label)
    {
        if (clientAI == null || !clientAI.Connected)
            return;
        try
        {
            streamAI.Write(data, 0, data.Length);

            byte[] labelByteArray = new byte[label.Length * 4];

            Buffer.BlockCopy(label, 0, labelByteArray, 0, labelByteArray.Length);

            streamAI.Write(labelByteArray, 0, labelByteArray.Length);

            streamAI.Flush();
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Error sending data to AI client: {ex.Message}");
            clientAI = null;
        }
    }

    public void SendData(byte[] data)
    {
        if (client == null || !client.Connected)
            return;
        try
        {
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"Error sending data to main client: {ex.Message}");
            client = null; // Mark for reconnection
        }
    }

    public void Register(string client_name, ICommandable commandable)
    {
        Debug.Log("Commandable: " + client_name + " registered");
        commandables.Add(client_name, commandable);
    }

    private void ListenForCommandConnections()
    {
        try
        {
            TcpListener tcpListener = new TcpListener(IPAddress.Parse("127.0.0.1"), 9999);
            tcpListener.Start();
            Debug.Log("Command server is listening");
            Byte[] bytes = new Byte[1024];
            while (true)
            {
                using (connectedTcpClient = tcpListener.AcceptTcpClient())
                {
                    using NetworkStream stream = connectedTcpClient.GetStream();
                    int length;
                    while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                    {
                        byte[] incomingData = new byte[length];
                        Array.Copy(bytes, 0, incomingData, 0, length);
                        string clientMessage = Encoding.ASCII.GetString(incomingData);
                        lock (syncLock)
                        {
                            messageQueue.Enqueue(clientMessage);
                        }
                    }
                }
            }
        }
        catch (SocketException socketException)
        {
            Debug.Log("SocketException " + socketException.ToString());
        }
    }

    void ProcessMessages()
    {
        lock (syncLock)
        {
            while (messageQueue.Count > 0)
            {
                string message = messageQueue.Dequeue();
                string[] message_words = message.Split(' ');

                if (message_words.Length > 1)
                {
                    string object_name = message_words[0];
                    if (commandables.ContainsKey(object_name))
                    {
                        commandables[object_name].OnCommand(message_words);
                    }
                }
            }
        }
    }

    void ProcessFloatData()
    {
        lock (floatDataLock)
        {
            if (floatDataQueue.Count > 0)
            {
                float[] floatArray = null;

                while (floatDataQueue.Count > 0)
                {
                    floatArray = floatDataQueue.Dequeue();
                }

                if (floatArray != null)
                {
                    ModifyGhostBox(floatArray);
                }
            }
        }
    }

    void ModifyGhostBox(float[] floatArray)
    {
        Vector3 directionVector = new Vector3(floatArray[0], floatArray[1], floatArray[2]);
        directionVector.Normalize();
        float predictedDistance = floatArray[3];
        Quaternion cargoRotation = new Quaternion(floatArray[4], floatArray[5], floatArray[6], floatArray[7]);

        Vector3 directionInWorld = camera.rotation * directionVector;
        directionInWorld.Normalize();
        Quaternion predictedRotation = camera.rotation * cargoRotation;

        Vector3 predictedPosition = drone.position + directionInWorld * predictedDistance;

    }

    void FixedUpdate()
    {
        ProcessMessages();
        ProcessFloatData();
    }

    void Update()
    {
        if (!isConnecting && (client == null || !client.Connected || clientAI == null || !clientAI.Connected))
        {
            // Debug.Log("Detected disconnection, attempting to reconnect...");
            StartCoroutine(ConnectAsync());
        }
    }

    private void OnDestroy()
    {
        if (client != null)
        {
            client.Close();
        }

        if (clientAI != null)
        {
            clientAI.Close();
        }

        if (tcpCommandListenerThread != null && tcpCommandListenerThread.IsAlive)
        {
            tcpCommandListenerThread.Abort();
        }

        if (floatDataListenerThread != null && floatDataListenerThread.IsAlive)
        {
            floatDataListenerThread.Abort();
        }
    }
}