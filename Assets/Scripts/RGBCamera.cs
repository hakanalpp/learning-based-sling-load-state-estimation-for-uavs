using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

public class RGBCamera : ICommandable
{
    public bool recording = false;
    public bool send_image_to_ros = false;

    RenderTexture cameraImage;
    public RenderTexture outputImage;
    Texture2D myTexture2D;
    public Camera camera;
    public Transform cargoBox;
    Rigidbody rb;

    public Transform connectionPoint;
    public int width = 224;
    public int height = 224;
    public float fieldOfView = 90.0f;
    public float farClipPlane = 65.535f;
    public bool monochrome = false;
    protected RenderTextureReadWrite render_texture_read_write = RenderTextureReadWrite.Default;
    public string folderPath = "/home/alp/noetic_ws/src/simulation/images";
    public string csvFileName = "cargo_data.csv";
    private string csvPath;
    private string runFolderPath;
    private int runNumber = 0;

    private ConcurrentQueue<ImageData> imageQueue = new ConcurrentQueue<ImageData>();
    private ConcurrentQueue<string> csvQueue = new ConcurrentQueue<string>();
    private Thread ioThread;
    private bool threadRunning = true;
    private StreamWriter csvWriter;

    // Synchronization data structure
    private struct FrameData
    {
        public long timestamp;
        public Vector3 dronePos;
        public Vector3 droneVel;
        public Vector3 cargoPos;
        public Vector3 cargoVel;
        public Quaternion cargoRot;
        public Quaternion droneRot;
        public Quaternion cameraRot;
        public bool isValid;
    }

    private FrameData pendingFrameData;
    private bool frameDataReady = false;
    private object frameLock = new object();

    struct ImageData
    {
        public byte[] data;
        public string filename;
    }

    void Awake()
    {
        Initialize();

        if (recording)
        {
            Directory.CreateDirectory(folderPath);
            string[] existingRuns = Directory.GetDirectories(folderPath, "run_*");
            runNumber = existingRuns.Length;
            string runId = $"run_{runNumber}";
            runFolderPath = Path.Combine(folderPath, runId);
            Directory.CreateDirectory(runFolderPath);

            csvPath = Path.Combine(runFolderPath, csvFileName);
            csvWriter = new StreamWriter(csvPath, false);
            csvWriter.WriteLine(
                "frameId,drone_pos_x,drone_pos_y,drone_pos_z," +
                "camera_rot_x,camera_rot_y,camera_rot_z,camera_rot_w," +
                "drone_rot_x,drone_rot_y,drone_rot_z,drone_rot_w," +
                "drone_vel_x,drone_vel_y,drone_vel_z," +
                "cargo_pos_x,cargo_pos_y,cargo_pos_z," +
                "cargo_rot_x,cargo_rot_y,cargo_rot_z,cargo_rot_w," +
                "cargo_vel_x,cargo_vel_y,cargo_vel_z"
            );

            ioThread = new Thread(ProcessIOQueue);
            ioThread.Start();
        }
    }

    void Initialize()
    {
        rb = gameObject.GetComponentInParent(typeof(Rigidbody)) as Rigidbody;
        if (rb == null)
        {
            Debug.LogError("Failed to find Rigidbody component in parent objects!");
        }

        cameraImage = new RenderTexture(width, height, 24, RenderTextureFormat.DefaultHDR, render_texture_read_write);
        outputImage = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, render_texture_read_write);

        camera ??= GetComponent<Camera>();
        camera.farClipPlane = farClipPlane;
        camera.fieldOfView = fieldOfView;
        camera.depthTextureMode = DepthTextureMode.Depth;
        camera.targetTexture = cameraImage;

        myTexture2D = new Texture2D(cameraImage.width, cameraImage.height, TextureFormat.RGB24, false);
    }

    // USE THIS INSTEAD OF OnPreRender
    void LateUpdate()
    {
        if (recording)
        {
            lock (frameLock)
            {
                // LateUpdate is called after all Update methods but before rendering
                float angleDifference = Quaternion.Angle(rb.rotation, camera.transform.rotation);

                // Print the angle difference
                Debug.Log($"Frame: {Time.frameCount}, Angle Difference: {angleDifference:F2}°");

                pendingFrameData = new FrameData
                {
                    timestamp = time_server.GetTimeNow(),
                    dronePos = rb.position,
                    droneRot = rb.rotation,
                    droneVel = rb.velocity,
                    cargoPos = connectionPoint.position,
                    cargoVel = cargoBox.GetComponent<Rigidbody>().velocity,
                    cargoRot = cargoBox.rotation,
                    cameraRot = camera.transform.rotation,
                    isValid = true
                };
                frameDataReady = true;
            }
        }
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        Graphics.Blit(src, outputImage);

        if (recording && frameDataReady)
        {
            FrameData dataToSave;
            lock (frameLock)
            {
                dataToSave = pendingFrameData;
                frameDataReady = false;
            }

            if (dataToSave.isValid)
            {
                // Capture the image
                RenderTexture.active = outputImage;
                myTexture2D.ReadPixels(new Rect(0, 0, outputImage.width, outputImage.height), 0, 0, false);
                myTexture2D.Apply();

                byte[] bytes = myTexture2D.EncodeToJPG();

                // Use the timestamp from LateUpdate
                imageQueue.Enqueue(new ImageData { data = bytes, filename = $"{dataToSave.timestamp}.jpg" });

                // Create CSV line with the synchronized data
                string csvLine = $"{dataToSave.timestamp}," +
                    $"{dataToSave.dronePos.x},{dataToSave.dronePos.y},{dataToSave.dronePos.z}," +
                    $"{dataToSave.cameraRot.x},{dataToSave.cameraRot.y},{dataToSave.cameraRot.z},{dataToSave.cameraRot.w}," +
                    $"{dataToSave.droneRot.x},{dataToSave.droneRot.y},{dataToSave.droneRot.z},{dataToSave.droneRot.w}," +
                    $"{dataToSave.droneVel.x},{dataToSave.droneVel.y},{dataToSave.droneVel.z}," +
                    $"{dataToSave.cargoPos.x},{dataToSave.cargoPos.y},{dataToSave.cargoPos.z}," +
                    $"{dataToSave.cargoRot.x},{dataToSave.cargoRot.y},{dataToSave.cargoRot.z},{dataToSave.cargoRot.w}," +
                    $"{dataToSave.cargoVel.x},{dataToSave.cargoVel.y},{dataToSave.cargoVel.z}";

                csvQueue.Enqueue(csvLine);

                if (send_image_to_ros)
                {
                    server.SendHeader(1, full_name, time_server.GetTimeNow());
                    SendImageToRos(outputImage);
                }

                SendImageToAI(outputImage);

            }
        }

    }

    protected void SendImageToAI(RenderTexture tex)
    {
        RenderTexture.active = tex;
        myTexture2D.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0, false);
        byte[] imageBytes = myTexture2D.GetRawTextureData();

        Vector3 vec_world = cargoBox.position - rb.position;

        Vector3 vec_local = Quaternion.Inverse(camera.transform.rotation) * vec_world;
        float distance = vec_world.magnitude;
        Vector3 normalized_vec_local = vec_local / distance;

        Rigidbody cargoRb = cargoBox.GetComponent<Rigidbody>();
        Vector3 cargoVelocity = cargoRb.velocity;

        float[] label = new float[]
        {
            normalized_vec_local.x,
            normalized_vec_local.y,
            normalized_vec_local.z,
            distance,
            cargoBox.rotation.x,
            cargoBox.rotation.y,
            cargoBox.rotation.z,
            cargoBox.rotation.w,
            cargoVelocity.x,
            cargoVelocity.y,
            cargoVelocity.z
        };

        server.SendDataToAI(imageBytes, label);
    }
    void ProcessIOQueue()
    {
        while (threadRunning)
        {
            if (csvQueue.TryDequeue(out string csvLine))
            {
                csvWriter.WriteLine(csvLine);
            }

            if (imageQueue.TryDequeue(out ImageData imageData))
            {
                File.WriteAllBytes(Path.Combine(runFolderPath, imageData.filename), imageData.data);
            }

            if (csvQueue.IsEmpty && imageQueue.IsEmpty)
            {
                Thread.Sleep(1);
            }
        }
    }

    protected void SendImageToRos(RenderTexture tex)
    {
        RenderTexture.active = tex;
        myTexture2D.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0, false);
        byte[] imageBytes = myTexture2D.GetRawTextureData();
        server.SendData(monochrome ? 1 : 0);
        server.SendData(fieldOfView);
        server.SendData(farClipPlane);
        server.SendData(tex.width);
        server.SendData(tex.height);
        server.SendData(imageBytes);
    }

    void OnDestroy()
    {
        threadRunning = false;
        if (ioThread != null && ioThread.IsAlive)
        {
            ioThread.Join(1000);
        }
        csvWriter?.Close();
    }
}