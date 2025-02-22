using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;

public class QuadrotorDynamics_w : MonoBehaviour {
	float thrust;
	public Rigidbody rb;
	
	public string host = "localhost";
    public int port = 12346; 
    public byte[] timestamp;
    public bool timestamp_valid;
    private UdpClient udp_client;
    private IPEndPoint ip_endpoint;
    public GameObject vehicle;
	public float c_f = (float)0.001;
	public float c_d = (float)0.00001;
	public float arm_length = (float)0.3;

	float[] w_ = { 0, 0, 0, 0 };

	
	const int POSE_LENGTH = 4;
    const int STAMP_LENGTH = 0;
    void Connect() {
        try {
            Debug.Log ("Pose Grabber Setting Up Connection");
            udp_client = new UdpClient(port);
        } catch(Exception ex) {
            Debug.Log (ex.Message);
        }
    }

	// Use this for initialization
	void Start () {
		Connect ();
        timestamp = new byte[STAMP_LENGTH * sizeof(int)];
        timestamp_valid = false;
		rb = GetComponent<Rigidbody> ();
		rb.inertiaTensor = new Vector3 ((float)0.1,(float)0.1,(float) 0.2);
	}
	
	float[] GrabW() {
        byte[] poseBytes = null;
        //Debug.Log ("Trying to grab pose");
        if (udp_client.Available > 0) {

            //Debug.Log ("UDP client available");

            while(udp_client.Available > 0) {
                poseBytes = udp_client.Receive (ref ip_endpoint);
            }

            float[] w = new float[POSE_LENGTH];
			//Debug.Log ("Expected packet length: " + poseBytes.Length + "Actual Value: " + POSE_LENGTH * sizeof(float));
			if (POSE_LENGTH * sizeof(float) == poseBytes.Length) {
                Array.Copy(poseBytes, timestamp, STAMP_LENGTH * sizeof(int));
                timestamp_valid = true;
                for (int x = 0; x < POSE_LENGTH; x++) {
					w [x] = BitConverter.ToSingle (poseBytes, sizeof(float) * x + STAMP_LENGTH * sizeof(int));
                }


            } else {
                Debug.Log ("Failed to read a pose from UDP data");
                return null;
            }
            return w;
        } else {
            Debug.Log ("UDP client not available");
            return null;
        }
    }
		

	Vector3 MatrixVectorMultiplication(float[,] Matrix,Vector3 Vector) {

		Vector3 ReturnVector = new Vector3 (0, 0, 0);
		ReturnVector.x = Matrix [0, 0] * Vector.x + Matrix [0, 1] * Vector.y + Matrix [0, 2] * Vector.z;
		ReturnVector.y = Matrix [1, 0] * Vector.x + Matrix [1, 1] * Vector.y + Matrix [1, 2] * Vector.z;
		ReturnVector.z = Matrix [2, 0] * Vector.x + Matrix [2, 1] * Vector.y + Matrix [2, 2] * Vector.z;
		return ReturnVector;
	}
		

	float [,] GetRotationMatrix(Quaternion q){
		float[,]MatrixR=new float[,]{{0,0,0},{0,0,0},{0,0,0}};
		MatrixR [0, 0] = 1 - 2 * q.y * q.y - 2 * q.z * q.z;
		MatrixR [0, 1] = 2 * q.x * q.y - 2 * q.z * q.w;
		MatrixR [0, 2] = 2 * q.x * q.z + 2 * q.y * q.w;

		MatrixR [1, 0] = 2 * q.x * q.y + 2 * q.z * q.w;
		MatrixR [1, 1] = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
		MatrixR [1, 2] = 2 * q.y * q.z - 2 * q.x * q.w;	

		MatrixR [2, 0] = 2 * q.x * q.z - 2 * q.y * q.w;
		MatrixR [2, 1] = 2 * q.y * q.z + 2 * q.x * q.w;
		MatrixR [2, 2] = 1 - 2 * q.x * q.x - 2 * q.y * q.y;	


		return MatrixR;
	}

	Vector4 Dynamics(Vector4 w){
		float T_0 = c_f * w.x * Mathf.Abs(w.x);
		float T_1 = c_f * w.y * Mathf.Abs(w.y);
		float T_2 = c_f * w.z * Mathf.Abs(w.z);
		float T_3 = c_f * w.w * Mathf.Abs(w.w);
		float Force = (T_0 + T_1 + T_2 + T_3);
		Vector3 Torque_drag= new Vector3(0, c_d * ((w.x * Mathf.Abs(w.x)) - (w.y * Mathf.Abs(w.y)) + (w.z * Mathf.Abs(w.z)) - (w.w * Mathf.Abs(w.w))), 0);
		float d = Mathf.Sqrt (2 * arm_length * arm_length);
		Vector3 Torque_thrust = new Vector3 (d * (T_0 + T_1 - T_2 - T_3), (float)0, d * (-T_0 - T_3 + T_1 + T_2));
		return new Vector4 (Force, Torque_thrust.x + Torque_drag.x, Torque_thrust.y + Torque_drag.y, Torque_thrust.z + Torque_drag.z);

	}
	
	void FixedUpdate () {

		w_ = GrabW();
		Vector4 w = new Vector4 (w_[0], w_[1], w_[2], w_[3]);
		//Debug.Log("W vector is " + w.ToString());

		Vector4 TorqueForce = Dynamics(w);
		
		float[,]R = GetRotationMatrix(rb.transform.rotation);
		Vector3 Torque = new Vector3 (-TorqueForce.y, -TorqueForce.z, -TorqueForce.w);
		//Debug.Log("Torque around z: " + Torque.x.ToString());

		Vector3 TorqueBF = MatrixVectorMultiplication (R, Torque);
		rb.AddForce (transform.up * TorqueForce.x);
		rb.AddTorque (TorqueBF);

	}
}
