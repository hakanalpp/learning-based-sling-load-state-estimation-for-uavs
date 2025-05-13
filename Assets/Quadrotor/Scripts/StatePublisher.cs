using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;

public class StatePublisher : MonoBehaviour {

	public string host = "localhost";
	public int port = 12347; 

	private TcpClient client;
	private Stream stream;

	public GameObject quadrotor;
	public Rigidbody rb;


	void Connect() {
		try {
			client = new TcpClient ();
			client.Connect (host, port);
			stream = client.GetStream (); 
		} catch(Exception ex) {
			// Debug.Log (ex.Message);
		}
	}
		

	void SendPose(List<float> PoseReturn) {
		byte[] poseBytes = new byte[13 * sizeof(float)];

		for(int i = 0; i < 13; i++) {
			int offset = sizeof(float) * i;
			BitConverter.GetBytes(PoseReturn[i]).CopyTo(poseBytes, offset);
		}
		stream.Write (poseBytes, 0, poseBytes.Length);
		Debug.Log ("Pose Sent");
	}

 
    void Start () {
        //quadrotor = GameObject.Find("Quadrotor");
		rb = quadrotor.GetComponent<Rigidbody>();
		Connect ();
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

	Vector3 MatrixVectorMultiplication(float[,] Matrix,Vector3 Vector) {

		Vector3 ReturnVector = new Vector3 (0, 0, 0);
		ReturnVector.x = Matrix [0, 0] * Vector.x + Matrix [0, 1] * Vector.y + Matrix [0, 2] * Vector.z;
		ReturnVector.y = Matrix [1, 0] * Vector.x + Matrix [1, 1] * Vector.y + Matrix [1, 2] * Vector.z;
		ReturnVector.z = Matrix [2, 0] * Vector.x + Matrix [2, 1] * Vector.y + Matrix [2, 2] * Vector.z;
		return ReturnVector;
	}

	float [,] MatrixTranspose(float[,] Matrix){
		float[,] returnMatrix=new float[,]{{Matrix[0,0], Matrix[1,0],Matrix[2,0]},{Matrix[0,1], Matrix[1,1],Matrix[2,1]},{Matrix[0,2], Matrix[1,2],Matrix[2,2]}};
		return returnMatrix;
	}
    
    // Update is called once per frame
    void FixedUpdate () {

		Vector3 quadrotor_position = quadrotor.transform.position;
		Quaternion quadrotor_orientation = quadrotor.transform.rotation;
		Vector3 quadrotor_velocity = rb.velocity;
		Vector3 quadrotor_angular_velocity = rb.angularVelocity;

		float[,]R=GetRotationMatrix(rb.transform.rotation);
		Vector3 quadrotor_angular_velocity_bf = MatrixVectorMultiplication (MatrixTranspose(R), quadrotor_angular_velocity);

		List<float> PoseReturn = new List<float> ();
		PoseReturn.Add (quadrotor_position.x);
		PoseReturn.Add (quadrotor_position.y);
		PoseReturn.Add (quadrotor_position.z);
		PoseReturn.Add (quadrotor_orientation.x);
		PoseReturn.Add (quadrotor_orientation.y);
		PoseReturn.Add (quadrotor_orientation.z);
		PoseReturn.Add (quadrotor_orientation.w);
		PoseReturn.Add (quadrotor_velocity.x);
		PoseReturn.Add (quadrotor_velocity.y);
		PoseReturn.Add (quadrotor_velocity.z);
		PoseReturn.Add (quadrotor_angular_velocity.x);
		PoseReturn.Add (quadrotor_angular_velocity.y);
		PoseReturn.Add (quadrotor_angular_velocity.z);


		if (client.Connected) {
			SendPose (PoseReturn);
		} else {
			Connect ();
		}
		//Debug.Log ("Velocity linear is " + quadrotor_velocity.x + " " + quadrotor_velocity.y + " " +quadrotor_velocity.z);
		//Debug.Log ("Velocity rotational is " + quadrotor_angular_velocity.x + " " + quadrotor_angular_velocity.y + " " +quadrotor_angular_velocity.z);
		//Debug.Log ("Quadrotor orientation is " + quadrotor_orientation.x + " " + quadrotor_orientation.y + " " +quadrotor_orientation.z + " " + quadrotor_orientation.w);
			
    }
		
}
