using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;

public class PoseGrabber : MonoBehaviour {
	public string host = "localhost";
	public int port = 12346; 
	public byte[] timestamp;
	public bool timestamp_valid;
	private UdpClient udp_client;
	private IPEndPoint ip_endpoint;
	public GameObject vehicle;

	const int POSE_LENGTH = 7;
	const int STAMP_LENGTH = 2;
	void Connect() {
		try {
			Debug.Log ("Pose Grabber Setting Up Connection");
			udp_client = new UdpClient(port);
		} catch(Exception ex) {
			Debug.Log (ex.Message);
		}
	}
	
	void Start () {
		Connect ();
		timestamp = new byte[STAMP_LENGTH * sizeof(int)];
		timestamp_valid = false;

	}

	float[] GrabPose() {
		byte[] poseBytes = null;
		//Debug.Log ("Trying to grab pose");
		if (udp_client.Available > 0) {

			Debug.Log ("UDP client available");

			while(udp_client.Available > 0) {
				poseBytes = udp_client.Receive (ref ip_endpoint);
			}

			float[] pose = new float[POSE_LENGTH];
			if (POSE_LENGTH * sizeof(float) + STAMP_LENGTH * sizeof(int) == poseBytes.Length) {
				Array.Copy(poseBytes, timestamp, STAMP_LENGTH * sizeof(int));
				timestamp_valid = true;
				for (int x = 0; x < POSE_LENGTH; x++) {
					pose [x] = BitConverter.ToSingle (poseBytes, sizeof(float) * x + STAMP_LENGTH * sizeof(int));
				}


			} else {
				Debug.Log ("Failed to read a pose from UDP data");
				return null;
			}
			return pose;
		} else {
			Debug.Log ("UDP client not available");
			return null;
		}
	}

	void SetVehiclePose(float[] pose) {
		//GameObject vehicle = transform.parent.gameObject;
		vehicle.transform.position = new Vector3 (pose [0], pose [1], pose [2]);
		vehicle.transform.rotation = new Quaternion(pose [3], pose [4], pose [5], pose[6]);
	}

	void Update () {
		float[] pose = GrabPose();
		if(pose != null) {
			SetVehiclePose(pose);
			Debug.Log ("Pose and Timestamp Updated");
		}
	}
}
