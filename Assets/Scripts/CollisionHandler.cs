using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;

public class CollisionHandler : MonoBehaviour {
	public GameObject parameter_object;
	Dictionary<string, string> parameters;
	private TcpClient client;
	public string host = "localhost";
	public int port = 12348; 
	private Stream stream;
	public float collision_radius;
	void Start () {
		parameters = parameter_object.GetComponent<LoadParameters>().GetParameters();
		SphereCollider collider = GetComponent<SphereCollider> ();
		collision_radius = float.Parse(parameters ["collision_radius"]);
		collider.radius = collision_radius;
		Connect ();
	}

	void Connect() {
		try {
			client = new TcpClient ();
			client.Connect (host, port);
			stream = client.GetStream (); 
		} catch(Exception ex) {
			Debug.Log (ex.Message);
		}
	}

	void SendPosition() {
		byte [] xBytes = BitConverter.GetBytes (transform.position.x);
		byte [] yBytes = BitConverter.GetBytes (transform.position.y);
		byte [] zBytes = BitConverter.GetBytes (transform.position.z);

		stream.Write(xBytes, 0, 4);
		stream.Write(yBytes, 0, 4);
		stream.Write(zBytes, 0, 4);
	}

	public void CollisionHappened() {
		if (transform.position.y > 0.5) {
			if (client.Connected) {
				SendPosition();
			} else {
				Connect ();
			}	
		}
	}
	void OnCollisionEnter (Collision col) {
		CollisionHappened ();
	}

	void Update() {
		if (!client.Connected) {
			Connect ();
		}
	}
}
