using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;

public class QuadrotorDynamics : MonoBehaviour {
	public float thrust;
	public Rigidbody rb;
	
	public string host = "localhost";
    public int port = 12346; 
    public byte[] timestamp;
    public bool timestamp_valid;
    private UdpClient udp_client;
    private IPEndPoint ip_endpoint;
    public GameObject vehicle;
	
	const int POSE_LENGTH = 7;
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
	}
	
	float[] GrabPose() {
        byte[] poseBytes = null;
        Debug.Log ("Trying to grab pose");
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

	float [,] MatrixTranspose(float[,] Matrix){
		float[,] returnMatrix=new float[,]{{Matrix[0,0], Matrix[1,0],Matrix[2,0]},{Matrix[0,1], Matrix[1,1],Matrix[2,1]},{Matrix[0,2], Matrix[1,2],Matrix[2,2]}};
		return returnMatrix;
	}

	float [,] MatrixSubtraction(float[,] MatrixA,float[,] MatrixB) {

		float[,]MatrixC=new float[,]{{MatrixA[0,0]-MatrixB[0,0],MatrixA[0,1]-MatrixB[0,1],MatrixA[0,2]-MatrixB[0,2]},{MatrixA[1,0]-MatrixB[1,0],MatrixA[1,1]-MatrixB[1,1],MatrixA[1,2]-MatrixB[1,2]},{MatrixA[2,0]-MatrixB[2,0],MatrixA[2,1]-MatrixB[2,1],MatrixA[2,2]-MatrixB[2,2]}};
		return MatrixC;
	}

	float [,] MatrixMultiplication(float[,] MatrixA,float[,] MatrixB) {

		float[,]Matrix3=new float[,]{{0,0,0},{0,0,0},{0,0,0}};
		Matrix3[0,0]=MatrixA[0,0]*MatrixB[0,0]+MatrixA[0,1]*MatrixB[1,0]+MatrixA[0,2]*MatrixB[2,0];
		Matrix3[0,1]=MatrixA[0,0]*MatrixB[0,1]+MatrixA[0,1]*MatrixB[1,1]+MatrixA[0,2]*MatrixB[2,1];
		Matrix3[0,2]=MatrixA[0,0]*MatrixB[0,2]+MatrixA[0,1]*MatrixB[1,2]+MatrixA[0,2]*MatrixB[2,2];
		Matrix3[1,0]=MatrixA[1,0]*MatrixB[0,0]+MatrixA[1,1]*MatrixB[1,0]+MatrixA[1,2]*MatrixB[2,0];
		Matrix3[1,1]=MatrixA[1,0]*MatrixB[0,1]+MatrixA[1,1]*MatrixB[1,1]+MatrixA[1,2]*MatrixB[2,1];
		Matrix3[1,2]=MatrixA[1,0]*MatrixB[0,2]+MatrixA[1,1]*MatrixB[1,2]+MatrixA[1,2]*MatrixB[2,2];
		Matrix3[2,0]=MatrixA[2,0]*MatrixB[0,0]+MatrixA[2,1]*MatrixB[1,0]+MatrixA[2,2]*MatrixB[2,0];
		Matrix3[2,1]=MatrixA[2,0]*MatrixB[0,1]+MatrixA[2,1]*MatrixB[1,1]+MatrixA[2,2]*MatrixB[2,1];
		Matrix3[2,2]=MatrixA[2,0]*MatrixB[0,2]+MatrixA[2,1]*MatrixB[1,2]+MatrixA[2,2]*MatrixB[2,2];

		return Matrix3;
	}

	Vector3 MatrixVectorMultiplication(float[,] Matrix,Vector3 Vector) {

		Vector3 ReturnVector = new Vector3 (0, 0, 0);
		ReturnVector.x = Matrix [0, 0] * Vector.x + Matrix [0, 1] * Vector.y + Matrix [0, 2] * Vector.z;
		ReturnVector.y = Matrix [1, 0] * Vector.x + Matrix [1, 1] * Vector.y + Matrix [1, 2] * Vector.z;
		ReturnVector.z = Matrix [2, 0] * Vector.x + Matrix [2, 1] * Vector.y + Matrix [2, 2] * Vector.z;
		return ReturnVector;
	}

	float VectorVectorMultiplication(Vector3 VectorA,Vector3 VectorB){

		float Result = VectorA.x * VectorB.x + VectorA.y * VectorB.y + VectorA.z * VectorB.z;
		return Result;
	}

	float [,] MatrixScalarMultiplication(float[,] MatrixA,float Scalar) {

		float[,] returnMatrix = { {Scalar*MatrixA[0,0],Scalar*MatrixA[0,1],Scalar*MatrixA[0,2]}, {Scalar*MatrixA[1,0],Scalar*MatrixA[1,1],Scalar*MatrixA[1,2]}, {Scalar*MatrixA[2,0],Scalar*MatrixA[2,1],Scalar*MatrixA[2,2]} };
		return returnMatrix;
	}

	Vector4 Controller(float[] pose, Rigidbody rb){


		float m = rb.mass;
		float g = (float)9.81;
		
		float k_x=(float)10;
		float k_v=(float)12;
		float k_R=(float)18;
		float k_omega=(float)-3.5;

		Vector3 P_temp = rb.transform.position;
		Vector3 P = new Vector3( P_temp.x, P_temp.z, P_temp.y );
		Vector3 v = rb.velocity;
		v = new Vector3( v.x, v.z, v.y );

		Vector3 rpy=rb.transform.eulerAngles;
		rpy = new Vector3 (-rpy.x/180*Mathf.PI, -rpy.z/180*Mathf.PI, -rpy.y/180*Mathf.PI);
		//Debug.Log ("Euler Angles are: "+ rpy);

		Vector3 P_d = new Vector3 (pose [0], pose [1], pose [2]);
		Vector3 v_d = new Vector3 (0, 0, 0);
		Vector3 a_d = new Vector3 (0, 0, 0);

		Vector3 omega_d = new Vector3 (0, 0, 0);
		Vector3 dot_omega_d = new Vector3 (0, 0, 0);


		Vector3 e_x = P - P_d;
		//Debug.Log ("Position error is: "+ e_x.x + " "+ e_x.y + " "+ e_x.z + " ");
		Vector3 e_v = v - v_d;

		Vector3 b1_d = new Vector3 (1, 0, 0);

		Vector3 temp = -k_x * e_x - k_v * e_v - m * g * new Vector3 (0, 0, -1) + m * a_d;
		Vector3 b3_d;
		if (temp.magnitude < 1/10) {
			b3_d = (-k_x * e_x - k_v * e_v - m * g * new Vector3 (0, 0, -1) + m * a_d) / (float)0.1;
		}
		else{
			temp = -k_x * e_x - k_v * e_v - m * g * new Vector3 (0, 0, -1) + m * a_d;
			b3_d=(-k_x * e_x - k_v * e_v - m * g * new Vector3(0, 0, -1) + m * a_d) / temp.magnitude;
		}

		Vector3 b2_d_nom=Vector3.Cross(b3_d,b1_d);
		Vector3 b2_d=Vector3.Cross(b3_d,b1_d)/b2_d_nom.magnitude;

		float[,] R_d= (new float[,]{{b1_d[0], b1_d[1], b1_d[2]},{b2_d[0], b2_d[1], b2_d[2]},{b3_d[0], b3_d[1], b3_d[2]}});
		//float[3,3] R_d=[b1_d b2_d b3_d]';

		float[,] Rx = MatrixTranspose(new float[,]{ { 1, 0, 0 }, { 0, Mathf.Cos(rpy.x), -Mathf.Sin (rpy.x) }, {0, Mathf.Sin(rpy.x), Mathf.Cos(rpy.x) } });
		float[,] Ry = MatrixTranspose(new float[,]{ { Mathf.Cos (rpy.y), 0, Mathf.Sin (rpy.y) }, { 0, 1, 0 }, { -Mathf.Sin (rpy.y), 0, Mathf.Cos (rpy.y) } });
		float[,] Rz = MatrixTranspose(new float[,]{ { Mathf.Cos (rpy.z), -Mathf.Sin (rpy.z), 0 }, { Mathf.Sin (rpy.z), Mathf.Cos (rpy.z), 0 }, { 0, 0, 1 } });

		//float[,]R=new float[,]{{1,0,0},{0,1,0},{0,0,1}};
		float[,]R=MatrixMultiplication(Rx,MatrixMultiplication(Ry,Rz));
		float[,]J=new float[,]{{1,0,0},{0,1,0},{0,0,1}};
		Vector3 omega = rb.angularVelocity;
		omega = new Vector3 (-omega.x, -omega.z, -omega.y);
		//Debug.Log ("RollPitchYaw: " + rpy);
		//Debug.Log ("Desired Orientation is: [" + R_d [0, 0] + " " + R_d [0, 1] + " " + R_d [0, 2] + "], [" + R_d [1, 0] + " " + R_d [1, 1] + " " + R_d [1, 2] + "], [" + R_d [2, 0] + " " + R_d [2, 1] + " " + R_d [2, 2] + "]");
		//Debug.Log ("Current Orientation is: [" + R [0, 0] + " " + R [0, 1] + " " + R [0, 2] + "], [" + R [1, 0] + " " + R [1, 1] + " " + R [1, 2] + "], [" + R [2, 0] + " " + R [2, 1] + " " + R [2, 2] + "]");

		//float[,] ttt = MatrixSubtraction(MatrixMultiplication(MatrixTranspose(R_d),R),MatrixMultiplication(MatrixTranspose(R),R_d));
		//Debug.Log ("Matrix test: [" + ttt [0, 0] + " " + ttt [0, 1] + " " + ttt [0, 2] + "], [" + ttt [1, 0] + " " + ttt [1, 1] + " " + ttt [1, 2] + "], [" + ttt [2, 0] + " " + ttt [2, 1] + " " + ttt [2, 2] + "]");
		float[,]E_R=MatrixScalarMultiplication(MatrixSubtraction(MatrixMultiplication(MatrixTranspose(R_d),R),MatrixMultiplication(MatrixTranspose(R),R_d)),(float)0.5);
		//Debug.Log ("Orientation error is: [" + E_R [0, 0] + " " + E_R [0, 1] + " " + E_R [0, 2] + "], [" + E_R [1, 0] + " " + E_R [1, 1] + " " + E_R [1, 2] + "], [" + E_R [2, 0] + " " + E_R [2, 1] + " " + E_R [2, 2] + "]");
		Vector3 e_R=new Vector3(E_R[2,1],E_R[0,2],E_R[1,0]);
		//Debug.Log ("Orientation error is: " + e_R);
		Vector3 e_omega=omega-MatrixVectorMultiplication(MatrixTranspose(R),MatrixVectorMultiplication(R_d,omega_d));
	
		float[,] omega_h={{0, -omega.z, omega.y},{omega.z, 0, -omega.x},{-omega.y, omega.x, 0}};



		float f=-VectorVectorMultiplication((k_x*e_x+k_v*e_v-m*g*(new Vector3(0,0,-1))+m*a_d),MatrixVectorMultiplication(R,new Vector3(0,0,1)));
		//Debug.Log ("Force f_1_1 is: " + k_x * e_x);
		//Debug.Log ("Force f_1_2 is: " + k_v * e_v);
		//Debug.Log ("Force f_1_3 is: " + m * g * (new Vector3(0,0,-1)));
		//Debug.Log ("Force f_1_4 is: " + m * a_d);
		//Debug.Log ("Force f_1 is: "+ (k_x * e_x + k_v * e_v + m * g * (new Vector3(0,0,-1))+m*a_d));
		//Debug.Log ("Force f_2 is: "+ MatrixVectorMultiplication(R,new Vector3(0,0,1)));
		//Debug.Log ("Orientation error is: "+ e_R);

		Vector3 M = -k_R * e_R - k_omega * e_omega;//-Vector3.Cross(omega,MatrixVectorMultiplication(J,omega))-J*(omega_h*R'*R_d*omega_d-R'*R_d*dot_omega_d);
		//Debug.Log ("Force: "+ f);
		//Debug.Log (" Torque: "+ M);

		Vector4 Result = new Vector4(M.x,M.y,M.z,f);
		return Result;

	}

	// Update is called once per frame
	void Update () {
		//float[] pose = GrabPose();
		//Debug.Log ("Pose and Timestamp Updated: "+pose[0]);
		//vehicle.transform.position = new Vector3 (108, 26, 108);
		//Vector3 Torque = new Vector3 (TorqueForce.w, TorqueForce.x, TorqueForce.y);
		//rb.AddTorque (Torque);

        //if(pose != null) {
        //    SetVehiclePose(pose);
        //    //Debug.Log ("Pose and Timestamp Updated");

        //}
	}
	void FixedUpdate () {
		
		//float[] pose ={106, 106, 28};
		float[] pose = GrabPose();
		Vector4 TorqueForce = Controller(pose,rb);
		rb.AddForce (transform.up * TorqueForce.w);
		Vector3 Torque = new Vector3 (TorqueForce.x, TorqueForce.z, TorqueForce.y);
		rb.AddTorque (Torque);

	}
}
