using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SensorSwitch : MonoBehaviour {
	
	public GameObject rgbCameraLeft;
	public GameObject rgbCameraRight;
	public GameObject depthCamera;
	public GameObject capsule;

	// Use this for initialization
	void Start () {
		rgbCameraLeft = GameObject.Find("RGBCameraLeft");
		rgbCameraRight = GameObject.Find("RGBCameraRight");
		depthCamera = GameObject.Find("DepthCamera");
		capsule	= GameObject.Find("Capsule");
		
	}
	
	// Update is called once per frame
	void Update () {
		if (Input.GetKeyDown (KeyCode.Alpha1)){
			rgbCameraLeft.SetActive(!rgbCameraLeft.activeInHierarchy);
			Debug.Log ("toggle" + !rgbCameraLeft.activeInHierarchy);
			Debug.Log ("toggle" + !rgbCameraLeft.activeInHierarchy);
		}
		if (Input.GetKeyDown (KeyCode.Alpha2)){
			rgbCameraRight.SetActive(!rgbCameraRight.activeInHierarchy);
			Debug.Log ("toggle" + !rgbCameraRight.activeInHierarchy);
		}
		if (Input.GetKeyDown (KeyCode.Alpha3)){
			depthCamera.SetActive(!depthCamera.activeInHierarchy);
			Debug.Log ("toggle" + !depthCamera.activeInHierarchy);
		}
		if (Input.GetKeyDown (KeyCode.Alpha4)){
			capsule.SetActive(!capsule.activeInHierarchy);
			Debug.Log ("toggle" + !capsule.activeInHierarchy);
		}
		if (Input.GetKeyDown ("escape")){
			Application.Quit();
			//System.Diagnostics.Process.GetCurrentProcess ().Kill();
		}
		
	}
}