using UnityEngine;
using System.Collections;

public class SegCamera : MonoBehaviour {
	public Shader shader;

	// Use this for initialization
	void Start () {
		GetComponent<Camera> ().SetReplacementShader (shader, "RenderType");
	}

	// Update is called once per frame
	void Update () {

	}
}