using UnityEngine;
using System.Collections;

public class RenderSemanticSegmentation : MonoBehaviour {
	public Color segmentationColor;
	private MaterialPropertyBlock mat;

	// Use this for initialization
	void Start () {
		mat = new MaterialPropertyBlock();
		mat.SetColor("_SegmentationColor", segmentationColor);

		// Now add this to all of the children as well
		Traverse (gameObject);
	}

	void Traverse(GameObject obj) {
		//Debug.Log (obj.name);
		Renderer rend = obj.GetComponent<Renderer> ();
		if (rend != null) {
			rend.SetPropertyBlock (mat);
		}
		foreach (Transform child in obj.transform) {
			Traverse(child.gameObject);
		}
	}

	// Update is called once per frame
	void Update () {

	}
}