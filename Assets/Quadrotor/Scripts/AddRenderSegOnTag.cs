using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Kafasını kopardın bunun schatzi
public class AddRenderSegOnTag : MonoBehaviour {

	Terrain terrain;

	// Use this for initialization
	void Start () {
		List<Vector4> colors = new List<Vector4>();
		colors.Add (Color.grey);
		colors.Add (Color.red);
		colors.Add (Color.yellow);
		colors.Add (Color.green);
		colors.Add (Color.magenta);
		colors.Add (Color.cyan);
		colors.Add (Color.white);
		List<string> objects_tagged = new List<string>();
		//objects_tagged.Add("Road");
		//objects_tagged.Add("Car");
		//objects_tagged.Add("TrafficLight");
		//objects_tagged.Add("Tree");
		//objects_tagged.Add("Landscape");
		//objects_tagged.Add("Human");
		//objects_tagged.Add("Streetlight");

		int i = 0;
		foreach (string _object in objects_tagged){
			var objects = GameObject.FindGameObjectsWithTag(_object);
			var objectCount = objects.Length;
			foreach (var obj in objects) {
				var RSS = obj.AddComponent<RenderSemanticSegmentation>();
				Debug.Log (" Script added to object "+ obj);
				RSS.segmentationColor = colors[i];
			}
			i++;
		}		
	}
	
	// Update is called once per frame
	void Update () {
		
	}
}
