using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomBeaconLocation : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
		Vector3[] randomLocations ={ new Vector3(0f,0f,0f), 
		                             new Vector3(150.2f,35.2f,80f),
									 new Vector3(117.9f,34.5f,64.8f),
									 new Vector3(199.8f,46.3f,64.8f),
									 new Vector3(201.8f,45.9f,75.4f),
									 new Vector3(389.8f,85.27f,52.6f), 
									 new Vector3(339.72f,74.79f,23.01f), 
									 new Vector3(209.5f,48.31f,40.6f), 
									 new Vector3(153.1f,37.7f,58.6f), 
									 new Vector3(219.7f,49f,96.4f), 
									 };
									 
		int randNumb = Random.Range(0,10);
	    this.gameObject.transform.position = randomLocations[randNumb];
        
    }
	

    // Update is called once per frame
    void Update()
    {
        
    }
}
