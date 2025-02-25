using UnityEngine;

public class GodCamera : MonoBehaviour {
    public Transform drone;  // Assign the drone's transform in the Inspector
    public Vector3 offset = new Vector3(0, 5, -10);  // Adjust as needed

    void LateUpdate() {
        if (drone != null) {
            transform.position = drone.position + offset;
            transform.LookAt(drone);
        }
    }
}
