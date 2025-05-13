using UnityEngine;

public class GodCamera : MonoBehaviour {
    public Transform drone;
    public Vector3 offset = new Vector3(0, 5, -10);
    public float rotationSpeed = 10f;
    public float zoomSpeed = 20f;
    public float minZoom = 2f;  // Minimum distance from the drone
    public float maxZoom = 15f; // Maximum distance from the drone

    private float yaw = 0f;
    private float pitch = 0f;
    private float currentZoom;

    void Start() {
        currentZoom = offset.magnitude; // Start with the initial offset length
    }

    void LateUpdate() {
        if (drone == null) return;

        // Rotate camera when right mouse button is held
        if (Input.GetMouseButton(1)) {
            yaw += Input.GetAxis("Mouse X") * rotationSpeed;
            pitch -= Input.GetAxis("Mouse Y") * rotationSpeed;
        }

        // Zoom in/out with mouse scroll wheel
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        currentZoom -= scroll * zoomSpeed;
        currentZoom = Mathf.Clamp(currentZoom, minZoom, maxZoom); // Limit zoom range

        // Update offset direction and apply zoom
        Vector3 zoomedOffset = offset.normalized * currentZoom;
        Quaternion rotation = Quaternion.Euler(pitch, yaw, 0);
        transform.position = drone.position + rotation * zoomedOffset;

        // Always look at the drone
        transform.LookAt(drone.position);
    }
}

