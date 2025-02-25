using UnityEngine;

public class CameraSwitcher : MonoBehaviour {
    public Camera mainCamera;   // your default camera
    public Camera followCamera; // the camera following the drone

    void Update() {
        if (Input.GetKeyDown(KeyCode.Alpha1)) { // press 'C' to switch
            bool isMainActive = mainCamera.enabled;
            mainCamera.enabled = !isMainActive;
            followCamera.enabled = isMainActive;
        }
    }
}
